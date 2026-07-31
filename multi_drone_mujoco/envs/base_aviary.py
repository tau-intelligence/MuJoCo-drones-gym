"""Base Aviary environment for multi-drone MuJoCo simulation.

Implements the core physics, observation, and rendering logic.
Subclasses define specific tasks (hover, velocity tracking, etc.).
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

# Paths
# CF2 mesh files (.obj) are vendored under multi_drone_mujoco/assets/cf2.
# They were copied from google-deepmind/mujoco_menagerie (bitcraze_crazyflie_2)
# so users no longer need to clone the whole menagerie repo. To use a different
# checkout, set the MJ_DRONES_CF2_ASSETS env var to its assets dir.
ASSETS_PATH = Path(__file__).resolve().parent.parent / "assets"
CF2_MESH_DIR = Path(os.environ.get("MJ_DRONES_CF2_ASSETS",
                                   str(ASSETS_PATH / "cf2")))

################################################################################
# Crazyflie 2.x physical parameters (from Forster 2015 system identification)
################################################################################

DRONE_PARAMS = {
    DroneModel.CF2X: {
        "mass": 0.027,                          # kg
        "arm_length": 0.0397,                   # m (motor-to-center)
        "thrust2weight_ratio": 2.25,
        "ixx": 1.4e-5,                          # kg*m^2
        "iyy": 1.4e-5,
        "izz": 2.17e-5,
        "kf": 3.16e-10,                         # thrust coeff (N/(rad/s)^2)
        "km": 7.94e-12,                         # torque coeff (Nm/(rad/s)^2)
        "prop_radius": 0.02325,                 # m
        "max_speed_kmh": 30.0,
        "gnd_eff_coeff": 11.36859,
        "drag_coeff_xy": 9.1785e-7,
        "drag_coeff_z": 10.311e-7,
        "dw_coeff_1": 2267.18,
        "dw_coeff_2": 0.16,
        "dw_coeff_3": -0.11,
        "collision_h": 0.03,
        "collision_r": 0.06,
        "collision_z_offset": 0.0,
    },
    DroneModel.CF2P: {
        "mass": 0.027,
        "arm_length": 0.0397,
        "thrust2weight_ratio": 2.25,
        "ixx": 1.4e-5,
        "iyy": 1.4e-5,
        "izz": 2.17e-5,
        "kf": 3.16e-10,
        "km": 7.94e-12,
        "prop_radius": 0.02325,
        "max_speed_kmh": 30.0,
        "gnd_eff_coeff": 11.36859,
        "drag_coeff_xy": 9.1785e-7,
        "drag_coeff_z": 10.311e-7,
        "dw_coeff_1": 2267.18,
        "dw_coeff_2": 0.16,
        "dw_coeff_3": -0.11,
        "collision_h": 0.03,
        "collision_r": 0.06,
        "collision_z_offset": 0.0,
    },
    DroneModel.RACE: {
        "mass": 0.250,
        "arm_length": 0.125,
        "thrust2weight_ratio": 4.0,
        "ixx": 4.86e-4,
        "iyy": 4.86e-4,
        "izz": 8.8e-4,
        "kf": 1.28e-8,
        "km": 5.964552e-10,
        "prop_radius": 0.0635,
        "max_speed_kmh": 100.0,
        "gnd_eff_coeff": 11.36859,
        "drag_coeff_xy": 9.1785e-7,
        "drag_coeff_z": 10.311e-7,
        "dw_coeff_1": 2267.18,
        "dw_coeff_2": 0.16,
        "dw_coeff_3": -0.11,
        "collision_h": 0.06,
        "collision_r": 0.15,
        "collision_z_offset": 0.0,
    },
}


def _generate_aviary_xml(
    num_drones: int,
    drone_model: DroneModel,
    init_xyzs: np.ndarray,
    init_rpys: np.ndarray,
    obstacles: bool = False,
    vision: bool = False,
    timestep: float = 1 / 240,
) -> str:
    """Generate MuJoCo XML for the aviary with N drones."""
    meshdir = str(CF2_MESH_DIR)
    params = DRONE_PARAMS[drone_model]

    # Visual and collision meshes
    visual_meshes = "\n".join(
        f'    <mesh file="{meshdir}/cf2_{i}.obj" name="cf2_vis_{i}"/>'
        for i in range(7)
    )
    collision_meshes = "\n".join(
        f'    <mesh file="{meshdir}/cf2_collision_{i}.obj" name="cf2_col_{i}"/>'
        for i in range(32)
    )

    # Drone bodies with 4 propeller sites for force application
    mass = params["mass"]
    ixx, iyy, izz = params["ixx"], params["iyy"], params["izz"]
    L = params["arm_length"]

    drone_bodies = ""
    actuators = ""
    sensors = ""

    for d in range(num_drones):
        x, y, z = init_xyzs[d]
        # Convert RPY to quaternion for initial orientation
        r, p_angle, yaw = init_rpys[d]
        # Simple RPY to quat (small angles)
        cr, sr = np.cos(r / 2), np.sin(r / 2)
        cp, sp = np.cos(p_angle / 2), np.sin(p_angle / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        prefix = f"drone{d}"

        # Propeller positions (X-configuration for CF2X)
        if drone_model == DroneModel.CF2X:
            prop_offsets = [
                (L / np.sqrt(2), L / np.sqrt(2), 0),    # front-left
                (-L / np.sqrt(2), L / np.sqrt(2), 0),   # front-right (corrected)
                (-L / np.sqrt(2), -L / np.sqrt(2), 0),  # rear-right
                (L / np.sqrt(2), -L / np.sqrt(2), 0),   # rear-left
            ]
        elif drone_model == DroneModel.CF2P:
            prop_offsets = [
                (L, 0, 0),     # front
                (0, L, 0),     # left
                (-L, 0, 0),    # rear
                (0, -L, 0),    # right
            ]
        else:  # RACE (same as X)
            prop_offsets = [
                (L / np.sqrt(2), L / np.sqrt(2), 0),
                (-L / np.sqrt(2), L / np.sqrt(2), 0),
                (-L / np.sqrt(2), -L / np.sqrt(2), 0),
                (L / np.sqrt(2), -L / np.sqrt(2), 0),
            ]

        prop_sites = ""
        for pi, (px, py, pz) in enumerate(prop_offsets):
            prop_sites += f'      <site name="{prefix}_prop{pi}" pos="{px} {py} {pz}" group="5"/>\n'

        drone_bodies += f"""
    <body name="{prefix}" pos="{x} {y} {z}" quat="{qw} {qx} {qy} {qz}">
      <freejoint name="{prefix}_joint"/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="{prefix}_collision" type="cylinder" size="{params['collision_r']} {params['collision_h'] / 2}" rgba="0 0 0 0" contype="1" conaffinity="1"/>
      <geom mesh="cf2_vis_0" material="propeller_plastic" class="visual"/>
      <geom mesh="cf2_vis_1" material="medium_gloss_plastic" class="visual"/>
      <geom mesh="cf2_vis_2" material="polished_gold" class="visual"/>
      <geom mesh="cf2_vis_3" material="polished_plastic" class="visual"/>
      <geom mesh="cf2_vis_4" material="burnished_chrome" class="visual"/>
      <geom mesh="cf2_vis_5" material="body_frame_plastic" class="visual"/>
      <geom mesh="cf2_vis_6" material="white" class="visual"/>
      <site name="{prefix}_center" pos="0 0 0" group="5"/>
{prop_sites}"""

        # Add camera for vision
        if vision:
            drone_bodies += f'      <camera name="{prefix}_cam" pos="0.02 0 0" xyaxes="0 -1 0 0 0 1" fovy="60"/>\n'

        drone_bodies += "    </body>\n"

        # Sensors
        sensors += f"""
    <gyro name="{prefix}_gyro" site="{prefix}_center"/>
    <accelerometer name="{prefix}_acc" site="{prefix}_center"/>
    <framequat name="{prefix}_quat" objtype="site" objname="{prefix}_center"/>
    <framepos name="{prefix}_pos" objtype="site" objname="{prefix}_center"/>
    <framelinvel name="{prefix}_vel" objtype="site" objname="{prefix}_center"/>
    <frameangvel name="{prefix}_angvel" objtype="site" objname="{prefix}_center"/>"""

    # Obstacles
    obstacle_bodies = ""
    if obstacles:
        obstacle_bodies = """
    <body name="obstacle_box" pos="0.5 0.5 0.3">
      <geom type="box" size="0.1 0.1 0.3" rgba="0.8 0.2 0.2 1"/>
    </body>
    <body name="obstacle_sphere" pos="-0.5 0.5 0.5">
      <geom type="sphere" size="0.1" rgba="0.2 0.8 0.2 1"/>
    </body>
    <body name="obstacle_cylinder" pos="0 -0.5 0.4">
      <geom type="cylinder" size="0.05 0.3" rgba="0.2 0.2 0.8 1"/>
    </body>"""

    xml = f"""<mujoco model="aviary_{num_drones}x_{drone_model.value}">
  <option integrator="RK4" density="1.225" viscosity="1.8e-5" timestep="{timestep}"/>
  <compiler inertiafromgeom="false" autolimits="true"/>

  <default>
    <default class="cf2">
      <default class="visual">
        <geom group="2" type="mesh" contype="0" conaffinity="0"/>
      </default>
    </default>
  </default>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <material name="polished_plastic" rgba="0.631 0.659 0.678 1"/>
    <material name="polished_gold" rgba="0.969 0.878 0.6 1"/>
    <material name="medium_gloss_plastic" rgba="0.109 0.184 0.0 1"/>
    <material name="propeller_plastic" rgba="0.792 0.820 0.933 1"/>
    <material name="white" rgba="1 1 1 1"/>
    <material name="body_frame_plastic" rgba="0.102 0.102 0.102 1"/>
    <material name="burnished_chrome" rgba="0.898 0.898 0.898 1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
{visual_meshes}
{collision_meshes}
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" size="10 10 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
{drone_bodies}{obstacle_bodies}
  </worldbody>

  <sensor>
{sensors}
  </sensor>
</mujoco>
"""
    return xml


class BaseAviary(gym.Env):
    """Base class for drone aviary environments using MuJoCo.

    Implements full quadrotor dynamics with configurable physics effects:
    - Per-motor RPM → thrust/torque mapping
    - Aerodynamic drag model
    - Ground effect model
    - Downwash between drones
    - Explicit or MuJoCo-integrated dynamics

    Supports multiple action/observation types and N drones.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    ############################################################################

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 1,
        neighbourhood_radius: float = np.inf,
        initial_xyzs: Optional[np.ndarray] = None,
        initial_rpys: Optional[np.ndarray] = None,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 240,
        gui: bool = False,
        record: bool = False,
        obstacles: bool = False,
        vision_attributes: bool = False,
        obs_type: ObservationType = ObservationType.KIN,
        act_type: ActionType = ActionType.RPM,
        output_folder: str = "results",
        render_mode: Optional[str] = None,
    ):
        """Initialize the aviary.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to simulate.
        num_drones : int
            Number of drones in the environment.
        neighbourhood_radius : float
            Radius for adjacency matrix computation (meters).
        initial_xyzs : ndarray, shape (num_drones, 3)
            Initial positions. Defaults to grid layout.
        initial_rpys : ndarray, shape (num_drones, 3)
            Initial roll-pitch-yaw (radians). Defaults to zeros.
        physics : Physics
            Physics engine mode (determines which effects are active).
        sim_freq : int
            Simulation frequency (Hz). Must be a multiple of ctrl_freq.
        ctrl_freq : int
            Control frequency (Hz). Environment steps at this rate.
        gui : bool
            Whether to launch interactive viewer.
        record : bool
            Whether to record video frames.
        obstacles : bool
            Whether to add obstacles to the scene.
        vision_attributes : bool
            Whether to enable per-drone camera rendering.
        obs_type : ObservationType
            Type of observations returned.
        act_type : ActionType
            Type of actions accepted.
        output_folder : str
            Directory for recordings and logs.
        render_mode : str
            Gymnasium render mode ("human" or "rgb_array").
        """
        super().__init__()

        # Constants
        self.G = 9.81
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180

        # Frequencies
        self.SIM_FREQ = sim_freq
        self.CTRL_FREQ = ctrl_freq
        if sim_freq % ctrl_freq != 0:
            raise ValueError("sim_freq must be divisible by ctrl_freq")
        self.SIM_STEPS_PER_CTRL = int(sim_freq / ctrl_freq)
        self.CTRL_TIMESTEP = 1.0 / ctrl_freq
        self.SIM_TIMESTEP = 1.0 / sim_freq

        # Configuration
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        self.DRONE_MODEL = drone_model
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.RECORD = record
        self.GUI = gui
        self.VISION_ATTR = vision_attributes
        self.OBS_TYPE = obs_type
        self.ACT_TYPE = act_type
        self.OUTPUT_FOLDER = output_folder
        self.render_mode = render_mode if render_mode else ("human" if gui else None)

        # Load drone parameters
        params = DRONE_PARAMS[drone_model]
        self.M = params["mass"]
        self.L = params["arm_length"]
        self.THRUST2WEIGHT_RATIO = params["thrust2weight_ratio"]
        self.J = np.diag([params["ixx"], params["iyy"], params["izz"]])
        self.J_INV = np.linalg.inv(self.J)
        self.KF = params["kf"]
        self.KM = params["km"]
        self.PROP_RADIUS = params["prop_radius"]
        self.MAX_SPEED_KMH = params["max_speed_kmh"]
        self.GND_EFF_COEFF = params["gnd_eff_coeff"]
        self.DRAG_COEFF = np.array([params["drag_coeff_xy"], params["drag_coeff_xy"], params["drag_coeff_z"]])
        self.DW_COEFF_1 = params["dw_coeff_1"]
        self.DW_COEFF_2 = params["dw_coeff_2"]
        self.DW_COEFF_3 = params["dw_coeff_3"]
        self.COLLISION_H = params["collision_h"]
        self.COLLISION_R = params["collision_r"]
        self.COLLISION_Z_OFFSET = params["collision_z_offset"]

        # Derived constants
        self.GRAVITY = self.G * self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = 4 * self.KF * self.MAX_RPM ** 2
        if drone_model in (DroneModel.CF2X, DroneModel.RACE):
            self.MAX_XY_TORQUE = (2 * self.L * self.KF * self.MAX_RPM ** 2) / np.sqrt(2)
        else:
            self.MAX_XY_TORQUE = self.L * self.KF * self.MAX_RPM ** 2
        self.MAX_Z_TORQUE = 2 * self.KM * self.MAX_RPM ** 2
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt(
            (15 * self.MAX_RPM ** 2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST
        )

        # Initial positions/orientations
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([
                np.array([x * 4 * self.L for x in range(num_drones)]),
                np.array([y * 4 * self.L for y in range(num_drones)]),
                np.ones(num_drones) * (self.COLLISION_H / 2 - self.COLLISION_Z_OFFSET + 0.1)
            ]).T.reshape(num_drones, 3)
        else:
            self.INIT_XYZS = np.array(initial_xyzs).reshape(num_drones, 3)

        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((num_drones, 3))
        else:
            self.INIT_RPYS = np.array(initial_rpys).reshape(num_drones, 3)

        # Generate and load MuJoCo model
        xml_str = _generate_aviary_xml(
            num_drones=num_drones,
            drone_model=drone_model,
            init_xyzs=self.INIT_XYZS,
            init_rpys=self.INIT_RPYS,
            obstacles=obstacles,
            vision=vision_attributes,
            timestep=self.SIM_TIMESTEP,
        )
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)

        # Vision attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.SIM_FREQ / self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros((num_drones, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
            self.dep = np.ones((num_drones, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.float32)
            self.seg = np.zeros((num_drones, self.IMG_RES[1], self.IMG_RES[0]), dtype=np.int32)

        # Recording
        if self.RECORD:
            self.FRAME_NUM = 0
            self.IMG_PATH = os.path.join(
                self.OUTPUT_FOLDER,
                "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
            )
            os.makedirs(self.IMG_PATH, exist_ok=True)

        # Spaces
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

        # State storage (updated each sim step for performance)
        self.pos = np.zeros((num_drones, 3))
        self.quat = np.zeros((num_drones, 4))
        self.rpy = np.zeros((num_drones, 3))
        self.vel = np.zeros((num_drones, 3))
        self.ang_v = np.zeros((num_drones, 3))
        self.last_clipped_action = np.zeros((num_drones, 4))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((num_drones, 3))

        # Renderer
        self._renderer = None
        self._viewer = None

        # Optional shared-renderer delegation (see multi_drone_mujoco.rendering).
        # When set, _getDroneImages() renders through this object instead of
        # creating a per-env GL context. Only valid for static worlds -- the
        # caller asserts that, it is never inferred. None => original behaviour.
        self._external_renderer = None
        # Segmentation is rendered by default for backward compatibility, but it
        # is a full extra render pass and callers that discard it can opt out.
        self._render_seg = True

        # Wind disturbance (set via set_wind() or subclass)
        self._wind_field = None

        # Counters
        self.step_counter = 0
        self.RESET_TIME = time.time()

        # Do initial forward pass
        mujoco.mj_forward(self.model, self.data)
        self._updateAndStoreKinematicInformation()

    ############################################################################

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Set initial states
        for i in range(self.NUM_DRONES):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_joint")
            qpos_addr = self.model.jnt_qposadr[joint_id]
            # Position
            self.data.qpos[qpos_addr:qpos_addr + 3] = self.INIT_XYZS[i]
            # Quaternion from RPY
            r, p_a, y = self.INIT_RPYS[i]
            cr, sr = np.cos(r / 2), np.sin(r / 2)
            cp, sp = np.cos(p_a / 2), np.sin(p_a / 2)
            cy, sy = np.cos(y / 2), np.sin(y / 2)
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            self.data.qpos[qpos_addr + 3:qpos_addr + 7] = [qw, qx, qy, qz]

        mujoco.mj_forward(self.model, self.data)

        # Reset state
        self.step_counter = 0
        self.RESET_TIME = time.time()
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))

        self._updateAndStoreKinematicInformation()

        return self._computeObs(), self._computeInfo()

    ############################################################################

    def step(self, action):
        """Advance the environment by one control step.

        Parameters
        ----------
        action : ndarray
            Action array, format depends on self.ACT_TYPE.
        """
        # Preprocess action to RPMs
        clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))

        # Simulation sub-steps
        for _ in range(self.SIM_STEPS_PER_CTRL):
            # Update kinematics between sub-steps for physics effects
            if self.SIM_STEPS_PER_CTRL > 1 and self.PHYSICS in (
                Physics.DYN, Physics.MJC_GND, Physics.MJC_DRAG,
                Physics.MJC_DW, Physics.MJC_GND_DRAG_DW
            ):
                self._updateAndStoreKinematicInformation()

            # Apply physics for each drone
            for i in range(self.NUM_DRONES):
                if self.PHYSICS == Physics.MJC:
                    self._physics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.MJC_GND:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                elif self.PHYSICS == Physics.MJC_DRAG:
                    self._physics(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                elif self.PHYSICS == Physics.MJC_DW:
                    self._physics(clipped_action[i, :], i)
                    self._downwash(i)
                elif self.PHYSICS == Physics.MJC_GND_DRAG_DW:
                    self._physics(clipped_action[i, :], i)
                    self._groundEffect(clipped_action[i, :], i)
                    self._drag(self.last_clipped_action[i, :], i)
                    self._downwash(i)

                # Apply wind disturbance if configured
                if self._wind_field is not None:
                    body_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{i}")
                    wind_force = self._wind_field.get_force(
                        dt=self.SIM_TIMESTEP, position=self.pos[i], velocity=self.vel[i])
                    self.data.xfrc_applied[body_id, :3] += wind_force

            # Step MuJoCo (unless using explicit dynamics)
            if self.PHYSICS != Physics.DYN:
                mujoco.mj_step(self.model, self.data)

            self.last_clipped_action = clipped_action

        # Update kinematic state
        self._updateAndStoreKinematicInformation()
        self.step_counter += self.SIM_STEPS_PER_CTRL

        # Record frames
        if self.RECORD and self.step_counter % max(1, int(self.SIM_FREQ / 24)) == 0:
            self._saveFrame()

        # Compute returns
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        return obs, reward, terminated, truncated, info

    ############################################################################
    # PHYSICS IMPLEMENTATIONS
    ############################################################################

    def _physics(self, rpm, nth_drone):
        """Apply forces/torques from RPMs using MuJoCo's external force API.

        Parameters
        ----------
        rpm : ndarray (4,)
            RPMs of the 4 motors.
        nth_drone : int
            Drone index.
        """
        forces = np.array(rpm ** 2) * self.KF
        torques = np.array(rpm ** 2) * self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques

        # Net z-torque from motor reaction torques
        z_torque = -torques[0] + torques[1] - torques[2] + torques[3]

        # Total thrust (all motors push up in body frame)
        total_thrust = np.sum(forces)

        # Torques from thrust differences
        if self.DRONE_MODEL == DroneModel.CF2X or self.DRONE_MODEL == DroneModel.RACE:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
        else:  # CF2P
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L

        # Apply force and torque to drone body via xfrc_applied
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{nth_drone}")

        # Get rotation matrix for body frame
        rot_mat = np.array(self.data.xmat[body_id]).reshape(3, 3)

        # Thrust in world frame (body z-axis * total_thrust)
        thrust_world = rot_mat @ np.array([0, 0, total_thrust])

        # Torques in world frame
        torque_body = np.array([x_torque, y_torque, z_torque])
        torque_world = rot_mat @ torque_body

        # Apply via xfrc_applied [fx, fy, fz, tx, ty, tz]
        self.data.xfrc_applied[body_id, :3] = thrust_world
        self.data.xfrc_applied[body_id, 3:] = torque_world

    def _groundEffect(self, rpm, nth_drone):
        """Ground effect model (Shi et al., 2019).

        Increases thrust when propellers are close to the ground.
        """
        z = self.pos[nth_drone, 2]
        prop_heights = np.full(4, z)  # Approximate all props at same height
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)

        gnd_effects = np.array(rpm ** 2) * self.KF * self.GND_EFF_COEFF * (
            self.PROP_RADIUS / (4 * prop_heights)
        ) ** 2

        # Only apply if drone is roughly level
        if np.abs(self.rpy[nth_drone, 0]) < np.pi / 2 and np.abs(self.rpy[nth_drone, 1]) < np.pi / 2:
            gnd_force = np.sum(gnd_effects)
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{nth_drone}")
            self.data.xfrc_applied[body_id, 2] += gnd_force  # Add to z-force in world frame

    def _drag(self, rpm, nth_drone):
        """Drag model based on Forster (2015) system identification."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{nth_drone}")
        rot_mat = np.array(self.data.xmat[body_id]).reshape(3, 3)

        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))
        drag = rot_mat @ (drag_factors * (rot_mat.T @ self.vel[nth_drone, :]))

        self.data.xfrc_applied[body_id, :3] += drag

    def _downwash(self, nth_drone):
        """Downwash effect between drones (Zhou, DSL).

        A drone below another drone experiences a downward force.
        """
        for i in range(self.NUM_DRONES):
            if i == nth_drone:
                continue
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(self.pos[i, 0:2] - self.pos[nth_drone, 0:2])
            if delta_z > 0 and delta_xy < 10:
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS / (4 * delta_z)) ** 2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash_force = -alpha * np.exp(-0.5 * (delta_xy / beta) ** 2)

                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{nth_drone}")
                self.data.xfrc_applied[body_id, 2] += downwash_force

    def _dynamics(self, rpm, nth_drone):
        """Explicit dynamics (bypasses MuJoCo physics step).

        Based on the explicit integration in gym-pybullet-drones by James Xu.
        """
        pos = self.pos[nth_drone, :]
        quat = self.quat[nth_drone, :]
        vel = self.vel[nth_drone, :]
        rpy_rates = self.rpy_rates[nth_drone, :]

        # Rotation matrix from quaternion
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"drone{nth_drone}")
        rotation = np.array(self.data.xmat[body_id]).reshape(3, 3)

        # Forces and torques
        forces = np.array(rpm ** 2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = rotation @ thrust
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])

        z_torques = np.array(rpm ** 2) * self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            z_torques = -z_torques
        z_torque = -z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3]

        if self.DRONE_MODEL in (DroneModel.CF2X, DroneModel.RACE):
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L / np.sqrt(2))
            y_torque = (-forces[0] + forces[1] + forces[2] - forces[3]) * (self.L / np.sqrt(2))
        else:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L

        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, self.J @ rpy_rates)
        rpy_rates_deriv = self.J_INV @ torques
        accel = force_world_frame / self.M

        # Integrate
        vel_new = vel + self.SIM_TIMESTEP * accel
        rpy_rates_new = rpy_rates + self.SIM_TIMESTEP * rpy_rates_deriv
        pos_new = pos + self.SIM_TIMESTEP * vel_new
        quat_new = self._integrateQ(quat, rpy_rates_new, self.SIM_TIMESTEP)

        # Set state in MuJoCo
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{nth_drone}_joint")
        qpos_addr = self.model.jnt_qposadr[joint_id]
        qvel_addr = self.model.jnt_dofadr[joint_id]

        self.data.qpos[qpos_addr:qpos_addr + 3] = pos_new
        self.data.qpos[qpos_addr + 3:qpos_addr + 7] = quat_new
        self.data.qvel[qvel_addr:qvel_addr + 3] = vel_new
        self.data.qvel[qvel_addr + 3:qvel_addr + 6] = rotation @ rpy_rates_new

        self.rpy_rates[nth_drone, :] = rpy_rates_new
        mujoco.mj_forward(self.model, self.data)

    def _integrateQ(self, quat, omega, dt):
        """Integrate quaternion with angular velocity."""
        omega_norm = np.linalg.norm(omega)
        if omega_norm < 1e-10:
            return quat
        p, q, r = omega
        lambda_ = np.array([
            [0, r, -q, p],
            [-r, 0, p, q],
            [q, -p, 0, r],
            [-p, -q, -r, 0],
        ]) * 0.5
        theta = omega_norm * dt / 2
        quat_new = (np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta)) @ quat
        return quat_new / np.linalg.norm(quat_new)

    ############################################################################
    # STATE ACCESS
    ############################################################################

    def _updateAndStoreKinematicInformation(self):
        """Read and cache drone states from MuJoCo data."""
        for i in range(self.NUM_DRONES):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"drone{i}_joint")
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qvel_addr = self.model.jnt_dofadr[joint_id]

            self.pos[i] = self.data.qpos[qpos_addr:qpos_addr + 3]
            self.quat[i] = self.data.qpos[qpos_addr + 3:qpos_addr + 7]
            self.vel[i] = self.data.qvel[qvel_addr:qvel_addr + 3]
            self.ang_v[i] = self.data.qvel[qvel_addr + 3:qvel_addr + 6]
            # Quaternion to RPY
            self.rpy[i] = self._quatToRPY(self.quat[i])

        # Clear xfrc_applied for next step
        self.data.xfrc_applied[:] = 0

    def _quatToRPY(self, quat):
        """Convert quaternion [w,x,y,z] to roll-pitch-yaw."""
        w, x, y, z = quat
        # Roll
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        # Yaw
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def _getDroneStateVector(self, nth_drone):
        """Get the 20-dim state vector for a drone.

        Returns: [pos(3), quat(4), rpy(3), vel(3), ang_v(3), last_action(4)]
        """
        return np.hstack([
            self.pos[nth_drone, :],
            self.quat[nth_drone, :],
            self.rpy[nth_drone, :],
            self.vel[nth_drone, :],
            self.ang_v[nth_drone, :],
            self.last_clipped_action[nth_drone, :],
        ]).reshape(20,)

    def _getDroneImages(self, nth_drone):
        """Render RGB, depth, and segmentation from the nth drone's camera.

        Three behaviours, in priority order:

        1. ``self._external_renderer`` set -> delegate to a shared renderer
           (one GL context serving many envs). Static worlds only.
        2. ``self._render_seg`` False -> skip the segmentation render pass and
           return a zero segmentation buffer. Saves a full pass for callers
           that discard it.
        3. otherwise -> original per-env behaviour, unchanged.
        """
        cam_name = f"drone{nth_drone}_cam"
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            return (np.zeros((self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8),
                    np.ones((self.IMG_RES[1], self.IMG_RES[0]), dtype=np.float32),
                    np.zeros((self.IMG_RES[1], self.IMG_RES[0]), dtype=np.int32))

        if self._external_renderer is not None:
            return self._external_renderer.render(
                self.data, cam_name, want_seg=self._render_seg)

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.IMG_RES[1], width=self.IMG_RES[0])

        self._renderer.update_scene(self.data, camera=cam_name)
        rgb = self._renderer.render()

        # Depth rendering. The update_scene here looks redundant (same data,
        # unchanged) -- bench/baseline.py measures what dropping it would save,
        # but it stays until a pixel test proves depth is unaffected.
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=cam_name)
        dep = self._renderer.render()
        self._renderer.disable_depth_rendering()

        if self._render_seg:
            self._renderer.enable_segmentation_rendering()
            self._renderer.update_scene(self.data, camera=cam_name)
            seg = self._renderer.render()
            self._renderer.disable_segmentation_rendering()
        else:
            seg = np.zeros((self.IMG_RES[1], self.IMG_RES[0]), dtype=np.int32)

        return rgb, dep, seg

    ############################################################################
    # ADJACENCY & NEIGHBOURHOOD
    ############################################################################

    def _getAdjacencyMatrix(self):
        """Compute adjacency matrix based on neighbourhood radius."""
        adj = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES - 1):
            for j in range(i + 1, self.NUM_DRONES):
                if np.linalg.norm(self.pos[i] - self.pos[j]) < self.NEIGHBOURHOOD_RADIUS:
                    adj[i, j] = adj[j, i] = 1
        return adj

    ############################################################################
    # ACTION PREPROCESSING
    ############################################################################

    def _preprocessAction(self, action):
        """Convert action to RPMs based on action type."""
        if self.ACT_TYPE == ActionType.RPM:
            return np.clip(np.array(action).reshape(self.NUM_DRONES, 4), 0, self.MAX_RPM)

        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            # Single normalized thrust per drone → same RPM to all 4 motors
            action = np.array(action).reshape(self.NUM_DRONES, 1)
            rpms = np.repeat(self._normalizedActionToRPM(action), 4, axis=1)
            return np.clip(rpms, 0, self.MAX_RPM)

        elif self.ACT_TYPE == ActionType.VEL:
            # Velocity control: [vx, vy, vz, yaw_rate] per drone
            # Uses internal PID to compute RPMs
            from multi_drone_mujoco.control.pid_control import PIDControl
            action = np.array(action).reshape(self.NUM_DRONES, 4)
            rpms = np.zeros((self.NUM_DRONES, 4))
            for i in range(self.NUM_DRONES):
                state = self._getDroneStateVector(i)
                # Compute target position from velocity
                target_pos = self.pos[i] + action[i, :3] * self.CTRL_TIMESTEP
                target_rpy = np.array([0, 0, self.rpy[i, 2] + action[i, 3] * self.CTRL_TIMESTEP])
                rpms[i, :], _, _ = PIDControl(self).computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=self.pos[i],
                    cur_quat=self.quat[i],
                    cur_vel=self.vel[i],
                    cur_ang_vel=self.ang_v[i],
                    target_pos=target_pos,
                    target_rpy=target_rpy,
                )
            return np.clip(rpms, 0, self.MAX_RPM)

        elif self.ACT_TYPE == ActionType.PID:
            # Target [x, y, z, yaw] per drone → PID computes RPMs
            from multi_drone_mujoco.control.pid_control import PIDControl
            action = np.array(action).reshape(self.NUM_DRONES, 4)
            rpms = np.zeros((self.NUM_DRONES, 4))
            for i in range(self.NUM_DRONES):
                target_pos = action[i, :3]
                target_rpy = np.array([0, 0, action[i, 3]])
                rpms[i, :], _, _ = PIDControl(self).computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=self.pos[i],
                    cur_quat=self.quat[i],
                    cur_vel=self.vel[i],
                    cur_ang_vel=self.ang_v[i],
                    target_pos=target_pos,
                    target_rpy=target_rpy,
                )
            return np.clip(rpms, 0, self.MAX_RPM)

        elif self.ACT_TYPE == ActionType.ATTITUDE:
            # [thrust_normalized, roll, pitch, yaw_rate] → RPMs via mixer
            action = np.array(action).reshape(self.NUM_DRONES, 4)
            rpms = np.zeros((self.NUM_DRONES, 4))
            for i in range(self.NUM_DRONES):
                thrust_norm, roll_cmd, pitch_cmd, yaw_rate_cmd = action[i]
                # Convert thrust to collective RPM
                collective_rpm = self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM) * thrust_norm
                # Simple mixer (X-config)
                rpms[i, 0] = collective_rpm + roll_cmd * 0.25 * self.MAX_RPM - pitch_cmd * 0.25 * self.MAX_RPM - yaw_rate_cmd * 0.25 * self.MAX_RPM
                rpms[i, 1] = collective_rpm - roll_cmd * 0.25 * self.MAX_RPM - pitch_cmd * 0.25 * self.MAX_RPM + yaw_rate_cmd * 0.25 * self.MAX_RPM
                rpms[i, 2] = collective_rpm - roll_cmd * 0.25 * self.MAX_RPM + pitch_cmd * 0.25 * self.MAX_RPM - yaw_rate_cmd * 0.25 * self.MAX_RPM
                rpms[i, 3] = collective_rpm + roll_cmd * 0.25 * self.MAX_RPM + pitch_cmd * 0.25 * self.MAX_RPM + yaw_rate_cmd * 0.25 * self.MAX_RPM
            return np.clip(rpms, 0, self.MAX_RPM)

        raise ValueError(f"Unknown action type: {self.ACT_TYPE}")

    def _normalizedActionToRPM(self, action):
        """Convert [-1, 1] normalized action to [0, MAX_RPM]."""
        return np.where(
            action <= 0,
            (action + 1) * self.HOVER_RPM,
            self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM) * action,
        )

    ############################################################################
    # SPACES (default implementations, overridden by subclasses)
    ############################################################################

    def _actionSpace(self):
        """Define action space based on action type."""
        if self.ACT_TYPE == ActionType.RPM:
            act_lower = np.zeros(4 * self.NUM_DRONES)
            act_upper = np.full(4 * self.NUM_DRONES, self.MAX_RPM)
        elif self.ACT_TYPE == ActionType.VEL:
            # [vx, vy, vz, yaw_rate] each in [-1, 1]
            act_lower = np.full(4 * self.NUM_DRONES, -1.0)
            act_upper = np.full(4 * self.NUM_DRONES, 1.0)
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            act_lower = np.full(self.NUM_DRONES, -1.0)
            act_upper = np.full(self.NUM_DRONES, 1.0)
        elif self.ACT_TYPE == ActionType.PID:
            # [x, y, z, yaw]
            act_lower = np.tile(np.array([-10, -10, 0, -np.pi]), self.NUM_DRONES)
            act_upper = np.tile(np.array([10, 10, 10, np.pi]), self.NUM_DRONES)
        elif self.ACT_TYPE == ActionType.ATTITUDE:
            act_lower = np.tile(np.array([-1, -1, -1, -1]), self.NUM_DRONES)
            act_upper = np.tile(np.array([1, 1, 1, 1]), self.NUM_DRONES)
        else:
            raise ValueError(f"Unknown action type: {self.ACT_TYPE}")

        return spaces.Box(low=act_lower.astype(np.float32), high=act_upper.astype(np.float32))

    def _observationSpace(self):
        """Define observation space based on observation type."""
        if self.OBS_TYPE == ObservationType.KIN:
            # 20-dim per drone: pos(3), quat(4), rpy(3), vel(3), angvel(3), last_action(4)
            obs_lower = np.full(20 * self.NUM_DRONES, -np.inf)
            obs_upper = np.full(20 * self.NUM_DRONES, np.inf)
            return spaces.Box(low=obs_lower.astype(np.float32), high=obs_upper.astype(np.float32))
        elif self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(
                low=0, high=255,
                shape=(self.NUM_DRONES, 48, 64, 4),
                dtype=np.uint8,
            )
        elif self.OBS_TYPE == ObservationType.KIN_RGB:
            return spaces.Dict({
                "kin": spaces.Box(low=-np.inf, high=np.inf, shape=(20 * self.NUM_DRONES,), dtype=np.float32),
                "rgb": spaces.Box(low=0, high=255, shape=(self.NUM_DRONES, 48, 64, 4), dtype=np.uint8),
            })
        raise ValueError(f"Unknown obs type: {self.OBS_TYPE}")

    ############################################################################
    # COMPUTE METHODS (to be overridden by subclasses)
    ############################################################################

    def _computeObs(self):
        """Compute observation."""
        if self.OBS_TYPE == ObservationType.KIN:
            obs = np.hstack([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            return obs.astype(np.float32)
        elif self.OBS_TYPE == ObservationType.RGB:
            imgs = np.stack([self._getDroneImages(i)[0] for i in range(self.NUM_DRONES)])
            return imgs
        elif self.OBS_TYPE == ObservationType.KIN_RGB:
            kin = np.hstack([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
            imgs = np.stack([self._getDroneImages(i)[0] for i in range(self.NUM_DRONES)])
            return {"kin": kin.astype(np.float32), "rgb": imgs}
        return np.array([])

    def _computeReward(self):
        """Compute reward. Override in subclass."""
        return 0.0

    def _computeTerminated(self):
        """Check termination. Override in subclass."""
        return False

    def _computeTruncated(self):
        """Check truncation. Override in subclass."""
        return False

    def _computeInfo(self):
        """Compute info dict. Override in subclass."""
        return {}

    ############################################################################
    # RENDERING
    ############################################################################

    # Camera mode constants
    CAMERA_TRACK = "track"       # Follows drone from fixed distance
    CAMERA_FPV = "fpv"           # First-person (egocentric) from drone
    CAMERA_FIXED = "fixed"       # Fixed third-person view
    CAMERA_FRONT = "front"       # Front view looking along X-axis

    def render(self, camera_mode=None, track_drone_id=0):
        """Render the environment.

        Parameters
        ----------
        camera_mode : str, optional
            One of "track" (follows drone), "fpv" (first-person), "fixed" (static).
            Defaults to "track".
        track_drone_id : int
            Which drone to track/follow for "track" and "fpv" modes.
        """
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)

            if camera_mode is None:
                camera_mode = self.CAMERA_TRACK

            camera = mujoco.MjvCamera()

            if camera_mode == self.CAMERA_FPV:
                # First-person: use drone's onboard camera if available
                cam_name = f"drone{track_drone_id}_cam"
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cam_id >= 0:
                    self._renderer.update_scene(self.data, camera=cam_name)
                    return self._renderer.render()
                else:
                    # Fallback: place virtual FPV camera at drone position looking forward
                    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                    pos = self.pos[track_drone_id]
                    # Look ahead in x direction from drone's POV
                    camera.lookat[:] = pos + np.array([2.0, 0.0, -0.3])
                    camera.distance = 0.01
                    camera.azimuth = 90
                    camera.elevation = -10

            elif camera_mode == self.CAMERA_FRONT:
                # Side view: camera looks along X-axis to see Y-Z plane
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                centroid = self.pos.mean(axis=0)
                camera.lookat[:] = [0, 0, 1.0]
                camera.distance = 2.5
                camera.azimuth = 180
                camera.elevation = 0

            elif camera_mode == self.CAMERA_FIXED:
                # Fixed third-person: static camera looking at scene center
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                # Look at average drone height
                avg_z = self.pos[:, 2].mean()
                camera.lookat[:] = [0, 0, avg_z]
                camera.distance = 1.2
                camera.azimuth = -60
                camera.elevation = -25

            else:  # CAMERA_TRACK (default)
                # Tracks the drone(s) from a fixed relative distance
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                centroid = self.pos.mean(axis=0)
                camera.lookat[:] = centroid
                spread = np.linalg.norm(self.pos.max(0) - self.pos.min(0))
                camera.distance = max(0.5, spread * 1.2 + 0.3)
                camera.azimuth = -45
                camera.elevation = -15

            self._renderer.update_scene(self.data, camera)
            return self._renderer.render()
        return None

    def _saveFrame(self):
        """Save a frame to disk for recording."""
        from PIL import Image as PILImage
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        img = self._renderer.render()
        PILImage.fromarray(img).save(
            os.path.join(self.IMG_PATH, f"frame_{self.FRAME_NUM:06d}.png")
        )
        self.FRAME_NUM += 1

    def close(self):
        """Clean up resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer = None

    def set_wind(self, wind_config):
        """Enable wind disturbance.

        Parameters
        ----------
        wind_config : WindConfig
            Wind model configuration. Import from multi_drone_mujoco.wrappers.wind.
        """
        from multi_drone_mujoco.wrappers.wind import WindField
        self._wind_field = WindField(wind_config)

    ############################################################################
    # UTILITIES
    ############################################################################

    def _calculateNextStep(self, current_position, destination, step_size=1.0):
        """Calculate intermediate waypoint towards destination."""
        direction = destination - current_position
        distance = np.linalg.norm(direction)
        if distance <= step_size:
            return destination
        return current_position + (direction / distance) * step_size
