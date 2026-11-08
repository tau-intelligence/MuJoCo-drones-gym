"""Hover Aviary: single-drone hover task for RL training.

Task: hover at z=1.0 and remain stable.
Observation: 12-dim [pos(3), rpy(3), vel(3), ang_vel(3)]  (normalized)
Action: 4-dim normalized RPMs [-1, 1]
"""

import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class HoverAviary(BaseAviary):
    """Single drone hover task — matches gym-pybullet-drones HoverAviary."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        obstacles: bool = False,
        target_height: float = 1.0,
        initial_xyzs=None,
        render_mode=None,
    ):
        self.TARGET_HEIGHT = target_height
        self.EPISODE_LEN_SEC = 10
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 0.1]])

        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            physics=physics,
            sim_freq=sim_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            obs_type=ObservationType.KIN,
            act_type=ActionType.RPM,
            initial_xyzs=initial_xyzs,
            render_mode=render_mode,
        )

    # Fraction of HOVER_RPM that a full-scale action (+/-1) adds or removes.
    # Matches gym-pybullet-drones' ActionType.RPM convention: a zero action is
    # exactly hover thrust, and the policy only learns small corrections.
    ACTION_RPM_SCALE = 0.05

    # Episode is terminated (failure) when the drone leaves this box or tilts
    # past this angle. These are unrecoverable states for a hover policy.
    MAX_XY = 1.5
    MAX_Z = 2.0
    MAX_TILT = 0.4  # rad

    def _actionSpace(self):
        """Normalized [-1, 1] → mapped to RPM internally."""
        return spaces.Box(low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))

    def _observationSpace(self):
        """12-dim observation: pos, rpy, vel, ang_vel."""
        return spaces.Box(
            low=-np.inf * np.ones(12, dtype=np.float32),
            high=np.inf * np.ones(12, dtype=np.float32),
        )

    def _preprocessAction(self, action):
        """Convert normalized action to RPMs centred on hover.

        ``rpm = HOVER_RPM * (1 + ACTION_RPM_SCALE * action)``. The previous
        mapping spanned the full [0, MAX_RPM] range, so tiny policy outputs
        produced huge thrust changes and PPO could not discover hovering.
        """
        action = np.clip(np.array(action, dtype=np.float64).flatten(), -1, 1)
        rpms = self.HOVER_RPM * (1.0 + self.ACTION_RPM_SCALE * action)
        return np.clip(rpms, 0, self.MAX_RPM).reshape(1, 4)

    def _computeObs(self):
        """12-dim observation."""
        state = self._getDroneStateVector(0)
        # pos(3), rpy(3), vel(3), ang_vel(3)
        obs = np.hstack([state[0:3], state[7:10], state[10:13], state[13:16]])
        return obs.astype(np.float32)

    def _computeReward(self):
        """Dense, strictly non-negative reward.

        The previous reward was negative at every step (starting 0.9 m below
        the target) while crashing cost only a one-off -100. Ending the episode
        early was therefore the return-maximising strategy, and PPO learned to
        fly out of the bounds as fast as possible. With a non-negative reward
        every extra step alive is worth something, so termination is never
        attractive and no explicit crash penalty is needed.
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]
        rpy = state[7:10]

        target = np.array([0.0, 0.0, self.TARGET_HEIGHT])
        dist = np.linalg.norm(target - pos)

        reward = max(0.0, 2.0 - dist ** 4)               # in [0, 2]
        reward += 0.5 * np.exp(-5.0 * dist)               # sharpen near target
        reward -= 0.05 * min(np.linalg.norm(vel), 4.0)    # bounded damping terms
        reward -= 0.05 * min(abs(rpy[0]) + abs(rpy[1]), 2.0)
        return float(max(reward, 0.0))

    def _computeTerminated(self):
        """Terminate if the drone crashes, flips, or leaves the flight box."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]

        if pos[2] < 0.0 or pos[2] > self.MAX_Z:
            return True
        if abs(pos[0]) > self.MAX_XY or abs(pos[1]) > self.MAX_XY:
            return True
        if abs(rpy[0]) > self.MAX_TILT or abs(rpy[1]) > self.MAX_TILT:
            return True
        return False

    def _computeTruncated(self):
        """Truncate after episode time limit."""
        return self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {
            "position": self.pos[0].tolist(),
            "height_error": abs(self.pos[0, 2] - self.TARGET_HEIGHT),
        }
