"""Multi-Hover Aviary: multi-drone hover task for MARL.

Task: N drones hover at different target heights.
"""

import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType


class MultiHoverAviary(BaseAviary):
    """Multi-agent hover task — each drone must reach its target height."""

    def __init__(
        self,
        drone_model: DroneModel = DroneModel.CF2X,
        num_drones: int = 2,
        physics: Physics = Physics.MJC,
        sim_freq: int = 240,
        ctrl_freq: int = 48,
        gui: bool = False,
        record: bool = False,
        obstacles: bool = False,
        target_heights=None,
        initial_xyzs=None,
        render_mode=None,
    ):
        self.EPISODE_LEN_SEC = 10

        if target_heights is None:
            self.TARGET_HEIGHTS = np.linspace(0.7, 1.2, num_drones)
        else:
            self.TARGET_HEIGHTS = np.array(target_heights)

        if initial_xyzs is None:
            initial_xyzs = np.zeros((num_drones, 3))
            for i in range(num_drones):
                initial_xyzs[i] = [i * 0.3, 0, 0.1]

        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
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

    def _actionSpace(self):
        """Normalized actions for all drones."""
        return spaces.Box(
            low=-np.ones(4 * self.NUM_DRONES, dtype=np.float32),
            high=np.ones(4 * self.NUM_DRONES, dtype=np.float32),
        )

    def _observationSpace(self):
        """Per-drone: pos(3) + rpy(3) + vel(3) + angvel(3) + target_h(1) = 13 per drone."""
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13 * self.NUM_DRONES,),
            dtype=np.float32,
        )

    # See HoverAviary: zero action == hover thrust, +/-1 == +/-5% RPM.
    ACTION_RPM_SCALE = 0.05
    MAX_XY_DRIFT = 1.5
    MAX_Z = 2.0
    MAX_TILT = 0.4  # rad

    def _preprocessAction(self, action):
        action = np.clip(np.array(action, dtype=np.float64).reshape(self.NUM_DRONES, 4), -1, 1)
        rpms = self.HOVER_RPM * (1.0 + self.ACTION_RPM_SCALE * action)
        return np.clip(rpms, 0, self.MAX_RPM)

    def _computeObs(self):
        obs_list = []
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            drone_obs = np.hstack([
                state[0:3], state[7:10], state[10:13], state[13:16],
                [self.TARGET_HEIGHTS[i]],
            ])
            obs_list.append(drone_obs)
        return np.concatenate(obs_list).astype(np.float32)

    def _computeReward(self):
        """Non-negative per-drone reward (see HoverAviary._computeReward).

        A negative per-step reward combined with a one-off crash penalty makes
        early termination the optimal strategy, so the reward is kept >= 0.
        """
        total = 0.0
        for i in range(self.NUM_DRONES):
            target = np.array([self.INIT_XYZS[i, 0], self.INIT_XYZS[i, 1], self.TARGET_HEIGHTS[i]])
            dist = np.linalg.norm(target - self.pos[i])
            r = max(0.0, 2.0 - dist ** 4) + 0.5 * np.exp(-5.0 * dist)
            r -= 0.05 * min(np.linalg.norm(self.vel[i]), 4.0)
            r -= 0.05 * min(abs(self.rpy[i, 0]) + abs(self.rpy[i, 1]), 2.0)
            total += max(r, 0.0)
        return float(total / self.NUM_DRONES)

    def _computeTerminated(self):
        for i in range(self.NUM_DRONES):
            if self.pos[i, 2] < 0.0 or self.pos[i, 2] > self.MAX_Z:
                return True
            if np.any(np.abs(self.pos[i, 0:2] - self.INIT_XYZS[i, 0:2]) > self.MAX_XY_DRIFT):
                return True
            if abs(self.rpy[i, 0]) > self.MAX_TILT or abs(self.rpy[i, 1]) > self.MAX_TILT:
                return True
        return False

    def _computeTruncated(self):
        return self.step_counter / self.SIM_FREQ >= self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {
            "per_drone": [
                {
                    "height_error": abs(self.pos[i, 2] - self.TARGET_HEIGHTS[i]),
                    "position": self.pos[i].tolist(),
                }
                for i in range(self.NUM_DRONES)
            ]
        }
