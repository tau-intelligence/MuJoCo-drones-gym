"""A self-contained static-world env for benchmarking the render path.

Why this exists rather than benchmarking a task env: the shared renderer's whole
premise is a *static, identical* world across envs, and the measurement that
matters is VRAM per GL context -- which is dominated by how much geometry and
texture the scene holds. So the bench env needs a scene whose weight can be
dialled up, and it must not depend on any external asset files.

Everything here is procedural: MuJoCo's builtin checker/flat texture generators
produce real texture memory without a single file on disk. `n_clutter` controls
how many textured boxes the arena holds, so you can match the weight of the
world you actually train on.

The observation deliberately mirrors a vision policy's: RGB-D image plus a small
proprioception vector, so per-step cost is representative.

    from multi_drone_mujoco.bench.env import BenchAviary
    env = BenchAviary(img_w=96, img_h=96, n_clutter=24)

Scene weight is a benchmark parameter
-------------------------------------
A near-empty world understates VRAM per context and therefore understates what
shared rendering saves. If you are sizing this against a specific arena, raise
`n_clutter` until `vram_peak_mb` per env in bench/baseline.py is in the same
range as that arena's.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from multi_drone_mujoco.envs import base_aviary as _BA
from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType,
)


def _clutter_xml(n: int, seed: int = 0) -> tuple:
    """Deterministic procedural clutter: (asset_xml, body_xml).

    Placement is a fixed function of the index -- never random -- because every
    env must produce a byte-identical world for shared rendering to be valid.
    """
    assets, bodies = [], []
    for i in range(n):
        # Deterministic spread on a spiral, well clear of the drone's spawn.
        ang = 2.399963 * i           # golden angle, gives an even spread
        rad = 1.2 + 0.35 * (i % 7)
        x, y = rad * np.cos(ang), rad * np.sin(ang)
        z = 0.25 + 0.30 * (i % 5)
        r, g, b = 0.25 + 0.5 * ((i * 7) % 5) / 4.0, \
                  0.25 + 0.5 * ((i * 3) % 4) / 3.0, \
                  0.25 + 0.5 * ((i * 5) % 6) / 5.0
        kind = "checker" if i % 2 == 0 else "gradient"
        assets.append(
            f'    <texture name="bench_tex_{i}" type="2d" builtin="{kind}" '
            f'width="256" height="256" rgb1="{r:.3f} {g:.3f} {b:.3f}" '
            f'rgb2="{b:.3f} {r:.3f} {g:.3f}"/>\n'
            f'    <material name="bench_mat_{i}" texture="bench_tex_{i}" '
            f'texrepeat="3 3" specular="0.3" shininess="0.4"/>'
        )
        bodies.append(
            f'    <body name="bench_clutter_{i}" pos="{x:.4f} {y:.4f} {z:.4f}">\n'
            f'      <geom type="box" size="0.12 0.12 {0.15 + 0.05 * (i % 4):.3f}" '
            f'material="bench_mat_{i}" contype="0" conaffinity="0"/>\n'
            f'    </body>'
        )
    return "\n".join(assets), "\n".join(bodies)


def _make_injector(n_clutter: int):
    """Wrap _generate_aviary_xml so the clutter is spliced into the model.

    Same mechanism the task envs use: patch the module-level builder for the
    duration of construction, then restore it.
    """
    original = _BA._generate_aviary_xml

    def patched(*args, **kwargs):
        xml = original(*args, **kwargs)
        if n_clutter <= 0:
            return xml
        asset_xml, body_xml = _clutter_xml(n_clutter)
        if "</asset>" not in xml or "</worldbody>" not in xml:
            raise RuntimeError(
                "BenchAviary: generated XML has no <asset>/<worldbody> to splice "
                "into — the base XML layout changed."
            )
        xml = xml.replace("</asset>", asset_xml + "\n  </asset>", 1)
        xml = xml.replace("</worldbody>", body_xml + "\n  </worldbody>", 1)
        return xml

    return original, patched


class BenchAviary(BaseAviary):
    """Single drone, static procedural world, RGB-D + proprio observation.

    Parameters
    ----------
    img_w, img_h : int
        Onboard camera resolution. Defaults match a typical vision policy.
    n_clutter : int
        Number of textured boxes in the arena. Controls scene weight, and so
        VRAM per GL context -- the quantity shared rendering reduces.
    depth_max : float
        Depth clip (m) for packing depth into the 4th image channel.
    seed : int
        Only seeds the action/reset RNG. The *world* is identical regardless,
        which is what makes shared rendering valid.
    """

    IMG_W, IMG_H = 96, 96

    def __init__(self, img_w: int = 96, img_h: int = 96, n_clutter: int = 24,
                 depth_max: float = 8.0, seed: int = None,
                 sim_freq: int = 240, ctrl_freq: int = 48):
        self.IMG_W, self.IMG_H = int(img_w), int(img_h)
        self.DEPTH_MAX = float(depth_max)
        self.N_CLUTTER = int(n_clutter)
        self._rng = np.random.default_rng(seed)
        self.EPISODE_LEN_SEC = 30      # long, so resets don't dominate timings

        original, patched = _make_injector(self.N_CLUTTER)
        _BA._generate_aviary_xml = patched
        try:
            super().__init__(
                drone_model=DroneModel.CF2X,
                num_drones=1,
                initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
                initial_rpys=np.array([[0.0, 0.0, 0.0]]),
                physics=Physics.MJC,
                sim_freq=sim_freq, ctrl_freq=ctrl_freq,
                vision_attributes=True,
                obs_type=ObservationType.KIN,
                act_type=ActionType.RPM,
            )
        finally:
            _BA._generate_aviary_xml = original

        # BaseAviary forces 256x256 when vision_attributes is on; the renderer
        # is built lazily from IMG_RES, so overriding it here takes effect.
        self.IMG_RES = np.array([self.IMG_W, self.IMG_H])

    # -- spaces ------------------------------------------------------------

    def _observationSpace(self):
        return spaces.Dict({
            "image": spaces.Box(0, 255, shape=(self.IMG_H, self.IMG_W, 4), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32),
        })

    def _actionSpace(self):
        return spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def _preprocessAction(self, action):
        """Map [-1,1] onto a small band around hover.

        Keeps the drone airborne and roughly in place so the camera keeps seeing
        the scene -- a drone on the floor would render a degenerate view and
        misrepresent the cost.
        """
        a = np.clip(np.asarray(action, dtype=float).flatten(), -1.0, 1.0)
        rpm = self.HOVER_RPM * (1.0 + 0.03 * a)
        return np.clip(rpm, 0, self.MAX_RPM).reshape(1, 4)

    # -- observation -------------------------------------------------------

    def _computeObs(self):
        rgb, dep, _ = self._getDroneImages(0)
        depth_u8 = (np.clip(dep, 0.0, self.DEPTH_MAX) / self.DEPTH_MAX * 255.0).astype(np.uint8)
        image = np.concatenate([rgb[..., :3], depth_u8[..., None]], axis=2)

        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.quat[0])
        R = R.reshape(3, 3)
        grav_body = R.T @ np.array([0.0, 0.0, -1.0])
        angv_body = R.T @ self.ang_v[0]
        vel_body = R.T @ self.vel[0]
        proprio = np.concatenate([grav_body, angv_body, vel_body]).astype(np.float32)

        return {"image": image, "proprio": proprio}

    # -- task stubs --------------------------------------------------------
    # There is no task here: this env exists to be stepped and rendered, so the
    # reward is constant and episodes only end on the time limit. That keeps
    # resets out of the timing measurements.

    def _computeReward(self):
        return 0.0

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {}
