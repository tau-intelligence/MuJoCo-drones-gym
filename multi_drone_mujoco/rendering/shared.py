"""A single GL context that renders many envs' cameras from one uploaded scene.

Why this exists
---------------
With one ``mujoco.Renderer`` per env, running N envs means N GL contexts and N
copies of the arena geometry and textures in VRAM. For a static world those N
copies are byte-identical and never change, so VRAM -- not compute -- becomes
the ceiling on how many envs fit on the card.

``SharedStaticRenderer`` holds *one* model, *one* context and *one* scratch
``MjData``. To render env i it writes that env's configuration into the scratch
data, recomputes only what rendering needs (body/geom/camera poses), and
rasterises. The arena is uploaded once regardless of how many envs it serves.

Correctness requirements (caller's responsibility, asserted not inferred)
-------------------------------------------------------------------------
* every env shares an identical model -- same XML, same assets, same options
* the world is static: nothing moves except via ``qpos`` / mocap
* the drone camera is a real camera in the model

``assert_compatible()`` checks the parts that are cheaply checkable (model
sizes and the camera's existence) and raises rather than rendering something
subtly wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

import mujoco


@dataclass
class RenderStats:
    """Timing for one renderer. ``wait_ms`` is filled in by callers that queue."""

    frames: int = 0
    pose_ms: float = 0.0        # writing state + kinematics
    raster_ms: float = 0.0      # update_scene + render
    wait_ms: float = 0.0        # time callers spent queued (set externally)
    _raster_samples: list = field(default_factory=list, repr=False)

    def as_dict(self) -> dict:
        n = max(self.frames, 1)
        s = np.array(self._raster_samples) if self._raster_samples else np.zeros(1)
        return {
            "frames": self.frames,
            "pose_mean_ms": round(self.pose_ms / n, 4),
            "raster_mean_ms": round(self.raster_ms / n, 4),
            "raster_p99_ms": round(float(np.percentile(s, 99)), 4),
            "wait_mean_ms": round(self.wait_ms / n, 4),
            "fps": round(1000.0 / max(self.raster_ms / n + self.pose_ms / n, 1e-9), 1),
        }


class SharedStaticRenderer:
    """One GL context + one scene, reused across many envs.

    Parameters
    ----------
    model : mujoco.MjModel
        The shared (static) model. Any participating env's model works, since
        they are required to be identical.
    width, height : int
        Offscreen buffer size. Must match what the envs expect.
    want_seg : bool
        Whether the segmentation pass should ever run. When False the pass is
        skipped entirely and a zero buffer is returned -- callers that discard
        segmentation should leave this off.
    """

    #: How much of the calling env's MjData is transferred before rendering.
    #:
    #: "copy"    -- mj_copyData: the whole state. Exact by construction, and the
    #:              only mode that is correct without reasoning about which
    #:              fields mjv_updateScene happens to read.
    #: "minimal" -- copy qpos/mocap and recompute poses with mj_kinematics +
    #:              mj_camlight. Cheaper, but only valid if those are genuinely
    #:              sufficient. Must be proven with bench/verify.py before use.
    STATE_TRANSFER_MODES = ("copy", "minimal")

    def __init__(self, model: "mujoco.MjModel", width: int, height: int,
                 want_seg: bool = False, state_transfer: str = "copy"):
        if state_transfer not in self.STATE_TRANSFER_MODES:
            raise ValueError(
                f"state_transfer must be one of {self.STATE_TRANSFER_MODES}, "
                f"got {state_transfer!r}")
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self.default_want_seg = bool(want_seg)
        self.state_transfer = state_transfer

        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        # Scratch data used only for rendering. Never stepped -- it holds a
        # configuration copied from an env, not an independent simulation.
        self._scratch = mujoco.MjData(model)

        self.stats = RenderStats()

    # -- validation --------------------------------------------------------

    def assert_compatible(self, other_model: "mujoco.MjModel", cam_name: str) -> None:
        """Raise if ``other_model`` cannot safely be rendered by this instance.

        Catches the common failure -- a differently-built world -- before it can
        produce plausible but wrong images. Cannot catch every difference (two
        models can share sizes yet differ in geometry), which is why the caller
        must still assert that worlds are identical.
        """
        m, o = self.model, other_model
        for attr in ("nq", "nv", "nbody", "ngeom", "ncam", "nmat", "ntex", "nmesh"):
            if getattr(m, attr) != getattr(o, attr):
                raise ValueError(
                    f"SharedStaticRenderer: model mismatch on {attr} "
                    f"({getattr(m, attr)} vs {getattr(o, attr)}). Shared rendering "
                    f"requires identical worlds across envs."
                )
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) < 0:
            raise ValueError(
                f"SharedStaticRenderer: camera {cam_name!r} not present in the "
                f"shared model."
            )

    # -- rendering ---------------------------------------------------------

    def render(self, data: "mujoco.MjData", cam_name: str,
               want_seg: Optional[bool] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render one env's camera.

        ``data`` belongs to the *calling env*, not to this renderer. Only the
        configuration is copied across; the scratch data is then brought up to
        date with the minimum needed for rendering.

        Returns ``(rgb, depth, seg)`` matching BaseAviary._getDroneImages.
        """
        if want_seg is None:
            want_seg = self.default_want_seg

        t0 = time.perf_counter()

        s = self._scratch
        if self.state_transfer == "copy":
            # Whole-state copy. Slightly more work than recomputing poses, but
            # it cannot miss a field that the scene builder reads, which the
            # "minimal" path demonstrably could.
            mujoco.mj_copyData(s, self.model, data)
        else:
            s.qpos[:] = data.qpos
            if self.model.nmocap:
                s.mocap_pos[:] = data.mocap_pos
                s.mocap_quat[:] = data.mocap_quat
            mujoco.mj_kinematics(self.model, s)
            mujoco.mj_camlight(self.model, s)

        t1 = time.perf_counter()

        self._renderer.update_scene(s, camera=cam_name)
        rgb = self._renderer.render()

        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(s, camera=cam_name)
        dep = self._renderer.render()
        self._renderer.disable_depth_rendering()

        if want_seg:
            self._renderer.enable_segmentation_rendering()
            self._renderer.update_scene(s, camera=cam_name)
            seg = self._renderer.render()
            self._renderer.disable_segmentation_rendering()
        else:
            # Freshly allocated, matching BaseAviary's per-env path. Handing the
            # same array to every env would alias across them.
            seg = np.zeros((self.height, self.width), dtype=np.int32)

        t2 = time.perf_counter()

        self.stats.frames += 1
        self.stats.pose_ms += (t1 - t0) * 1e3
        self.stats.raster_ms += (t2 - t1) * 1e3
        self.stats._raster_samples.append((t2 - t1) * 1e3)

        return rgb, dep, seg

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def attach_shared_renderer(envs, want_seg: bool = False,
                           renderer: Optional[SharedStaticRenderer] = None,
                           state_transfer: str = "copy"
                           ) -> SharedStaticRenderer:
    """Point a list of same-process envs at one shared renderer.

    The caller is asserting that these envs have identical static worlds.
    """
    if not envs:
        raise ValueError("attach_shared_renderer: no envs given")

    head = envs[0]
    if renderer is None:
        renderer = SharedStaticRenderer(
            head.model, width=int(head.IMG_RES[0]), height=int(head.IMG_RES[1]),
            want_seg=want_seg, state_transfer=state_transfer,
        )

    for e in envs:
        renderer.assert_compatible(e.model, "drone0_cam")
        e._external_renderer = renderer
        e._render_seg = want_seg

    return renderer
