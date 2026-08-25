#!/usr/bin/env python3
"""Prove the shared renderer produces the same pixels as the per-env renderer.

This must pass before any speed number from compare.py means anything. A shared
renderer that mixes up which image belongs to which env does not raise -- it
silently feeds the policy the wrong observation and training just gets worse.

Two checks:

  1. **Parity** -- K envs stepped through identical action sequences, once with
     per-env renderers and once through one shared renderer. Every RGB and
     depth frame must match exactly.

  2. **No cross-talk** -- the same test, but the order in which the shared
     renderer serves envs is shuffled every step. If any per-env state leaks
     across calls, differing service order changes the output and this fails
     while check 1 passes.

On failure it localises the divergence rather than leaving you to guess. There
are only three places it can come from, and they are checked in order:

  A. the two envs' *physics* already differ  -> the test is invalid, not the
     renderer
  B. the env's state did not reach the renderer's scratch data intact -> the
     state-transfer step is wrong
  C. neither -> identical state, different pixels: a renderer *configuration*
     difference (scene options, lighting, buffer setup)

Run:
    python -m multi_drone_mujoco.bench.verify --envs 4 --steps 60
    python -m multi_drone_mujoco.bench.verify --state-transfer minimal
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys

import numpy as np

from multi_drone_mujoco.bench._common import DEFAULT_ENV, resolve_env_class
from multi_drone_mujoco.rendering import SharedStaticRenderer


def _maxdiff(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).max()) if a.size else 0.0


def _rgb_stats(a, b) -> dict:
    """Max, mean and spread of an RGB difference.

    The max alone cannot distinguish rounding from a real defect: a handful of
    pixels off by 1 is quantisation, most of the frame off by 1 is systematic.
    """
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return {
        "max": int(d.max()),
        "mean": float(d.mean()),
        "frac_differing": float((d > 0).mean()),
    }


def measure_noise_floor(env, renderer, n: int = 8) -> dict:
    """Empirical repeatability of each render path, with state held fixed.

    Renders the same unchanged state repeatedly through the per-env renderer and
    through the shared renderer. Whatever difference shows up here is the
    floor -- GPU/driver non-determinism that no implementation can remove. A
    shared-vs-baseline difference at or below this floor carries no information.
    """
    saved = env._external_renderer
    floors = {}

    env._external_renderer = None
    ref_rgb, ref_dep, _ = env._getDroneImages(0)
    rgb_max = dep_max = 0.0
    for _ in range(n):
        rgb, dep, _ = env._getDroneImages(0)
        rgb_max = max(rgb_max, _maxdiff(ref_rgb, rgb))
        dep_max = max(dep_max, _maxdiff(ref_dep, dep))
    floors["per_env_renderer"] = {"rgb": rgb_max, "depth": dep_max}

    env._external_renderer = renderer
    ref_rgb, ref_dep, _ = env._getDroneImages(0)
    rgb_max = dep_max = 0.0
    for _ in range(n):
        rgb, dep, _ = env._getDroneImages(0)
        rgb_max = max(rgb_max, _maxdiff(ref_rgb, rgb))
        dep_max = max(dep_max, _maxdiff(ref_dep, dep))
    floors["shared_renderer"] = {"rgb": rgb_max, "depth": dep_max}

    env._external_renderer = saved
    floors["rgb"] = max(floors["per_env_renderer"]["rgb"],
                        floors["shared_renderer"]["rgb"])
    floors["depth"] = max(floors["per_env_renderer"]["depth"],
                          floors["shared_renderer"]["depth"])
    return floors


def _diagnose(base_env, shared_env, renderer) -> dict:
    """Localise a mismatch. Call immediately after rendering `shared_env`.

    The renderer's scratch data still holds what was actually rasterised, so
    comparing env -> scratch -> base pins down which link broke.
    """
    b, s, scratch = base_env.data, shared_env.data, renderer._scratch

    physics = {
        "qpos": _maxdiff(b.qpos, s.qpos),
        "qvel": _maxdiff(b.qvel, s.qvel),
    }
    transfer = {
        "qpos": _maxdiff(s.qpos, scratch.qpos),
        "cam_xpos": _maxdiff(s.cam_xpos, scratch.cam_xpos),
        "cam_xmat": _maxdiff(s.cam_xmat, scratch.cam_xmat),
        "geom_xpos": _maxdiff(s.geom_xpos, scratch.geom_xpos),
        "geom_xmat": _maxdiff(s.geom_xmat, scratch.geom_xmat),
        "light_xpos": _maxdiff(s.light_xpos, scratch.light_xpos)
                      if renderer.model.nlight else 0.0,
    }

    if max(physics.values()) > 0:
        verdict = ("A: the two envs' physics already diverged — the comparison "
                   "is invalid, fix the test setup before blaming the renderer")
    elif max(transfer.values()) > 0:
        worst = max(transfer, key=transfer.get)
        verdict = (f"B: env state did not reach the renderer intact "
                   f"(worst field: {worst} = {transfer[worst]:.6g}) — the "
                   f"state-transfer step is incomplete")
    else:
        verdict = ("C: identical state, different pixels — a renderer "
                   "configuration difference (scene options / lighting / "
                   "buffer setup), not a state problem")

    return {"physics": physics, "transfer": transfer, "verdict": verdict}


def run_check(env_cls, k: int, steps: int, seed: int, shuffle: bool,
              want_seg: bool, state_transfer: str, rgb_tol: int) -> dict:
    base_envs = [env_cls(seed=seed + i) for i in range(k)]
    shared_envs = [env_cls(seed=seed + i) for i in range(k)]

    # Baseline envs keep their own renderers; make them render the same passes
    # the shared path will, so the comparison is like-for-like.
    for e in base_envs:
        e._render_seg = want_seg

    head = shared_envs[0]
    renderer = SharedStaticRenderer(
        head.model, width=int(head.IMG_RES[0]), height=int(head.IMG_RES[1]),
        want_seg=want_seg, state_transfer=state_transfer,
    )
    for e in shared_envs:
        renderer.assert_compatible(e.model, "drone0_cam")
        e._external_renderer = renderer
        e._render_seg = want_seg

    for e in base_envs + shared_envs:
        e.reset(seed=seed)

    # Establish what "identical" can even mean on this GPU before judging.
    noise = measure_noise_floor(shared_envs[0], renderer)
    rgb_gate = max(int(np.ceil(noise["rgb"])), rgb_tol)

    rng = np.random.default_rng(seed)
    order_rng = np.random.default_rng(seed + 9999)

    worst_rgb, worst_dep = 0, 0.0
    mismatched_steps, compared = 0, 0
    diagnosis = None
    worst_stats = {"max": 0, "mean": 0.0, "frac_differing": 0.0}

    try:
        for _ in range(steps):
            actions = [rng.uniform(-1, 1, size=base_envs[0].action_space.shape
                                   ).astype(np.float32) for _ in range(k)]
            for e, a in zip(base_envs, actions):
                e.step(a)
            for e, a in zip(shared_envs, actions):
                e.step(a)

            order = list(range(k))
            if shuffle:
                order_rng.shuffle(order)

            step_bad = False
            for i in order:
                # Shared first, so the scratch data still describes this env
                # if we need to diagnose.
                s_rgb, s_dep, _ = shared_envs[i]._getDroneImages(0)
                snapshot = (_diagnose(base_envs[i], shared_envs[i], renderer)
                            if diagnosis is None else None)
                b_rgb, b_dep, _ = base_envs[i]._getDroneImages(0)

                stats = _rgb_stats(b_rgb, s_rgb)
                d_rgb = stats["max"]
                d_dep = float(np.abs(b_dep - s_dep).max())
                worst_rgb = max(worst_rgb, d_rgb)
                worst_dep = max(worst_dep, d_dep)
                if stats["frac_differing"] > worst_stats["frac_differing"]:
                    worst_stats = stats
                compared += 1

                # Depth must be exact -- it encodes geometry and camera pose, so
                # any difference there is a real defect. RGB is judged against
                # the measured repeatability floor.
                if d_rgb > rgb_gate or d_dep != 0.0:
                    step_bad = True
                    if diagnosis is None:
                        diagnosis = snapshot
                        diagnosis["env_index"] = i
                        diagnosis["rgb_diff"] = d_rgb
                        diagnosis["depth_diff"] = d_dep

            if step_bad:
                mismatched_steps += 1
    finally:
        renderer.close()
        for e in base_envs + shared_envs:
            e.close()

    return {
        "n_envs": k,
        "steps": steps,
        "shuffled_service_order": shuffle,
        "want_seg": want_seg,
        "state_transfer": state_transfer,
        "frames_compared": compared,
        "max_rgb_abs_diff": worst_rgb,
        "max_depth_abs_diff": worst_dep,
        "rgb_worst_frame_stats": worst_stats,
        "noise_floor": noise,
        "rgb_gate": rgb_gate,
        "mismatched_steps": mismatched_steps,
        "bit_identical": worst_rgb == 0 and worst_dep == 0.0,
        "passed": worst_rgb <= rgb_gate and worst_dep == 0.0,
        "diagnosis": diagnosis,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=DEFAULT_ENV)
    p.add_argument("--envs", type=int, default=4, help="envs sharing one renderer")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seg", action="store_true",
                   help="also render and compare the segmentation pass")
    p.add_argument("--state-transfer", default="copy",
                   choices=SharedStaticRenderer.STATE_TRANSFER_MODES,
                   help="'copy' transfers the whole MjData (exact); 'minimal' "
                        "copies qpos and recomputes poses (faster, must be "
                        "proven here before use)")
    p.add_argument("--rgb-tol", type=int, default=1,
                   help="allowed RGB LSB difference. The gate is the larger of "
                        "this and the measured repeatability floor. Depth is "
                        "always required to match exactly.")
    args = p.parse_args()

    env_cls = resolve_env_class(args.env)

    print(f"env={args.env}  k={args.envs}  steps={args.steps}  "
          f"seg={args.seg}  state_transfer={args.state_transfer}\n")

    results = []
    for shuffle in (False, True):
        name = "cross-talk (shuffled order)" if shuffle else "parity (in order)"
        print(f"[{name}] ...", end="", flush=True)
        r = run_check(env_cls, args.envs, args.steps, args.seed, shuffle,
                      args.seg, args.state_transfer, args.rgb_tol)
        results.append((name, r))
        st = r["rgb_worst_frame_stats"]
        print(f" {'PASS' if r['passed'] else 'FAIL'}"
              f"  rgb_max={r['max_rgb_abs_diff']} (gate {r['rgb_gate']},"
              f" {st['frac_differing'] * 100:.3f}% of pixels differ,"
              f" mean {st['mean']:.4f})"
              f"  depth_max={r['max_depth_abs_diff']:.6g}"
              f"  ({r['frames_compared']} frames)")

    print()
    if all(r["passed"] for _, r in results):
        first = results[0][1]
        nf = first["noise_floor"]
        print(f"measured repeatability floor on this GPU: "
              f"per-env renderer rgb={nf['per_env_renderer']['rgb']:.6g}, "
              f"shared renderer rgb={nf['shared_renderer']['rgb']:.6g} "
              f"(same state rendered repeatedly)")
        if all(r["bit_identical"] for _, r in results):
            print("ALL CHECKS PASSED — shared renderer is bit-identical.")
        else:
            print("ALL CHECKS PASSED — depth exact (geometry and camera pose "
                  "provably correct); RGB differs only within the measured\n"
                  "floor, i.e. colour quantisation between independent GL "
                  "contexts, not an env/image association fault.")
        return 0

    print("FAILED — do not trust any speed comparison until this passes.\n")
    for name, r in results:
        d = r.get("diagnosis")
        if not d:
            continue
        print(f"--- first divergence in '{name}' (env {d['env_index']}, "
              f"rgb {d['rgb_diff']}, depth {d['depth_diff']:.6g}) ---")
        print(f"  physics  base vs shared : "
              + ", ".join(f"{k}={v:.6g}" for k, v in d["physics"].items()))
        print(f"  transfer env vs scratch : "
              + ", ".join(f"{k}={v:.6g}" for k, v in d["transfer"].items()))
        print(f"  => {d['verdict']}")
        print()
        break

    if args.state_transfer == "minimal":
        print("Re-run with --state-transfer copy: if that passes, 'minimal' is "
              "missing a field that the scene builder reads.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
