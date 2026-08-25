#!/usr/bin/env python3
"""Baseline measurement for the current one-renderer-per-env path.

Answers three questions:

  1. Where does a single env's step time go? (physics vs RGB vs depth vs seg)
  2. What does one more env cost in VRAM and host RAM?
  3. Where does env-steps/sec stop improving as N grows?

(1) prices the segmentation pass that most callers discard. (2) is the ceiling
this work exists to remove. (3) is the number the shared path has to beat.

Run:
    python -m multi_drone_mujoco.bench.baseline --envs 1,2,4,8,16,30
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from multi_drone_mujoco.bench._common import (
    DEFAULT_ENV, GPUSampler, cpu_count, gpu_info, host_rss_mb,
    make_env_factory, resolve_env_class, timing_summary,
)


# --------------------------------------------------------------------------
# Part 1 — single-env microbenchmark
# --------------------------------------------------------------------------

def microbench(env_spec: str, n_frames: int = 500, warmup: int = 50) -> dict:
    """Time physics and each render pass separately, in one process."""
    import mujoco

    env = resolve_env_class(env_spec)(seed=0)
    env.reset()

    # Force the lazy renderer into existence and warm it (shader compile,
    # texture upload) so one-off costs don't land in the measurement.
    for _ in range(warmup):
        env._getDroneImages(0)

    r = env._renderer
    cam = "drone0_cam"
    data = env.data

    def timed(fn, n):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return timing_summary(ts)

    def pass_rgb():
        r.update_scene(data, camera=cam)
        r.render()

    def pass_depth():
        r.enable_depth_rendering()
        r.update_scene(data, camera=cam)
        r.render()
        r.disable_depth_rendering()

    def pass_seg():
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam)
        r.render()
        r.disable_segmentation_rendering()

    def all_three():
        pass_rgb(); pass_depth(); pass_seg()

    def rgbd_two_updates():
        pass_rgb(); pass_depth()

    def rgbd_one_update():
        # Same pixels, update_scene issued once. If this is materially faster,
        # the extra update_scene in _getDroneImages is removable -- but only
        # once verify.py confirms depth is unaffected.
        r.update_scene(data, camera=cam)
        r.render()
        r.enable_depth_rendering()
        r.render()
        r.disable_depth_rendering()

    results = {
        "img_w": int(env.IMG_RES[0]),
        "img_h": int(env.IMG_RES[1]),
        "rgb": timed(pass_rgb, n_frames),
        "depth": timed(pass_depth, n_frames),
        "seg": timed(pass_seg, n_frames),
        "rgbd_two_update_scene": timed(rgbd_two_updates, n_frames),
        "rgbd_one_update_scene": timed(rgbd_one_update, n_frames),
        "all_three": timed(all_three, n_frames),
        "mj_step": timed(lambda: mujoco.mj_step(env.model, data), n_frames),
    }

    # A real env.step(): physics at sim rate + control + one observation.
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    env.reset()
    ts = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        _, _, term, trunc, _ = env.step(act)
        ts.append(time.perf_counter() - t0)
        if term or trunc:
            env.reset()
    results["full_env_step"] = timing_summary(ts)

    results["single_renderer_rgbd_fps"] = round(
        1000.0 / results["rgbd_two_update_scene"]["mean_ms"], 1)
    results["seg_share_of_render_pct"] = round(
        100.0 * results["seg"]["mean_ms"] / results["all_three"]["mean_ms"], 1)
    results["update_scene_saving_pct"] = round(
        100.0 * (results["rgbd_two_update_scene"]["mean_ms"]
                 - results["rgbd_one_update_scene"]["mean_ms"])
        / max(results["rgbd_two_update_scene"]["mean_ms"], 1e-9), 1)

    env.close()
    return results


# --------------------------------------------------------------------------
# Part 2 — scaling sweep
# --------------------------------------------------------------------------

def sweep_one(env_spec: str, n_envs: int, steps: int, warmup: int) -> dict:
    """Aggregate throughput for a given env count, via SubprocVecEnv.

    Uses SubprocVecEnv exactly as train_window.py does, so this describes the
    real training path rather than a synthetic proxy.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv([make_env_factory(env_spec, i) for i in range(n_envs)])
    try:
        venv.reset()
        act = np.zeros((n_envs,) + venv.action_space.shape, dtype=np.float32)

        for _ in range(warmup):
            venv.step(act)

        with GPUSampler() as gpu:
            times = []
            t_start = time.perf_counter()
            for _ in range(steps):
                t0 = time.perf_counter()
                venv.step(act)
                times.append(time.perf_counter() - t0)
            wall = time.perf_counter() - t_start
            rss = host_rss_mb()
            gpu_report = gpu.report()

        return {
            "n_envs": n_envs,
            "contexts": n_envs,
            "env_steps_per_sec": round(n_envs * steps / wall, 1),
            "vec_step": timing_summary(times),
            "host_rss_mb": rss,
            **gpu_report,
        }
    finally:
        venv.close()


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def plot(report: dict, png_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib unavailable — skipping figure")
        return

    sweep = report["sweep"]
    n = [s["n_envs"] for s in sweep]
    fps = [s["env_steps_per_sec"] for s in sweep]
    vram = [s["vram_peak_mb"] for s in sweep]
    rss = [s["host_rss_mb"] for s in sweep]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(n, fps, "o-", color="tab:blue")
    axes[0].set_title("Throughput vs env count\n(flattening = today's ceiling)")
    axes[0].set_ylabel("env-steps / sec")

    if any(v is not None for v in vram):
        axes[1].plot(n, vram, "o-", color="tab:red", label="VRAM")
        axes[1].set_title("Peak VRAM vs env count\n(slope = per-env context cost)")
        if any(v is not None for v in rss):
            ax2 = axes[1].twinx()
            ax2.plot(n, rss, "s--", color="tab:purple", alpha=0.6)
            ax2.set_ylabel("host RSS (MB)", color="tab:purple")
    else:
        axes[1].text(0.5, 0.5, "no NVML", ha="center", va="center")
        axes[1].set_title("Peak VRAM (unavailable)")
    axes[1].set_ylabel("MB")

    for ax in axes[:2]:
        ax.set_xlabel("n_envs")
        ax.grid(alpha=0.3)

    micro = report.get("micro")
    if micro:
        labels = ["mj_step", "rgb", "depth", "seg"]
        vals = [micro["mj_step"]["mean_ms"], micro["rgb"]["mean_ms"],
                micro["depth"]["mean_ms"], micro["seg"]["mean_ms"]]
        axes[2].bar(labels, vals,
                    color=["tab:green", "tab:blue", "tab:cyan", "tab:orange"])
        axes[2].set_ylabel("ms / call")
        axes[2].set_title(
            f"Per-call cost @ {micro['img_w']}x{micro['img_h']}\n"
            f"seg = {micro['seg_share_of_render_pct']}% of render, discarded")
        axes[2].grid(alpha=0.3, axis="y")
    else:
        axes[2].text(0.5, 0.5, "microbench skipped", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print(f"[plot] wrote {png_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=DEFAULT_ENV)
    p.add_argument("--envs", default="1,2,4,8,16,30",
                   help="comma-separated env counts to sweep")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--micro-frames", type=int, default=500)
    p.add_argument("--skip-micro", action="store_true")
    p.add_argument("--out", default="runs/bench/baseline")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "label": "baseline (one renderer per env)",
        "env": args.env,
        "cpu_count": cpu_count(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        **gpu_info(),
    }

    if not args.skip_micro:
        print("[1/2] single-env microbenchmark ...")
        m = microbench(args.env, n_frames=args.micro_frames)
        report["micro"] = m
        print(f"      {m['img_w']}x{m['img_h']}   "
              f"mj_step {m['mj_step']['mean_ms']}ms   "
              f"rgb {m['rgb']['mean_ms']}ms   "
              f"depth {m['depth']['mean_ms']}ms   "
              f"seg {m['seg']['mean_ms']}ms")
        print(f"      segmentation = {m['seg_share_of_render_pct']}% of render "
              f"time, and is discarded by the RL obs")
        print(f"      dropping the redundant update_scene would save "
              f"{m['update_scene_saving_pct']}% of RGB-D time")
        print(f"      one renderer sustains {m['single_renderer_rgbd_fps']} RGB-D fps")

    print("[2/2] scaling sweep ...")
    report["sweep"] = []
    for n in [int(x) for x in args.envs.split(",")]:
        print(f"      n_envs={n:>3} ...", end="", flush=True)
        row = sweep_one(args.env, n, args.steps, args.warmup)
        report["sweep"].append(row)
        print(f" {row['env_steps_per_sec']:>8} env-steps/s"
              f"  VRAM {row['vram_peak_mb']} MB"
              f"  RSS {row['host_rss_mb']} MB"
              f"  p99 {row['vec_step']['p99_ms']} ms")

    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\n[out] wrote {json_path}")
    plot(report, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
