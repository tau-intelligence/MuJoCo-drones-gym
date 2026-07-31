#!/usr/bin/env python3
"""Pick M -- the number of shared renderers -- from measured wait, not a formula.

For a fixed env count N, sweeps M (worker/context count) and reports for each:
throughput, peak VRAM, and the step-latency tail. The right M is the smallest
one whose latency has not yet started climbing: below that, physics parallelism
is being given up for nothing; above it, VRAM is being spent for nothing.

A formula (`M = ceil(N / (R_fps / f_env))`) assumes M contexts deliver M times
one context's throughput -- which is exactly the contention this is measuring.
So the curve is measured rather than predicted.

Run:
    python -m multi_drone_mujoco.bench.calibrate --n-envs 30 --workers 1,2,3,5,6,10,15,30
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
    make_env_factory, timing_summary,
)


def measure(env_spec: str, n_envs: int, n_workers: int, steps: int,
            warmup: int, want_seg: bool) -> dict:
    from multi_drone_mujoco.vec import SharedRenderVecEnv

    env_fns = [make_env_factory(env_spec, i) for i in range(n_envs)]
    venv = SharedRenderVecEnv(env_fns, n_workers=n_workers, want_seg=want_seg)
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

        render = venv.render_stats()
        envs_per_worker = [len(g) for g in venv.groups]

        return {
            "n_workers": n_workers,
            "envs_per_worker_max": max(envs_per_worker),
            "env_steps_per_sec": round(n_envs * steps / wall, 1),
            "vec_step": timing_summary(times),
            "host_rss_mb": rss,
            "renderer_raster_mean_ms": round(
                float(np.mean([r["raster_mean_ms"] for r in render if r])), 4),
            "renderer_raster_p99_ms": round(
                float(np.max([r["raster_p99_ms"] for r in render if r])), 4),
            **gpu_report,
        }
    finally:
        venv.close()


def choose_m(rows: list, tolerance: float = 0.05) -> dict:
    """Smallest M whose p99 step latency is within `tolerance` of the best seen.

    Latency, not throughput, because the stated goal is that envs should not sit
    waiting on a renderer.
    """
    usable = [r for r in rows if r.get("vec_step")]
    if not usable:
        return {}
    best_p99 = min(r["vec_step"]["p99_ms"] for r in usable)
    ceiling = best_p99 * (1.0 + tolerance)
    winners = [r for r in usable if r["vec_step"]["p99_ms"] <= ceiling]
    pick = min(winners, key=lambda r: r["n_workers"])
    return {
        "recommended_workers": pick["n_workers"],
        "best_p99_ms": best_p99,
        "accepted_p99_ms": pick["vec_step"]["p99_ms"],
        "tolerance": tolerance,
        "vram_peak_mb": pick["vram_peak_mb"],
        "env_steps_per_sec": pick["env_steps_per_sec"],
    }


def plot(report: dict, png_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib unavailable — skipping figure")
        return

    rows = report["sweep"]
    m = [r["n_workers"] for r in rows]
    fps = [r["env_steps_per_sec"] for r in rows]
    p99 = [r["vec_step"]["p99_ms"] for r in rows]
    vram = [r["vram_peak_mb"] for r in rows]
    rec = report.get("recommendation", {}).get("recommended_workers")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, y, title, ylab, color in (
        (axes[0], fps, "Throughput vs M", "env-steps / sec", "tab:blue"),
        (axes[1], p99, "Step latency tail vs M\n(the 'waiting' to minimise)",
         "p99 vec step (ms)", "tab:orange"),
        (axes[2], vram, "Peak VRAM vs M\n(cost of each extra context)",
         "MB", "tab:red"),
    ):
        if any(v is None for v in y):
            ax.text(0.5, 0.5, "no NVML", ha="center", va="center")
        else:
            ax.plot(m, y, "o-", color=color)
        if rec is not None:
            ax.axvline(rec, ls="--", color="k", alpha=0.5)
            ax.annotate(f"M={rec}", xy=(rec, ax.get_ylim()[1]),
                        xytext=(3, -12), textcoords="offset points", fontsize=9)
        ax.set_xlabel("M (workers / GL contexts)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    fig.suptitle(f"N = {report['n_envs']} envs", y=1.02)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {png_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=DEFAULT_ENV)
    p.add_argument("--n-envs", type=int, default=30)
    p.add_argument("--workers", default="",
                   help="comma-separated M values (default: divisors-ish of N)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--seg", action="store_true")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="accept M whose p99 is within this fraction of the best")
    p.add_argument("--out", default="runs/bench/calibrate", help="base path for .json and .png output")
    args = p.parse_args()

    n = args.n_envs
    if args.workers:
        ms = [int(x) for x in args.workers.split(",")]
    else:
        ms = sorted({m for m in (1, 2, 3, 4, 5, 6, 8, 10, 15, n) if 1 <= m <= n})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "label": "shared renderer — M sweep",
        "env": args.env,
        "n_envs": n,
        "want_seg": args.seg,
        "cpu_count": cpu_count(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        **gpu_info(),
        "sweep": [],
    }

    print(f"N={n} envs, sweeping M over {ms}  (cpu cores: {report['cpu_count']})\n")
    for m in ms:
        print(f"  M={m:>3} ...", end="", flush=True)
        row = measure(args.env, n, m, args.steps, args.warmup, args.seg)
        report["sweep"].append(row)
        print(f" {row['env_steps_per_sec']:>8} env-steps/s"
              f"  p99 {row['vec_step']['p99_ms']:>7} ms"
              f"  VRAM {row['vram_peak_mb']} MB"
              f"  RSS {row['host_rss_mb']} MB")

    report["recommendation"] = choose_m(report["sweep"], args.tolerance)
    rec = report["recommendation"]

    print()
    if rec:
        print(f"Recommended M = {rec['recommended_workers']}"
              f"  (p99 {rec['accepted_p99_ms']} ms vs best {rec['best_p99_ms']} ms,"
              f"  VRAM {rec['vram_peak_mb']} MB)")
        if rec["recommended_workers"] >= report["cpu_count"]:
            print("  M is at/above your core count — the simple grouped-worker")
            print("  design is the right fit; a render server would not help.")
        else:
            print("  M is below your core count, so grouping ties up physics")
            print("  parallelism you could otherwise use. If throughput matters")
            print("  more than VRAM here, a decoupled render server is the next step.")

    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\n[out] wrote {json_path}")
    plot(report, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
