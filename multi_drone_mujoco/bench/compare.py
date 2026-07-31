#!/usr/bin/env python3
"""Old vs new, head to head: SubprocVecEnv (N contexts) vs shared renderer (M).

Runs both configurations **interleaved** (old, new, old, new, ...) rather than
all-old-then-all-new. GPU clocks drift as the card heats, so a sequential layout
systematically penalises whichever variant ran second; interleaving spreads that
drift across both and the per-repeat spread shows how much it mattered.

Reports throughput, step-latency tail, peak VRAM and host RSS for each, plus the
headroom the VRAM saving buys -- an estimate of how many envs would now fit.

Run (after verify.py passes):
    python -m multi_drone_mujoco.bench.compare --n-envs 30 --workers 6 --repeats 3
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np

from multi_drone_mujoco.bench._common import (
    DEFAULT_ENV, GPUSampler, cpu_count, gpu_info, host_rss_mb,
    make_env_factory, timing_summary,
)


def _run(venv, n_envs: int, steps: int, warmup: int) -> dict:
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
        report = gpu.report()

    return {
        "env_steps_per_sec": round(n_envs * steps / wall, 1),
        "vec_step": timing_summary(times),
        "host_rss_mb": rss,
        **report,
    }


def run_baseline(env_spec: str, n_envs: int, steps: int, warmup: int) -> dict:
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv([make_env_factory(env_spec, i) for i in range(n_envs)])
    try:
        out = _run(venv, n_envs, steps, warmup)
        out["contexts"] = n_envs
        return out
    finally:
        venv.close()


def run_shared(env_spec: str, n_envs: int, n_workers: int, steps: int,
               warmup: int, want_seg: bool) -> dict:
    from multi_drone_mujoco.vec import SharedRenderVecEnv

    venv = SharedRenderVecEnv(
        [make_env_factory(env_spec, i) for i in range(n_envs)],
        n_workers=n_workers, want_seg=want_seg,
    )
    try:
        out = _run(venv, n_envs, steps, warmup)
        out["contexts"] = n_workers
        return out
    finally:
        venv.close()


def _settle(pause: float):
    """Let the driver actually release VRAM before the next configuration.

    Without this the next run's peak reading includes the previous run's
    not-yet-freed allocations.
    """
    gc.collect()
    time.sleep(pause)


def aggregate(runs: list) -> dict:
    """Median across repeats, plus spread (how much clock drift moved things)."""
    fps = [r["env_steps_per_sec"] for r in runs]
    p99 = [r["vec_step"]["p99_ms"] for r in runs]
    vram = [r["vram_peak_mb"] for r in runs if r["vram_peak_mb"] is not None]
    rss = [r["host_rss_mb"] for r in runs if r["host_rss_mb"] is not None]
    return {
        "repeats": len(runs),
        "contexts": runs[0]["contexts"],
        "env_steps_per_sec_median": round(statistics.median(fps), 1),
        "env_steps_per_sec_spread_pct": round(
            100.0 * (max(fps) - min(fps)) / max(statistics.median(fps), 1e-9), 1),
        "vec_step_p99_ms_median": round(statistics.median(p99), 3),
        "vram_peak_mb_median": round(statistics.median(vram), 1) if vram else None,
        "host_rss_mb_median": round(statistics.median(rss), 1) if rss else None,
    }


def plot(report: dict, png_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[plot] matplotlib unavailable — skipping figure")
        return

    b, s = report["baseline"], report["shared"]
    labels = [f"baseline\n{b['contexts']} contexts", f"shared\n{s['contexts']} contexts"]
    colors = ["tab:gray", "tab:green"]

    metrics = [
        ("env_steps_per_sec_median", "env-steps / sec", "Throughput (higher better)"),
        ("vec_step_p99_ms_median", "p99 vec step (ms)", "Latency tail (lower better)"),
        ("vram_peak_mb_median", "MB", "Peak VRAM (lower better)"),
        ("host_rss_mb_median", "MB", "Host RSS (lower better)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (key, ylab, title) in zip(axes, metrics):
        vals = [b.get(key), s.get(key)]
        if any(v is None for v in vals):
            ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
        else:
            bars = ax.bar(labels, vals, color=colors)
            for bar, v in zip(bars, vals):
                ax.annotate(f"{v:g}", xy=(bar.get_x() + bar.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=9)
            if vals[0]:
                delta = 100.0 * (vals[1] - vals[0]) / vals[0]
                ax.set_xlabel(f"{delta:+.1f}% vs baseline")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"N = {report['n_envs']} envs   |   {report['repeats']} interleaved repeats",
        y=1.03)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    print(f"[plot] wrote {png_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default=DEFAULT_ENV)
    p.add_argument("--n-envs", type=int, default=30)
    p.add_argument("--workers", type=int, required=True,
                   help="M — take this from calibrate.py's recommendation")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--settle", type=float, default=3.0,
                   help="seconds between configs, so VRAM is actually released")
    p.add_argument("--seg", action="store_true")
    p.add_argument("--out", default="runs/bench/compare")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "env": args.env,
        "n_envs": args.n_envs,
        "workers": args.workers,
        "repeats": args.repeats,
        "want_seg": args.seg,
        "cpu_count": cpu_count(),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        **gpu_info(),
        "baseline_runs": [],
        "shared_runs": [],
    }

    print(f"N={args.n_envs}  M={args.workers}  repeats={args.repeats} (interleaved)\n")

    for rep in range(args.repeats):
        print(f"  repeat {rep + 1}/{args.repeats}")

        print("    baseline (one renderer per env) ...", end="", flush=True)
        r = run_baseline(args.env, args.n_envs, args.steps, args.warmup)
        report["baseline_runs"].append(r)
        print(f" {r['env_steps_per_sec']} env-steps/s  VRAM {r['vram_peak_mb']} MB")
        _settle(args.settle)

        print("    shared   (M renderers)          ...", end="", flush=True)
        r = run_shared(args.env, args.n_envs, args.workers, args.steps,
                       args.warmup, args.seg)
        report["shared_runs"].append(r)
        print(f" {r['env_steps_per_sec']} env-steps/s  VRAM {r['vram_peak_mb']} MB")
        _settle(args.settle)

    report["baseline"] = aggregate(report["baseline_runs"])
    report["shared"] = aggregate(report["shared_runs"])

    b, s = report["baseline"], report["shared"]
    print("\n" + "=" * 66)
    print(f"{'metric':<28}{'baseline':>16}{'shared':>16}")
    print("-" * 66)
    for key, name in (
        ("contexts", "GL contexts"),
        ("env_steps_per_sec_median", "env-steps/sec"),
        ("vec_step_p99_ms_median", "p99 step (ms)"),
        ("vram_peak_mb_median", "peak VRAM (MB)"),
        ("host_rss_mb_median", "host RSS (MB)"),
    ):
        print(f"{name:<28}{str(b.get(key)):>16}{str(s.get(key)):>16}")
    print("=" * 66)
    print(f"run-to-run spread: baseline {b['env_steps_per_sec_spread_pct']}%,"
          f" shared {s['env_steps_per_sec_spread_pct']}%"
          "   (larger than the difference => inconclusive)")

    # What the VRAM saving actually buys: the point of the exercise.
    if b.get("vram_peak_mb_median") and s.get("vram_peak_mb_median"):
        total = report.get("vram_total_mb")
        per_env_base = b["vram_peak_mb_median"] / args.n_envs
        per_env_shared = s["vram_peak_mb_median"] / args.n_envs
        print(f"\nVRAM per env: {per_env_base:.1f} MB -> {per_env_shared:.1f} MB")
        if total:
            print(f"envs that fit in {total:.0f} MB (linear extrapolation, "
                  f"physics/CPU limits aside):")
            print(f"    baseline ~{int(total / max(per_env_base, 1e-9))}"
                  f"    shared ~{int(total / max(per_env_shared, 1e-9))}")

    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2))
    print(f"\n[out] wrote {json_path}")
    plot(report, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
