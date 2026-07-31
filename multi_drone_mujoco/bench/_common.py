"""Shared helpers for the render benchmarks: GPU/RAM sampling and env loading."""

from __future__ import annotations

import importlib
import statistics
import threading
from typing import Callable

import numpy as np


# --------------------------------------------------------------------------
# env factory resolution
# --------------------------------------------------------------------------

# Self-contained static-world env: no external assets, no task-repo imports.
# Override with --env 'module:Class' to benchmark a real task env instead.
DEFAULT_ENV = "multi_drone_mujoco.bench.env:BenchAviary"


def resolve_env_class(spec: str):
    """Turn 'package.module:ClassName' into the class object."""
    if ":" not in spec:
        raise ValueError(f"env spec must be 'module:Class', got {spec!r}")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def make_env_factory(spec: str, rank: int, seed: int = 0) -> Callable:
    """Picklable-by-cloudpickle factory for one env."""
    def _init():
        cls = resolve_env_class(spec)
        return cls(seed=seed + rank)
    return _init


# --------------------------------------------------------------------------
# GPU / host memory
# --------------------------------------------------------------------------

class GPUSampler:
    """Background sampler for VRAM + utilisation via NVML.

    No-op (fields None) when pynvml or a GPU is missing, so timing numbers are
    still produced on a CPU-only box.
    """

    def __init__(self, device_index: int = 0, period: float = 0.1):
        self.period = period
        self.samples_mb: list = []
        self.samples_util: list = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            self._pynvml = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def _loop(self):
        while not self._stop.is_set():
            try:
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.samples_mb.append(mem.used / 1024 / 1024)
                self.samples_util.append(util.gpu)
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        if self.available:
            self._stop.clear()
            self.samples_mb, self.samples_util = [], []
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)

    def report(self) -> dict:
        if not self.samples_mb:
            return {"vram_peak_mb": None, "vram_mean_mb": None, "gpu_util_mean_pct": None}
        return {
            "vram_peak_mb": round(max(self.samples_mb), 1),
            "vram_mean_mb": round(statistics.fmean(self.samples_mb), 1),
            "gpu_util_mean_pct": round(statistics.fmean(self.samples_util), 1),
        }


def host_rss_mb():
    """RSS of this process tree in MB (None if psutil absent)."""
    try:
        import psutil
    except ImportError:
        return None
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except Exception:
            pass
    return round(total / 1024 / 1024, 1)


def gpu_info() -> dict:
    """GPU name, total VRAM and current SM clock.

    The clock is recorded so runs made at different thermal states can be
    identified after the fact rather than silently compared.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        return {
            "gpu_name": name,
            "vram_total_mb": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 1024 / 1024, 1),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
        }
    except Exception:
        return {"gpu_name": None, "vram_total_mb": None, "sm_clock_mhz": None}


def cpu_count() -> int:
    import os
    return os.cpu_count() or 1


def timing_summary(samples_s) -> dict:
    t = np.asarray(samples_s, dtype=float) * 1e3
    return {
        "mean_ms": round(float(t.mean()), 3),
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p99_ms": round(float(np.percentile(t, 99)), 3),
        "max_ms": round(float(t.max()), 3),
    }
