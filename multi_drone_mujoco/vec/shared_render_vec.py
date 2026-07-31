"""VecEnv that runs N envs across M worker processes, one GL context each.

Contrast with ``SubprocVecEnv``, which puts one env in each process and so
creates N GL contexts and N copies of the arena in VRAM.

Here N envs are split into M groups. Each worker process holds its group's envs
and a single :class:`SharedStaticRenderer`, so VRAM cost scales with **M, not
N** -- which is what lets N grow past the point where VRAM currently stops it.

Total rasterisation work is unchanged: N images per vec step either way. What
changes is context count, VRAM, and CPU-side scene setup.

Choosing M
----------
M is both the render-context count and the physics-process count, so it trades
VRAM against physics parallelism. ``multi_drone_mujoco/bench/calibrate.py``
sweeps it and reports the wait-time curve; pick the smallest M whose per-step
wait stays flat.

Only valid for identical static worlds -- see multi_drone_mujoco.rendering.
"""

from __future__ import annotations

import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv, CloudpickleWrapper, VecEnvIndices, VecEnvObs, VecEnvStepReturn,
)

from multi_drone_mujoco.rendering import SharedStaticRenderer


# --------------------------------------------------------------------------
# observation stacking
# --------------------------------------------------------------------------

def _stack_obs(obs_list: List[Any], space: gym.Space):
    """Stack a list of per-env observations into batched form.

    Implemented locally rather than importing SB3's private ``_flatten_obs`` so
    an SB3 refactor cannot silently break the render path.
    """
    if isinstance(space, gym.spaces.Dict):
        return OrderedDict(
            (k, np.stack([o[k] for o in obs_list])) for k in space.spaces.keys()
        )
    if isinstance(space, gym.spaces.Tuple):
        n = len(space.spaces)
        return tuple(np.stack([o[i] for o in obs_list]) for i in range(n))
    return np.stack(obs_list)


def _split_indices(n_envs: int, n_workers: int) -> List[List[int]]:
    """Split env indices into n_workers contiguous, near-equal groups."""
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if n_workers > n_envs:
        n_workers = n_envs
    base, extra = divmod(n_envs, n_workers)
    groups, start = [], 0
    for w in range(n_workers):
        size = base + (1 if w < extra else 0)
        groups.append(list(range(start, start + size)))
        start += size
    return groups


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------

def _worker(remote, parent_remote, env_fns_wrapper, want_seg: bool,
            state_transfer: str) -> None:
    parent_remote.close()
    env_fns = env_fns_wrapper.var
    envs = [fn() for fn in env_fns]

    # One renderer for this whole group. Built after the envs so it can adopt
    # their model and validate compatibility.
    renderer: Optional[SharedStaticRenderer] = None
    try:
        head = envs[0]
        renderer = SharedStaticRenderer(
            head.model,
            width=int(head.IMG_RES[0]),
            height=int(head.IMG_RES[1]),
            want_seg=want_seg,
            state_transfer=state_transfer,
        )
        for e in envs:
            renderer.assert_compatible(e.model, "drone0_cam")
            e._external_renderer = renderer
            e._render_seg = want_seg

        while True:
            cmd, data = remote.recv()

            if cmd == "step":
                results = []
                for env, action in zip(envs, data):
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    if done:
                        info["terminal_observation"] = obs
                        obs, reset_info = env.reset()
                        info["reset_info"] = reset_info
                    results.append((obs, reward, done, info))
                remote.send(results)

            elif cmd == "reset":
                out = []
                for env, (seed, options) in zip(envs, data):
                    obs, reset_info = env.reset(seed=seed, options=options)
                    out.append((obs, reset_info))
                remote.send(out)

            elif cmd == "render":
                remote.send([env.render() for env in envs])

            elif cmd == "close":
                for env in envs:
                    env.close()
                if renderer is not None:
                    renderer.close()
                remote.close()
                break

            elif cmd == "get_spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))

            elif cmd == "get_attr":
                remote.send([getattr(env, data) for env in envs])

            elif cmd == "set_attr":
                for env in envs:
                    setattr(env, data[0], data[1])
                remote.send([None] * len(envs))

            elif cmd == "env_method":
                name, args, kwargs = data
                remote.send([getattr(env, name)(*args, **kwargs) for env in envs])

            elif cmd == "is_wrapped":
                from stable_baselines3.common import env_util
                remote.send([env_util.is_wrapped(env, data) for env in envs])

            elif cmd == "render_stats":
                remote.send(renderer.stats.as_dict() if renderer else None)

            else:
                raise NotImplementedError(f"worker: unknown command {cmd}")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        try:
            if renderer is not None:
                renderer.close()
            for env in envs:
                env.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# VecEnv
# --------------------------------------------------------------------------

class SharedRenderVecEnv(VecEnv):
    """N envs over M processes, each process owning one shared GL context.

    Parameters
    ----------
    env_fns : list of callables
        One factory per env, as with SubprocVecEnv.
    n_workers : int
        M -- number of processes, and therefore of GL contexts. Must be <= N.
        ``n_workers == len(env_fns)`` reproduces SubprocVecEnv's context count
        (useful as a control in benchmarks).
    want_seg : bool
        Render the segmentation pass. Off by default -- most callers discard it
        and it costs a full extra pass.
    start_method : str, optional
        Defaults to forkserver where available, else spawn (matching SB3).
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], n_workers: int,
                 want_seg: bool = False, start_method: Optional[str] = None,
                 state_transfer: str = "copy"):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if n_workers > n_envs:
            raise ValueError(f"n_workers ({n_workers}) > n_envs ({n_envs})")

        self.groups = _split_indices(n_envs, n_workers)
        self.n_workers = len(self.groups)

        if start_method is None:
            methods = mp.get_all_start_methods()
            start_method = "forkserver" if "forkserver" in methods else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes, self.processes = [], [], []
        for group in self.groups:
            remote, work_remote = ctx.Pipe()
            fns = [env_fns[i] for i in group]
            proc = ctx.Process(
                target=_worker,
                args=(work_remote, remote, CloudpickleWrapper(fns), want_seg,
                      state_transfer),
                daemon=True,
            )
            proc.start()
            self.remotes.append(remote)
            self.work_remotes.append(work_remote)
            self.processes.append(proc)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(n_envs, observation_space, action_space)

    # -- stepping ----------------------------------------------------------

    def step_async(self, actions: np.ndarray) -> None:
        for remote, group in zip(self.remotes, self.groups):
            remote.send(("step", [actions[i] for i in group]))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        per_worker = [remote.recv() for remote in self.remotes]
        self.waiting = False

        # Flatten back into env order. Groups are contiguous and in order, so
        # concatenation restores the caller's indexing exactly.
        results = [r for chunk in per_worker for r in chunk]
        obs, rews, dones, infos = zip(*results)
        return (_stack_obs(list(obs), self.observation_space),
                np.array(rews, dtype=np.float32),
                np.array(dones, dtype=bool),
                list(infos))

    def reset(self) -> VecEnvObs:
        for remote, group in zip(self.remotes, self.groups):
            payload = [(self._seeds[i], self._options[i]) for i in group]
            remote.send(("reset", payload))
        per_worker = [remote.recv() for remote in self.remotes]
        results = [r for chunk in per_worker for r in chunk]
        obs = [o for o, _ in results]
        self.reset_infos = [info for _, info in results]
        self._reset_seeds()
        self._reset_options()
        return _stack_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    remote.recv()
                except EOFError:
                    pass
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.processes:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
        self.closed = True

    # -- introspection -----------------------------------------------------

    def _remotes_for(self, indices: VecEnvIndices):
        """Map env indices to (remote, local_slot) pairs."""
        indices = self._get_indices(indices)
        out = []
        for i in indices:
            for w, group in enumerate(self.groups):
                if i in group:
                    out.append((self.remotes[w], group.index(i)))
                    break
        return out

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        targets = self._remotes_for(indices)
        cache = {}
        for remote, _ in targets:
            if id(remote) not in cache:
                remote.send(("get_attr", attr_name))
                cache[id(remote)] = remote.recv()
        return [cache[id(remote)][slot] for remote, slot in targets]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        seen = set()
        for remote, _ in self._remotes_for(indices):
            if id(remote) not in seen:
                remote.send(("set_attr", (attr_name, value)))
                remote.recv()
                seen.add(id(remote))

    def env_method(self, method_name: str, *method_args,
                   indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        targets = self._remotes_for(indices)
        cache = {}
        for remote, _ in targets:
            if id(remote) not in cache:
                remote.send(("env_method", (method_name, method_args, method_kwargs)))
                cache[id(remote)] = remote.recv()
        return [cache[id(remote)][slot] for remote, slot in targets]

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> List[bool]:
        targets = self._remotes_for(indices)
        cache = {}
        for remote, _ in targets:
            if id(remote) not in cache:
                remote.send(("is_wrapped", wrapper_class))
                cache[id(remote)] = remote.recv()
        return [cache[id(remote)][slot] for remote, slot in targets]

    # -- diagnostics -------------------------------------------------------

    def render_stats(self) -> List[dict]:
        """Per-worker renderer timings -- frames, raster ms, p99, fps."""
        for remote in self.remotes:
            remote.send(("render_stats", None))
        return [remote.recv() for remote in self.remotes]
