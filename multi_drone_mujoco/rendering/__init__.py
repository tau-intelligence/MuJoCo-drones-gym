"""Shared static-scene rendering.

One GL context serving many environments, instead of one context per env.

Valid only when every participating env has an identical, *static* world and a
single drone -- the caller asserts this explicitly; it is never auto-detected,
because a silently-wrong assertion produces wrong pixels rather than an error.

See multi_drone_mujoco/bench/ for the verification and benchmark harnesses.
"""

from .shared import SharedStaticRenderer, RenderStats

__all__ = ["SharedStaticRenderer", "RenderStats"]
