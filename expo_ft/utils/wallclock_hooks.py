"""Optional wall-clock instrumentation hooks (wallclock-vla-rl).

Zero-cost when WALLCLOCK_TRACE is unset or the `wallclock` package is not on
PYTHONPATH: every call becomes a no-op, so the fork keeps working standalone.

Env overrides (applied by apply_env_overrides, so a sweep can vary knobs
without editing YAML):
    WALLCLOCK_UTD        -> cfg.utd_ratio
    WALLCLOCK_SEED       -> cfg.seed
    WALLCLOCK_REPLAN     -> cfg.replan_steps
    WALLCLOCK_BATCH      -> cfg.batch_size
    WALLCLOCK_MAX_STEPS  -> cfg.max_steps
"""

from __future__ import annotations

import contextlib
import logging
import os

try:
    from wallclock.timing import Phase, WallClockLogger
except ImportError:  # fork used without wallclock-vla-rl
    Phase = None
    WallClockLogger = None

_ENV_OVERRIDES = {
    "WALLCLOCK_UTD": ("utd_ratio", int),
    "WALLCLOCK_SEED": ("seed", int),
    "WALLCLOCK_REPLAN": ("replan_steps", int),
    "WALLCLOCK_BATCH": ("batch_size", int),
    "WALLCLOCK_MAX_STEPS": ("max_steps", int),
    "WALLCLOCK_EVAL_EPISODES": ("rl_eval_episodes", int),
    "WALLCLOCK_EVAL_INTERVAL": ("rl_eval_interval", int),
}


def apply_env_overrides(cfg) -> None:
    for var, (field, cast) in _ENV_OVERRIDES.items():
        val = os.environ.get(var)
        if val:
            setattr(cfg, field, cast(val))
            logging.info("[wallclock] override %s=%s from %s", field, val, var)


class WallClock:
    """Thin wrapper: string phase names, counters, no-op when disabled."""

    def __init__(self, logger):
        self._log = logger
        self.n_updates = 0       # cumulative gradient steps
        self.enabled = logger is not None

    def phase(self, name: str, thread: str, step=None, update=None, extra=None):
        if self._log is None:
            return contextlib.nullcontext()
        return self._log.phase(Phase(name), thread=thread, step=step, update=update, extra=extra)

    def scalar(self, name: str, value, step=None) -> None:
        if self._log is not None:
            self._log.log_scalar(name, float(value), step=step)

    def elapsed(self) -> float:
        return 0.0 if self._log is None else self._log.elapsed()

    def close(self) -> None:
        if self._log is not None:
            self._log.close()


def make_wallclock(run_meta: dict) -> WallClock:
    path = os.environ.get("WALLCLOCK_TRACE")
    if not path or WallClockLogger is None:
        if path:
            logging.warning("[wallclock] WALLCLOCK_TRACE set but `wallclock` package not importable; disabled")
        return WallClock(None)
    logging.info("[wallclock] tracing to %s", path)
    return WallClock(WallClockLogger(path, run_meta=run_meta))
