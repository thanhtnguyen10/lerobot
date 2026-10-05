# Per-episode overrides for LIBERO-Plus processes (loaded automatically via PYTHONPATH).
# LIBERO_PLUS_EPISODES -> {variant name: {"slot": k, "seed": s, "instruction": str | null}}; each variant runs as
# vanilla episode k of its task:
#   slot         start from stored initial state k (lerobot-eval would use 0; Objects Layout keeps its own state)
#   seed         reset with vanilla episode k's seed instead of the command-line seed
#   instruction  vanilla instruction instead of the file-name-derived one; null keeps LIBERO-Plus's text
# Must be sitecustomize: lerobot-eval builds environments in spawned worker processes.

import json
import os

import numpy as _np

# LIBERO-Plus fog noise uses np.float_, removed in NumPy 2
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64

_path = os.environ.get("LIBERO_PLUS_EPISODES")
if _path:
    with open(_path) as f:
        _EPISODES = json.load(f)

    import lerobot.envs.libero as _libero_env

    _orig_init = _libero_env.LiberoEnv.__init__
    _orig_reset = _libero_env.LiberoEnv.reset

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._episode = _EPISODES.get(self.task)
        if self._episode is None:
            return
        if self.init_states:
            self.init_state_id = self._episode["slot"]
        if self._episode["instruction"] is not None:
            self.task_description = self._episode["instruction"]

    def _reset(self, seed=None, **kwargs):
        if self._episode is not None:
            seed = self._episode["seed"]
        return _orig_reset(self, seed=seed, **kwargs)

    _libero_env.LiberoEnv.__init__ = _init
    _libero_env.LiberoEnv.reset = _reset
