# Run lerobot-eval reproducibly (same arguments as lerobot-eval).
#   1. Deterministic GPU kernels: cudnn.benchmark stays off (its timing-based kernel choice depends on GPU load).
#   2. Per-episode policy seed: torch is reseeded before every episode, so episodes don't depend on earlier ones
#      and a LIBERO-Plus episode gets the same policy noise as its vanilla twin.
# Usage: <venv>/bin/python deterministic_eval.py <lerobot-eval arguments>

import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; set before CUDA init

import torch  # noqa: E402

_CudnnModule = type(sys.modules["torch.backends.cudnn"])


class _DeterministicCudnn(_CudnnModule):
    benchmark = property(lambda self: False, lambda self, value: None)  # ignore lerobot-eval's benchmark = True


sys.modules["torch.backends.cudnn"].__class__ = _DeterministicCudnn
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

import lerobot.scripts.lerobot_eval as _lerobot_eval  # noqa: E402

# LIBERO-Plus only: {variant name: {"seed": ...}}, written by eval_combined.py
_episodes_path = os.environ.get("LIBERO_PLUS_EPISODES")
_EPISODES = json.load(open(_episodes_path)) if _episodes_path else {}
_orig_rollout = _lerobot_eval.rollout


def _rollout(env, *args, seeds=None, **kwargs):
    # Reseed the policy noise with this episode's seed (batch size 1)
    seed = seeds[0] if seeds else None
    episode = _EPISODES.get(env.call("task")[0]) if _EPISODES else None
    if episode is not None:
        seed = episode["seed"]  # seed of the vanilla twin
    if seed is not None:
        torch.manual_seed(seed)  # CPU and all CUDA generators
    return _orig_rollout(env, *args, seeds=seeds, **kwargs)


_lerobot_eval.rollout = _rollout

if __name__ == "__main__":
    sys.exit(_lerobot_eval.main())
