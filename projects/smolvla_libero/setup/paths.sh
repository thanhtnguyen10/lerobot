# Paths and pinned versions; sourced by setup_envs.sh and train/train.sh (eval_vanilla.py mirrors the paths).
SETUP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(dirname "$SETUP_DIR")
REPO=$(cd "$PROJECT_DIR/../.." && pwd)

# Environments
VENV=$REPO/.venv                              # LeRobot env (uv sync --locked): training + vanilla LIBERO
PLUS_VENV=$REPO/.venv-liberoplus              # same env, LIBERO-Plus fork instead of hf-libero

# Third-party (git-ignored)
THIRD_PARTY=$REPO/third_party
LIBERO_PLUS_DIR=$THIRD_PARTY/LIBERO-plus
MAGICK_HOME=$THIRD_PARTY/imagemagick          # ImageMagick for LIBERO-Plus sensor noise (conda, no sudo)
MAGICK_PRELOAD=$MAGICK_HOME/lib/libjpeg.so.8:$MAGICK_HOME/lib/libglib-2.0.so.0:$MAGICK_HOME/lib/libgobject-2.0.so.0
LIBERO_CONFIG=$THIRD_PARTY/libero_config      # instead of ~/.libero
LIBERO_PLUS_CONFIG=$THIRD_PARTY/libero_plus_config
OUTPUTS=$REPO/outputs                         # checkpoints and eval results (git-ignored)

# Pinned versions
LIBERO_PLUS_SHA=4976dc3                        # same as LeRobot's Dockerfile.benchmark.libero_plus
DATASET_REVISION=a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4  # lerobot/libero
PYTHON_VERSION=3.12
EXTRAS=(--extra smolvla --extra libero --extra training)
