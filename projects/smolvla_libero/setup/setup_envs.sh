#!/usr/bin/env bash
# Build everything the pipeline needs, inside the repo (git-ignored):
#   .venv              LeRobot env from uv.lock (training + vanilla LIBERO)
#   .venv-liberoplus   same env, LIBERO-Plus fork instead of hf-libero
#   third_party/       LIBERO-Plus @ pinned commit + assets, ImageMagick, LIBERO configs
# Re-running skips finished steps. Needs uv, conda, git, ~35 GB disk.
# Usage: bash projects/smolvla_libero/setup/setup_envs.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
cd "$REPO"
export CMAKE_POLICY_VERSION_MINIMUM=3.5   # egl-probe (hf-libero -> robomimic) targets CMake < 3.5

echo "== 1/5 LeRobot environment ($VENV)"
uv sync --locked --python "$PYTHON_VERSION" "${EXTRAS[@]}"

echo "== 2/5 LIBERO config for vanilla LIBERO"
LIBERO_PKG=$("$VENV/bin/python" -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('libero').submodule_search_locations[0]))")/libero/libero
mkdir -p "$LIBERO_CONFIG"
printf "assets: %s/assets\nbddl_files: %s/bddl_files\nbenchmark_root: %s\ndatasets: %s/../datasets\ninit_states: %s/init_files\n" \
  "$LIBERO_PKG" "$LIBERO_PKG" "$LIBERO_PKG" "$LIBERO_PKG" "$LIBERO_PKG" > "$LIBERO_CONFIG/config.yaml"

echo "== 3/5 LIBERO-Plus fork @ $LIBERO_PLUS_SHA and assets"
mkdir -p "$THIRD_PARTY"
[ -d "$LIBERO_PLUS_DIR/.git" ] || git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_DIR"
git -C "$LIBERO_PLUS_DIR" checkout -q "$LIBERO_PLUS_SHA"
R=$LIBERO_PLUS_DIR/libero/libero
if [ ! -d "$R/assets" ]; then
  DL=$(mktemp -d -p "$THIRD_PARTY")
  "$VENV/bin/python" -c "from huggingface_hub import hf_hub_download as d; d(repo_id='Sylvest/LIBERO-plus', repo_type='dataset', filename='assets.zip', local_dir='$DL')"
  unzip -q "$DL/assets.zip" -d "$DL/x"
  mv "$(find "$DL/x" -type d -name assets | head -1)" "$R/assets"
  rm -rf "$DL"
fi
mkdir -p "$LIBERO_PLUS_CONFIG"
printf "assets: %s/assets\nbddl_files: %s/bddl_files\nbenchmark_root: %s\ndatasets: %s/../datasets\ninit_states: %s/init_files\n" \
  "$R" "$R" "$R" "$R" "$R" > "$LIBERO_PLUS_CONFIG/config.yaml"

echo "== 4/5 ImageMagick ($MAGICK_HOME)"
[ -d "$MAGICK_HOME" ] || conda create -y -q -p "$MAGICK_HOME" -c conda-forge --override-channels imagemagick

echo "== 5/5 LIBERO-Plus environment ($PLUS_VENV)"
UV_PROJECT_ENVIRONMENT=$PLUS_VENV uv sync --locked --python "$PYTHON_VERSION" "${EXTRAS[@]}"
uv pip uninstall --python "$PLUS_VENV/bin/python" hf-libero   # the fork replaces the `libero` package
uv pip install --python "$PLUS_VENV/bin/python" "Wand==0.6.13" "scikit-image==0.25.2" "usd-core==26.8" "gym==0.26.2"
uv pip install --python "$PLUS_VENV/bin/python" --no-deps -e "$LIBERO_PLUS_DIR"   # also needs PYTHONPATH at runtime

echo "== check"
LIBERO_CONFIG_PATH=$LIBERO_CONFIG "$VENV/bin/python" -c \
  "import libero, mujoco, torch; print('vanilla: libero from', list(libero.__path__)[0], '| mujoco', mujoco.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
# full lerobot-eval stack, as the evaluation imports it, so library clashes show up here
PYTHONPATH=$LIBERO_PLUS_DIR LIBERO_CONFIG_PATH=$LIBERO_PLUS_CONFIG MAGICK_HOME=$MAGICK_HOME LD_PRELOAD=$MAGICK_PRELOAD \
  "$PLUS_VENV/bin/python" -c \
  "import lerobot.scripts.lerobot_eval, libero.libero.envs, mujoco, wand.image, libero; print('plus:    libero from', list(libero.__path__)[0], '| mujoco', mujoco.__version__)"
