#!/usr/bin/env bash
# Fine-tune SmolVLA (0.45B) on LIBERO, SmolVLA paper recipe (Shukor et al., 2025, Sec. 4.3).
# Usage: RUN=<run_name> [SEED=1000] nohup bash projects/smolvla_libero/train/train.sh [extra flags] > train.log 2>&1 &
#   extra flags override the ones below (e.g. --steps=200 --wandb.enable=false for a quick test)
#   checkpoints: outputs/<run_name>/checkpoints/{020000,...,100000}/pretrained_model
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../setup/paths.sh"
: "${RUN:?set RUN=<run_name> (outputs/<run_name> must not exist yet)}"
SEED=${SEED:-1000}

ARGS=(
  # Policy: SmolVLM2 init, VLM frozen, only the action expert trained
  --policy.type=smolvla
  --policy.load_vlm_weights=true
  --policy.train_expert_only=true
  --policy.freeze_vision_encoder=true
  --policy.chunk_size=50                      # action chunk n = 50
  --policy.n_action_steps=50
  --policy.pad_language_to=max_length         # fixed sequence length
  --policy.push_to_hub=false

  # Optimization: 100k steps, batch 64, cosine LR 1e-4 -> 2.5e-6 over all steps
  --steps=100000
  --batch_size=64
  --policy.scheduler_decay_steps=100000
  --seed="$SEED"

  # Speed: bf16 + torch.compile (paper)
  --accelerator.mixed_precision=bf16
  --policy.compile_model=true

  # Data
  --dataset.repo_id=lerobot/libero
  --dataset.revision="$DATASET_REVISION"
  --dataset.video_backend=torchcodec
  --num_workers=0                             # /dev/shm is 64 MB here; remove with a larger /dev/shm (~2x faster)

  # Output and logging
  --output_dir="$OUTPUTS/$RUN"
  --job_name="$RUN"
  --wandb.enable=true
  --wandb.project=smolvla-libero-repro
  --wandb.disable_artifact=true
)

"$VENV/bin/lerobot-train" "${ARGS[@]}" "$@"
