#!/usr/bin/env bash
set -euo pipefail

# Speedrun-style EBT training on 4 GPUs.
# Mirrors the high-level flow in nanochat/runs/speedrun.sh but uses EBT train.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS=1

# Optional: keep HF cache local to the repo to avoid shared cache contention.
export HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT_DIR/.hf_cache}"

# -----------------------------------------------------------------------------
# Dataset prep (OpenWebText) will be auto-triggered by train.py if missing.
# -----------------------------------------------------------------------------

# EBT pretraining (OpenWebText) on 4 GPUs.
# Adjust hyperparameters as needed for your hardware.
torchrun --standalone --nproc_per_node=4 train.py \
  --dataset=openwebtext \
  --data_dir=nanoGPT/data/openwebtext \
  --out_dir=out_ebt_openwebtext \
  --max_iters=100000 \
  --lr_decay_iters=100000 \
  --warmup_iters=2000 \
  --eval_interval=100 \
  --batch_size=24 \
  --gradient_accumulation_steps=20 \
  --block_size=512 \
  --n_layer=8 \
  --n_head=8 \
  --n_embd=512 \
  --tokenizer=gpt2 \
  --compile=True \
  --mcmc_num_steps=1 \
  --mcmc_step_size=60.0
