#!/usr/bin/env bash
set -euo pipefail

# Speedrun-style EBT training on 8 GPUs.
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
# --standalone --nproc_per_node=4 
torchrun train.py \
  --dataset=openwebtext \
  --data_dir=nanoGPT/data/openwebtext \
  --out_dir=out_ebt_openwebtext \
  --max_steps=100000 \
  --max_scheduling_steps=100000 \
  --warm_up_steps=2000 \
  --eval_interval=100 \
  --batch_size_per_device=24 \
  --accumulate_grad_batches=20 \
  --context_length=512 \
  --num_transformer_blocks=8 \
  --multiheaded_attention_heads=8 \
  --embedding_dim=512 \
  --tokenizer=gpt2 \
  --gpus=8 \
  --distributed_strategy=ddp \
  --compile=True \
  --mcmc_num_steps=1 \
  --mcmc_replay_buffer_size=48 \
  --mcmc_step_size=60.0 \

