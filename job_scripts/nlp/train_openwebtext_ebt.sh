#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Single-GPU run (override any defaults as needed)
python train.py \
  --dataset=openwebtext \
  --data_dir=nanoGPT/data/openwebtext \
  --out_dir=out_ebt_openwebtext \
  --batch_size=12 \
  --block_size=1024 \
  --n_layer=12 \
  --n_head=12 \
  --n_embd=768 \
  --tokenizer=gpt2 \
  --mcmc_num_steps=2 \
  --mcmc_step_size=60.0

# DDP example (4 GPUs):
# torchrun --standalone --nproc_per_node=4 train.py \
#   --dataset=openwebtext \
#   --data_dir=nanoGPT/data/openwebtext \
#   --out_dir=out_ebt_openwebtext \
#   --batch_size=12 \
#   --block_size=1024 \
#   --n_layer=12 \
#   --n_head=12 \
#   --n_embd=768 \
#   --tokenizer=gpt2 \
#   --mcmc_num_steps=2 \
#   --mcmc_step_size=60.0
