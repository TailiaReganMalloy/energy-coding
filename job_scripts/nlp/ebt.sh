#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Single-GPU run (override any defaults as needed)
# nembed 1536
# Single-process run (override any defaults as needed)
# nembed 1536
if [[ "$(uname -s)" == "Darwin" ]]; then
  # macOS: DDP/torchrun is not supported; use a single process (MPS if available)
  DEVICE_FLAG="--device=cpu"
  if python - <<'PY'
import torch
print(int(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
PY
  then
    DEVICE_FLAG="--device=mps"
  fi
  python train.py \
    "$DEVICE_FLAG" \
    --compile=False \
    --dataset=openwebtext \
    --data_dir=nanoGPT/data/openwebtext \
    --out_dir=out_ebt_openwebtext \
    --resume_latest=True \
    --max_iters=500000 \
    --lr_decay_iters=500000 \
    --warmup_iters=2000 \
    --eval_interval=500 \
    --batch_size=2 \
    --gradient_accumulation_steps=4 \
    --block_size=512 \
    --n_layer=8 \
    --n_head=8 \
    --n_embd=512 \
    --tokenizer=gpt2 \
    --mcmc_num_steps=2 \
    --mcmc_step_size=16.0 \
    --normalize_initial_condition=True \
    --clamp_futures_grad=True
else
  torchrun --standalone --nproc_per_node=4 train.py \
    --dataset=openwebtext \
    --data_dir=nanoGPT/data/openwebtext \
    --out_dir=out_ebt_openwebtext \
    --resume_latest=True \
    --max_iters=500000 \
    --lr_decay_iters=500000 \
    --warmup_iters=2000 \
    --eval_interval=500 \
    --batch_size=2 \
    --gradient_accumulation_steps=4 \
    --block_size=512 \
    --n_layer=8 \
    --n_head=8 \
    --n_embd=512 \
    --tokenizer=gpt2 \
    --compile=True \
    --mcmc_num_steps=2 \
    --mcmc_step_size=16.0 \
    --normalize_initial_condition=True \
    --clamp_futures_grad=True
fi
