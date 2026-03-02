"""
Print a concrete example of how instruction data is fed into EBT during train.py.

python test_cli.py --ckpt_path ./out_ebt_instruct/ckpt_iter_65000.pt --mcmc_num_steps 64

This mirrors the train.py data path:
1) read tokenized memmap (train.bin)
2) sample a block of length block_size + 1
3) build batch as {"input_ids": x.unsqueeze(1)}
4) derive model input and next-token targets

Example:
  python3 test_instruct.py \
    --data_dir self-instruct/data/finetuning/self_instruct_221203 \
    --tokenizer gpt2 \
    --block_size 512 \
    --batch_size 1 > output2.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer


def get_batch(data_dir: Path, block_size: int, batch_size: int, start_index: int | None = None):
    train_path = data_dir / "train.bin"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run train.py once to build binaries.")

    data = np.memmap(train_path, dtype=np.uint16, mode="r")
    if len(data) <= block_size + 1:
        raise ValueError(
            f"train.bin too small for block_size={block_size}. Tokens available={len(data)}"
        )

    if start_index is not None:
        max_start = len(data) - (block_size + 1)
        if not 0 <= start_index <= max_start:
            raise ValueError(f"start_index must be in [0, {max_start}] (got {start_index})")
        starts = torch.full((batch_size,), start_index, dtype=torch.long)
    else:
        starts = torch.randint(len(data) - (block_size + 1), (batch_size,))

    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size + 1]).astype(np.int64)) for i in starts]
    )
    return {"input_ids": x.unsqueeze(1)}, starts.tolist()


def decode_safe(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Inspect instruction batch format used by train.py")
    parser.add_argument(
        "--data_dir",
        default="self-instruct/data/finetuning/self_instruct_221203",
        help="Directory containing train.bin/val.bin",
    )
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)

    batch, starts = get_batch(data_dir, args.block_size, args.batch_size, args.start_index)
    input_ids = batch["input_ids"]  # [B, 1, S+1]

    print("=== train.py-style batch ===")
    print(f"data_dir: {data_dir}")
    print(f"sample starts: {starts}")
    print(f"batch['input_ids'].shape: {tuple(input_ids.shape)}")

    sample = input_ids[0, 0]  # [S+1]
    model_input = sample[:-1]  # x['input_ids'].squeeze(1)[:, :-1]
    targets = sample[1:]  # x['input_ids'].squeeze(1)[:, 1:]

    print("\n=== tensor preview ===")
    print(f"raw sample token ids (first 48): {sample[:48].tolist()}")
    print(f"model_input shape: {tuple(model_input.shape)}")
    print(f"targets shape: {tuple(targets.shape)}")

    print("\n=== decoded sample (exact window fed to training) ===")
    print(decode_safe(tokenizer, sample.tolist()))

    print("\n=== decoded model_input (x[:, :-1]) ===")
    print(decode_safe(tokenizer, model_input.tolist()))

    print("\n=== decoded targets (x[:, 1:]) ===")
    print(decode_safe(tokenizer, targets.tolist()))


if __name__ == "__main__":
    main()
