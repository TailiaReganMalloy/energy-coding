#!/usr/bin/env python3
"""Scan a PyTorch checkpoint for NaN/Inf tensors.

Usage:
  python scan_checkpoint.py --ckpt out_ebt_openwebtext/ckpt_iter_910000.pt
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Tuple

import torch


def iter_tensors(obj, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_tensors(value, next_prefix)
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from iter_tensors(value, next_prefix)
        return


def is_finite_tensor(tensor: torch.Tensor) -> bool:
    return torch.isfinite(tensor).all().item()


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan a checkpoint for NaN/Inf tensors.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--max-print",
        type=int,
        default=20,
        help="Max number of non-finite tensors to print",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    total_tensors = 0
    nonfinite_tensors = 0

    print(f"Scanning checkpoint: {ckpt_path}")

    for name, tensor in iter_tensors(checkpoint):
        total_tensors += 1
        if not is_finite_tensor(tensor):
            nonfinite_tensors += 1
            if nonfinite_tensors <= args.max_print:
                flat = tensor.flatten()
                first_bad_idx = (~torch.isfinite(flat)).nonzero(as_tuple=False)[0].item()
                first_bad_val = flat[first_bad_idx].item()
                print(
                    "Non-finite tensor: {} shape={} dtype={} first_bad_idx={} first_bad_val={}".format(
                        name, tuple(tensor.shape), tensor.dtype, first_bad_idx, first_bad_val
                    )
                )

    print(f"Total tensors: {total_tensors}")
    print(f"Non-finite tensors: {nonfinite_tensors}")

    return 1 if nonfinite_tensors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
