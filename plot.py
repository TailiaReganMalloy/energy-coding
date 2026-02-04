#!/usr/bin/env python3
"""
Plot training/validation losses for an EBT run and export loss tables with metadata.
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    ckpt_iter_paths = []
    for path in output_dir.glob("ckpt_iter_*.pt"):
        try:
            iter_num = int(path.stem.split("ckpt_iter_", 1)[1])
        except (IndexError, ValueError):
            continue
        ckpt_iter_paths.append((iter_num, path))
    if ckpt_iter_paths:
        ckpt_iter_paths.sort(key=lambda item: item[0])
        return ckpt_iter_paths[-1][1]
    ckpt_path = output_dir / "ckpt.pt"
    return ckpt_path if ckpt_path.exists() else None


def load_loss_log(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    loss_log_path = output_dir / "losses.pkl"
    if not loss_log_path.exists():
        raise FileNotFoundError(f"Missing loss log at {loss_log_path}")
    with loss_log_path.open("rb") as f:
        loss_log = pickle.load(f)
    if not isinstance(loss_log, dict):
        raise ValueError("losses.pkl did not contain a dictionary")
    loss_log.setdefault("train", [])
    loss_log.setdefault("eval", [])
    return loss_log


def attach_metadata(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        enriched_row = dict(row)
        enriched_row.update(metadata)
        enriched.append(enriched_row)
    return enriched


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            values = []
            for key in keys:
                value = row.get(key, "")
                values.append(str(value))
            f.write(",".join(values) + "\n")


def plot_losses(output_dir: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    if train_rows:
        train_iters = [row["iter"] for row in train_rows if "iter" in row]
        train_losses = [row["loss"] for row in train_rows if "loss" in row]
        ax.plot(train_iters, train_losses, label="train")

    if eval_rows:
        eval_iters = [row["iter"] for row in eval_rows if "iter" in row]
        val_losses = [row.get("val") for row in eval_rows if "val" in row]
        ax.plot(eval_iters, val_losses, label="val")

    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title("EBT training/validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = output_dir / "loss_plot.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="out_ebt_openwebtext")
    parser.add_argument("--model", default="ebt")
    parser.add_argument("--dataset", default="openwebtext")
    args = parser.parse_args()

    output_dir = Path(args.out_dir)
    loss_log = load_loss_log(output_dir)

    ckpt_path = find_latest_checkpoint(output_dir)
    config: dict[str, Any] = {}
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    metadata = {"model": args.model, "dataset": args.dataset}
    if isinstance(config, dict):
        metadata.update(config)

    train_rows = attach_metadata(loss_log.get("train", []), metadata)
    eval_rows = attach_metadata(loss_log.get("eval", []), metadata)

    train_csv = output_dir / "train_losses.csv"
    val_csv = output_dir / "val_losses.csv"
    write_csv(train_csv, train_rows)
    write_csv(val_csv, eval_rows)

    plot_path = plot_losses(output_dir, train_rows, eval_rows)
    print(f"Wrote {train_csv}")
    print(f"Wrote {val_csv}")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
