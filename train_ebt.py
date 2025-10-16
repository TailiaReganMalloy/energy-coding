"""Train an Energy-Based Transformer (EBT) on the LiveBench coding dataset.

This script prepares the LiveBench coding completion split, instantiates the
EBT NLP model, and launches a lightweight PyTorch Lightning training loop.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping, Tuple

# Allow Torch to transparently fall back to CPU for ops not yet implemented on MPS.
if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from EBT.data.nlp.collator import NLP_HF_Collator
from EBT.data.nlp.programming_dataloader import ProgrammingDataset
from EBT.model.nlp.ebt import EBT_NLP


def _ensure_livebench_cached(split: str, *, max_samples: int | None = None) -> Path:
    """Materialise the LiveBench coding dataset as a datasets "save_to_disk" cache.

    ProgrammingDataset expects a HuggingFace dataset directory. The LiveBench
    repo ships JSONL files, so we lazily convert them on first run.
    """

    raw_json = (
        PROJECT_ROOT
        / "livebench"
        / "livebench"
        / "data"
        / "live_bench"
        / "coding"
        / "coding_completion"
        / "question.jsonl"
    )
    if not raw_json.exists():
        raise FileNotFoundError(
            "LiveBench coding completion JSONL not found: " f"{raw_json}"
        )

    cache_name = "hf_cache_full" if max_samples is None else f"hf_cache_{max_samples}"
    cache_dir = raw_json.parent / cache_name
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_dict = load_dataset("json", data_files={split: str(raw_json)})
        dataset = dataset_dict[split]
        total = min(max_samples, len(dataset)) if max_samples is not None else len(dataset)
        subset = dataset.select(range(total)) if total < len(dataset) else dataset
        subset = subset.map(
            lambda example: example,
            desc=f"Caching {total} LiveBench coding sample{'s' if total != 1 else ''}",
            load_from_cache_file=False,
        )
        subset.save_to_disk(str(cache_dir))

    return cache_dir


def _default_hparams(dataset_dir: Path, limit_samples: int | None) -> Dict[str, object]:
    """Return a baseline hyperparameter dictionary for LiveBench training."""

    base: Dict[str, object] = {
        # optimisation
        "lr": 3e-4,
        "batch_size_per_device": 2,
        "num_workers_per_gpu": 0,
        "max_steps": 200,
        "gradient_clip_val": 1.0,
        # dataset
        "dataset_dir": str(dataset_dir),
        "dataset_name": "livebench_coding",
        "dataset_split": "test",
        "dataset_map_workers": 1,
        "max_dataset_samples": limit_samples,
        "context_length": 512,
        "pretokenize_dataset": False,
        "tokenizer": "EleutherAI/gpt-neox-20b",
        # model choice + size
        "model_name": "ebt",
        "embedding_dim": 384,
        "num_transformer_blocks": 6,
        "multiheaded_attention_heads": 6,
        "ffn_dim_multiplier": 1,
        "weight_initialization_method": "xavier",
        "weight_initialization_gain": 1.0,
        # execution flags
        "execution_mode": "pretrain",
        "debug_unused_parameters": False,
        "mcmc_replay_buffer": False,
    }

    ebt_specific: Dict[str, object] = {
        "mcmc_step_size": 400.0,
        "mcmc_step_size_lr_multiplier": 1200.0,
        "mcmc_num_steps": 2,
        "ebt_type": "time_embed",
        "normalize_initial_condition": True,
        "denoising_initial_condition": "random_noise",
        "mcmc_step_size_learnable": True,
        "no_mcmc_detach": False,
        "ebt_norm": "rms",
        "ebt_act_func": "silu",
        "dyt_alpha_init": 0.5,
        "mcmc_replay_buffer_size": 1024,
        "mcmc_replay_buffer_sample_bs_percent": 0.25,
        "gaussian_random_noise_scaling": 1.0,
        "normalize_initial_condition_only_first_step": False,
        "randomize_mcmc_step_size_scale": 1.0,
        "randomize_mcmc_num_steps": 0,
        "randomize_mcmc_num_steps_min": 0,
        "randomize_mcmc_num_steps_final_landscape": False,
        "langevin_dynamics_noise": 0.0,
        "langevin_dynamics_noise_learnable": False,
        "vocab_to_embed_uses_prob_dist": False,
        "num_modality_processing_mlp_layers": 1,
        "truncate_mcmc": False,
        "clamp_futures_grad": False,
        "clamp_futures_grad_max_change": 9.0,
        "absolute_clamp": 0.0,
        "clamp_max_after_warm_up": 0.0,
        "sharpen_predicted_distribution": 0.0,
        "reconstruction_coeff": 1.0,
        "contrastive_loss": False,
        "contrastive_loss_coeff": 5e-4,
        "soften_target_prob_dist": 0.0,
    }

    return {**base, **ebt_specific}


def _resolve_device_configuration(
    requested_accelerator: str, requested_devices: int, requested_precision: str
) -> Tuple[str, int, str]:
    """Resolve accelerator, device count, and precision for the current hardware.

    Preference order when ``requested_accelerator`` is ``"auto"``:
    Apple Metal (MPS) → CUDA GPU → CPU.
    Ensures macOS users transparently run on the Apple GPU while keeping
    configuration overridable via command-line flags.
    """

    accelerator = requested_accelerator
    devices = requested_devices
    precision = requested_precision

    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if accelerator == "auto":
        if mps_available:
            accelerator = "mps"
            devices = 1
        elif torch.cuda.is_available():
            accelerator = "gpu"
        else:
            accelerator = "cpu"

    if accelerator == "mps":
        devices = 1  # PyTorch currently exposes a single logical MPS device.
        torch.set_float32_matmul_precision("medium")
        if precision in {"16-mixed", "bf16-mixed", "bf16-true"}:
            # Mixed precision is not yet supported on MPS; fall back to float32.
            precision = "32-true"
            print("[train_ebt] MPS does not support mixed precision; using 32-true.")
        print("[train_ebt] Using Apple Metal (MPS) accelerator for training.")
    elif accelerator == "gpu" and torch.cuda.is_available():
        # Default to efficient mixed precision on CUDA when not explicitly overridden.
        if precision == "32-true":
            precision = "16-mixed"
            print("[train_ebt] CUDA accelerator detected; defaulting to 16-mixed precision.")

    return accelerator, devices, precision


class LiveBenchEBTModule(pl.LightningModule):
    """Minimal Lightning wrapper around the EBT NLP model."""

    def __init__(self, hparams: Mapping[str, object]):
        super().__init__()
        hparam_dict = dict(hparams)
        self.save_hyperparameters(hparam_dict)
        self._hparams = SimpleNamespace(**hparam_dict)

        self.model = EBT_NLP(self._hparams)
        self.dataset = ProgrammingDataset(self._hparams)
        self._collate_fn = NLP_HF_Collator(self._hparams)

    @property
    def effective_batch_size(self) -> int:
        return self._hparams.batch_size_per_device

    def training_step(self, batch: Mapping[str, torch.Tensor], batch_idx: int):  # type: ignore[override]
        metrics = self.model.forward_loss_wrapper(batch, phase="train")
        loss = metrics["loss"]
        loggable: Dict[str, torch.Tensor] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                loggable[f"train_{key}"] = value.detach()
            elif isinstance(value, (int, float)):
                loggable[f"train_{key}"] = torch.tensor(value, device=loss.device)

        self.log_dict(loggable, on_step=True, on_epoch=True, batch_size=batch["input_ids"].size(0))
        return loss

    def configure_optimizers(self):  # type: ignore[override]
        return torch.optim.AdamW(self.model.parameters(), lr=self._hparams.lr)

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        num_workers = self._hparams.num_workers_per_gpu
        if torch.cuda.is_available():
            num_workers *= max(1, torch.cuda.device_count())

        accelerator = getattr(self._hparams, "accelerator", "cpu")
        pin_memory = accelerator == "gpu" and torch.cuda.is_available()

        return DataLoader(
            self.dataset,
            batch_size=self._hparams.batch_size_per_device,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EBT model on LiveBench coding")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum optimisation steps")
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum optimisation epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=1,
        help="Optional cap on the number of training examples (default: 1 for smoke testing)",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Torch accelerator to use (auto|cpu|gpu|mps)",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to train on")
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="PyTorch Lightning precision flag (e.g., 32-true, bf16-true)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(42, workers=True)

    dataset_dir = _ensure_livebench_cached("test", max_samples=args.limit_samples)
    hparams = _default_hparams(dataset_dir, args.limit_samples)
    hparams["max_steps"] = args.max_steps
    hparams["max_epochs"] = args.max_epochs
    hparams["batch_size_per_device"] = args.batch_size

    accelerator, devices, precision = _resolve_device_configuration(
        args.accelerator, args.devices, args.precision
    )
    hparams["accelerator"] = accelerator

    lightning_module = LiveBenchEBTModule(hparams)

    checkpointing = ModelCheckpoint(
        dirpath=PROJECT_ROOT / "logs" / "livebench_coding",
        save_top_k=1,
        save_last=True,
        monitor="train_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        precision=precision,
        gradient_clip_val=hparams["gradient_clip_val"],
        log_every_n_steps=1,
        callbacks=[checkpointing],
        enable_checkpointing=True,
    )

    trainer.fit(lightning_module)


if __name__ == "__main__":
    main()
