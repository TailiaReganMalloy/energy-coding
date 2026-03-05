"""EBT fine-tuning on The Stack (python).

Example:
  python finetuning/train.py --ckpt_path out_ebt_openwebtext/ckpt_iter_135000.pt
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "EBT"))

from model.nlp.ebt import EBT_NLP


def normalize_state_dict_keys(state_dict):
	if not any(key.startswith("_orig_mod.") for key in state_dict):
		return state_dict
	return {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}


def build_hparams(config, overrides):
	block_size = overrides.block_size or config.get("block_size", config.get("context_length", 256))
	return SimpleNamespace(
		modality="NLP",
		model_name=config.get("model_name", "ebt"),
		tokenizer=config.get("tokenizer", overrides.tokenizer),
		context_length=block_size,
		num_transformer_blocks=config.get("n_layer", config.get("num_transformer_blocks", 6)),
		multiheaded_attention_heads=config.get("n_head", config.get("multiheaded_attention_heads", 6)),
		embedding_dim=config.get("n_embd", config.get("embedding_dim", 384)),
		ffn_dim_multiplier=config.get("ffn_dim_multiplier", None),
		batch_size_per_device=overrides.batch_size,
		ebt_type=config.get("ebt_type", "default"),
		ebt_norm=config.get("ebt_norm", "rms"),
		ebt_act_func=config.get("ebt_act_func", "silu"),
		dyt_alpha_init=config.get("dyt_alpha_init", 0.5),
		weight_initialization_method=config.get("weight_initialization_method", "xavier"),
		weight_initialization_gain=config.get("weight_initialization_gain", 1.0),
		mcmc_num_steps=overrides.mcmc_num_steps if overrides.mcmc_num_steps is not None else config.get("mcmc_num_steps", 1),
		mcmc_step_size=overrides.mcmc_step_size if overrides.mcmc_step_size is not None else config.get("mcmc_step_size", 60.0),
		mcmc_step_size_learnable=config.get("mcmc_step_size_learnable", False),
		langevin_dynamics_noise=config.get("langevin_dynamics_noise", 0.0),
		langevin_dynamics_noise_learnable=config.get("langevin_dynamics_noise_learnable", False),
		randomize_mcmc_step_size_scale=config.get("randomize_mcmc_step_size_scale", 1.0),
		randomize_mcmc_num_steps=config.get("randomize_mcmc_num_steps", 0),
		randomize_mcmc_num_steps_final_landscape=config.get("randomize_mcmc_num_steps_final_landscape", False),
		randomize_mcmc_num_steps_min=config.get("randomize_mcmc_num_steps_min", 0),
		denoising_initial_condition=config.get("denoising_initial_condition", "random_noise"),
		gaussian_random_noise_scaling=config.get("gaussian_random_noise_scaling", 1.0),
		normalize_initial_condition=config.get("normalize_initial_condition", False),
		normalize_initial_condition_only_first_step=config.get("normalize_initial_condition_only_first_step", False),
		vocab_to_embed_uses_prob_dist=config.get("vocab_to_embed_uses_prob_dist", False),
		num_modality_processing_mlp_layers=config.get("num_modality_processing_mlp_layers", 1),
		learnable_process_memory=config.get("learnable_process_memory", False),
		process_memory_type=config.get("process_memory_type", None),
		process_memory_linear_layer=config.get("process_memory_linear_layer", False),
		clamp_futures_grad=config.get("clamp_futures_grad", False),
		clamp_futures_grad_max_change=config.get("clamp_futures_grad_max_change", 9.0),
		absolute_clamp=config.get("absolute_clamp", 0.0),
		clamp_max_after_warm_up=config.get("clamp_max_after_warm_up", 0.0),
		sharpen_predicted_distribution=config.get("sharpen_predicted_distribution", 0.0),
		truncate_mcmc=config.get("truncate_mcmc", False),
		no_mcmc_detach=config.get("no_mcmc_detach", False),
		contrastive_loss=config.get("contrastive_loss", False),
		contrastive_loss_coeff=config.get("contrastive_loss_coeff", 0.0005),
		discrete_contrastive_loss_true_logit_val=config.get("discrete_contrastive_loss_true_logit_val", 0.0),
		soften_target_prob_dist=config.get("soften_target_prob_dist", 0.0),
		reconstruction_coeff=config.get("reconstruction_coeff", 1.0),
		mcmc_replay_buffer=False,
		execution_mode="pretrain",
		debug_unused_parameters=False,
	)


def iter_token_chunks(dataset, tokenizer, block_size):
	for record in dataset:
		text = record.get("content") or ""
		if not text:
			continue
		ids = tokenizer.encode(text)
		if len(ids) < block_size + 1:
			continue
		for start in range(0, len(ids) - block_size, block_size):
			chunk = ids[start : start + block_size + 1]
			if len(chunk) == block_size + 1:
				yield torch.tensor(chunk, dtype=torch.long)


def batch_iter(chunks, batch_size):
	batch = []
	for chunk in chunks:
		batch.append(chunk)
		if len(batch) == batch_size:
			x = torch.stack(batch)
			yield {"input_ids": x.unsqueeze(1)}
			batch = []


def main():
	parser = argparse.ArgumentParser(description="EBT fine-tuning on The Stack (python)")
	parser.add_argument("--ckpt_path", required=True, help="Path to ckpt_iter_XXXX.pt or ckpt.pt")
	parser.add_argument("--dataset", default="bigcode/the-stack", help="Hugging Face dataset name")
	parser.add_argument("--data_dir", default="data/python", help="Dataset subdir for The Stack")
	parser.add_argument("--split", default="train", help="Dataset split")
	parser.add_argument("--streaming", action="store_true", default=True)
	parser.add_argument("--take", type=int, default=100, help="Number of samples to take when streaming")
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--block_size", type=int, default=None)
	parser.add_argument("--learning_rate", type=float, default=6e-5)
	parser.add_argument("--max_steps", type=int, default=100)
	parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
	parser.add_argument("--tokenizer", default="gpt2")
	parser.add_argument("--mcmc_num_steps", type=int, default=None)
	parser.add_argument("--mcmc_step_size", type=float, default=None)
	parser.add_argument("--infer_temp", help="[Inference] Sampling temperature (higher = more random)", type=float, default=0.6)
	parser.add_argument("--infer_topp", help="[Inference] Nucleus sampling probability threshold", type=float, default=0.9)
	parser.add_argument("--infer_topk", help="[Inference] Limit sampling to top k most likely tokens", type=int, default=None)
	parser.add_argument("--infer_logprobs", help="[Inference] Return log probabilities of generated tokens", type=bool, default=True)

	args = parser.parse_args()

	if args.device == "auto":
		if torch.cuda.is_available():
			device = "cuda"
		elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			device = "mps"
		else:
			device = "cpu"
	else:
		device = args.device

	checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=False)
	config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
	if not isinstance(config, dict) or not config:
		raise ValueError("Checkpoint is missing a config dict. Provide a checkpoint saved from train.py.")

	hparams = build_hparams(config, args)
	model = EBT_NLP(hparams).to(device)
	state = normalize_state_dict_keys(checkpoint["model"])
	model.load_state_dict(state, strict=False)

	tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces=False)
	ds = load_dataset(args.dataset, data_dir=args.data_dir, split=args.split, streaming=args.streaming)
	if args.streaming:
		ds = ds.take(args.take)
	
	print("Loading samples...", flush=True)
	max_len = min(hparams.context_length, getattr(tokenizer, "model_max_length", hparams.context_length))
	samples = []
	for record in tqdm(ds, total=args.take, desc="Loading samples"):
		text = record.get("content") or ""
		ids = tokenizer.encode(text, truncation=True, max_length=max_len)
		if len(ids) < max_len:
			continue
		ids = ids[:max_len]
		samples.append(torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0))
		if len(samples) >= args.take:
			break

	if not samples:
		print("No valid samples found.")
		return

	print(f"Loaded {len(samples)} samples.")
	print("Running model.forward sweep...", flush=True)
	results = []
	seq_len = samples[0].shape[1]
	print("config block_size:", config.get("block_size", config.get("context_length")))
	with torch.no_grad():
		for mcmc_steps in range(0, 11):
			model.hparams.mcmc_num_steps = mcmc_steps
			if mcmc_steps == 0:
				results.append({"mcmc_steps": mcmc_steps, "energy": float("nan")})
				continue
			energies = []
			for input_ids in samples:
				predicted_distributions, predicted_energies = model.forward(
					input_ids,
					start_pos=0,
					learning=False,
					return_raw_logits=True,
					replay_buffer_logits=None,
					no_randomness=False,
				)
				if not predicted_energies:
					continue
				energy_preds = predicted_energies[-1].reshape(input_ids.shape[0], -1)
				if energy_preds.shape[1] == seq_len:
					energies.append(energy_preds[0, 0].item())
				elif energy_preds.shape[1] >= 2 * seq_len:
					energies.append(energy_preds[:, seq_len:seq_len * 2][0, 0].item())
			if energies:
				energy_val = float(sum(energies) / len(energies))
			else:
				energy_val = float("nan")
			results.append({"mcmc_steps": mcmc_steps, "energy": energy_val, "energies": energies})

	df = pd.DataFrame(results)
	plot_df = df.dropna(subset=["energy"])
	if len(plot_df) >= 2:
		reg = linregress(plot_df["mcmc_steps"], plot_df["energy"])
		r2 = reg.rvalue ** 2
		x_vals = plot_df["mcmc_steps"]
		y_fit = reg.intercept + reg.slope * x_vals
	else:
		reg = None
		r2 = float("nan")

	plt.figure(figsize=(8, 5))
	box_data = [row["energies"] for _, row in df.iterrows() if row["energies"]]
	box_positions = [row["mcmc_steps"] for _, row in df.iterrows() if row["energies"]]
	if box_data:
		plt.boxplot(box_data, positions=box_positions, widths=0.6)
	plt.scatter(df["mcmc_steps"], df["energy"], label="mean energy", zorder=3)
	if reg is not None:
		plt.plot(x_vals, y_fit, color="red", label="linear fit")
		plt.annotate(
			f"R²={r2:.4f}\np={reg.pvalue:.4g}",
			xy=(0.05, 0.95),
			xycoords="axes fraction",
			va="top",
		)
	plt.xlabel("mcmc_steps")
	plt.ylabel("next-token energy")
	plt.title("MCMC steps vs next-token energy")
	plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()