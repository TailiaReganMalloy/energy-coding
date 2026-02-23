"""EBT fine-tuning on The Stack (python).

Example:
  python finetuning/train.py --ckpt_path out_ebt_openwebtext/ckpt_iter_135000.pt
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
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
	parser.add_argument("--take", type=int, default=1000, help="Number of samples to take when streaming")
	parser.add_argument("--batch_size", type=int, default=2)
	parser.add_argument("--block_size", type=int, default=None)
	parser.add_argument("--learning_rate", type=float, default=6e-5)
	parser.add_argument("--max_steps", type=int, default=100)
	parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
	parser.add_argument("--tokenizer", default="gpt2")
	parser.add_argument("--mcmc_num_steps", type=int, default=None)
	parser.add_argument("--mcmc_step_size", type=float, default=None)
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
	model.train()

	tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces=False)

	ds = load_dataset(args.dataset, data_dir=args.data_dir, split=args.split, streaming=args.streaming)
	if args.streaming:
		ds = ds.take(args.take)

	chunks = iter_token_chunks(ds, tokenizer, hparams.context_length)
	batches = batch_iter(chunks, args.batch_size)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

	step = 0
	for batch in batches:
		batch = {"input_ids": batch["input_ids"].to(device)}
		loss_dict = model.forward_loss_wrapper(batch, phase="train")
		loss = loss_dict["loss"]
		loss.backward()
		optimizer.step()
		optimizer.zero_grad(set_to_none=True)

		if step % 10 == 0:
			print(f"step {step}: loss {loss.item():.4f}")
		step += 1
		if step >= args.max_steps:
			break

	print("Fine-tuning finished.")


if __name__ == "__main__":
	main()