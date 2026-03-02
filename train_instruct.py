"""
Fine-tune an EBT checkpoint on the Alpaca instruction dataset.

Example:
  python instruct_finetune.py --ckpt_path ckpt_iter_1000000.pt
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "EBT"))

def load_env_file(env_path: Path) -> None:
	if not env_path.exists():
		return
	for line in env_path.read_text().splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip("\"").strip("'")
		os.environ.setdefault(key, value)


load_env_file(Path("/.env"))
load_env_file(PROJECT_ROOT / ".env")

from model.nlp.ebt import EBT_NLP

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
wandb_run = None


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
		execution_mode=overrides.execution_mode,
		debug_unused_parameters=False,
	)


def format_alpaca_prompt(record):
	instruction = (record.get("instruction") or "").strip()
	input_text = (record.get("input") or "").strip()
	output_text = (record.get("output") or "").strip()
	if not instruction or not output_text:
		return None
	if input_text:
		prompt = (
			"### Instruction:\n"
			f"{instruction}\n\n"
			"### Input:\n"
			f"{input_text}\n\n"
			"### Response:\n"
		)
	else:
		prompt = (
			"### Instruction:\n"
			f"{instruction}\n\n"
			"### Response:\n"
		)
	return prompt + output_text


def iter_token_chunks(dataset, tokenizer, block_size):
	for record in dataset:
		text = format_alpaca_prompt(record)
		if not text:
			continue
		if tokenizer.eos_token:
			text = text + tokenizer.eos_token
		ids = tokenizer.encode(text, add_special_tokens=False)
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


def resolve_device(device):
	if device != "auto":
		return device
	if torch.cuda.is_available():
		return "cuda"
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def parse_bool(value):
	value_lower = str(value).lower()
	if value_lower in ("1", "true", "yes", "y", "on"):
		return True
	if value_lower in ("0", "false", "no", "n", "off"):
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main():
	parser = argparse.ArgumentParser(description="EBT instruction fine-tuning on Alpaca")
	parser.add_argument("--ckpt_path", required=True, help="Path to ckpt_iter_XXXX.pt or ckpt.pt")
	parser.add_argument("--dataset", default="yizhongw/self_instruct", help="Hugging Face dataset name")
	parser.add_argument("--split", default="train", help="Dataset split")
	parser.add_argument("--eval_split", default="train", help="Dataset split for evaluation")
	parser.add_argument("--streaming", action="store_true", default=False)
	parser.add_argument("--take", type=int, default=5000, help="Number of samples to take when streaming")
	# train_openweb.py-style aliases
	parser.add_argument("--batch_size_per_device", type=int, default=None)
	parser.add_argument("--accumulate_grad_batches", type=int, default=1)
	parser.add_argument("--context_length", type=int, default=None)
	parser.add_argument("--num_transformer_blocks", type=int, default=None)
	parser.add_argument("--multiheaded_attention_heads", type=int, default=None)
	parser.add_argument("--embedding_dim", type=int, default=None)
	parser.add_argument("--peak_learning_rate", type=float, default=None)
	parser.add_argument("--max_scheduling_steps", type=int, default=None)
	parser.add_argument("--warm_up_steps", type=int, default=None)
	parser.add_argument("--gpus", default="1")
	parser.add_argument("--distributed_strategy", default="ddp")
	parser.add_argument("--mcmc_replay_buffer_size", type=int, default=192)
	parser.add_argument("--normalize_initial_condition", type=parse_bool, default=None)
	parser.add_argument("--clamp_futures_grad", type=parse_bool, default=None)
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--block_size", type=int, default=None)
	parser.add_argument("--learning_rate", type=float, default=3e-3)
	parser.add_argument("--max_steps", type=int, default=10000)
	parser.add_argument("--eval_interval", type=int, default=1000)
	parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "mps", "cpu"])
	parser.add_argument("--tokenizer", default="gpt2")
	parser.add_argument("--execution_mode", type=str, choices=["pretrain", "finetune", "inference"], default="finetune")
	parser.add_argument("--mcmc_num_steps", type=int, default=None)
	parser.add_argument("--mcmc_step_size", type=float, default=None)
	parser.add_argument("--save_path", default="out_ebd_instruct/ckpt_finetuned.pt")
	parser.add_argument("--log_every", type=int, default=10)
	args = parser.parse_args()

	# Normalize aliases to canonical names (same pattern as train_openweb.py).
	if args.batch_size_per_device is not None:
		args.batch_size = args.batch_size_per_device
	if args.context_length is not None:
		args.block_size = args.context_length
	if args.peak_learning_rate is not None:
		args.learning_rate = args.peak_learning_rate
	gradient_accumulation_steps = max(1, args.accumulate_grad_batches)

	ddp = args.distributed_strategy == "ddp" and int(os.environ.get("RANK", -1)) != -1
	if ddp:
		ddp_rank = int(os.environ["RANK"])
		ddp_local_rank = int(os.environ["LOCAL_RANK"])
		master_process = ddp_rank == 0
		device = f"cuda:{ddp_local_rank}"
		torch.cuda.set_device(ddp_local_rank)
	else:
		master_process = True
		device = resolve_device(args.device)

	checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
	config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
	if not isinstance(config, dict) or not config:
		raise ValueError("Checkpoint is missing a config dict. Provide a checkpoint saved from train.py.")

	hparams = build_hparams(config, args)
	if args.num_transformer_blocks is not None:
		hparams.num_transformer_blocks = args.num_transformer_blocks
	if args.multiheaded_attention_heads is not None:
		hparams.multiheaded_attention_heads = args.multiheaded_attention_heads
	if args.embedding_dim is not None:
		hparams.embedding_dim = args.embedding_dim
	if args.normalize_initial_condition is not None:
		hparams.normalize_initial_condition = args.normalize_initial_condition
	if args.clamp_futures_grad is not None:
		hparams.clamp_futures_grad = args.clamp_futures_grad
	hparams.batch_size_per_device = args.batch_size
	hparams.accumulate_grad_batches = gradient_accumulation_steps
	hparams.context_length = args.block_size if args.block_size is not None else hparams.context_length
	hparams.peak_learning_rate = args.learning_rate
	hparams.max_steps = args.max_steps
	hparams.max_scheduling_steps = args.max_scheduling_steps if args.max_scheduling_steps is not None else args.max_steps
	hparams.warm_up_steps = args.warm_up_steps if args.warm_up_steps is not None else 0
	hparams.gpus = args.gpus
	hparams.distributed_strategy = args.distributed_strategy
	hparams.mcmc_replay_buffer_size = args.mcmc_replay_buffer_size
	hparams.execution_mode = args.execution_mode
	model = EBT_NLP(hparams).to(device)
	state = normalize_state_dict_keys(checkpoint["model"])
	model.load_state_dict(state, strict=False)
	model.train()

	tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces=False)

	ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
	if args.streaming:
		ds = ds.take(args.take)

	try:
		eval_ds = load_dataset(args.dataset, split=args.eval_split, streaming=args.streaming)
	except Exception as exc:
		print(f"Eval split '{args.eval_split}' unavailable, falling back to '{args.split}': {exc}")
		eval_ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
	if args.streaming:
		eval_ds = eval_ds.take(args.take)

	chunks = iter_token_chunks(ds, tokenizer, hparams.context_length)
	batches = batch_iter(chunks, args.batch_size)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

	if master_process and WANDB_API_KEY:
		try:
			import wandb
			wandb_run = wandb.init(
				entity=WANDB_ENTITY,
				project=WANDB_PROJECT,
				config={
					"dataset": args.dataset,
					"split": args.split,
					"eval_split": args.eval_split,
					"batch_size": args.batch_size,
					"batch_size_per_device": args.batch_size,
					"accumulate_grad_batches": gradient_accumulation_steps,
					"block_size": hparams.context_length,
					"context_length": hparams.context_length,
					"learning_rate": args.learning_rate,
					"peak_learning_rate": args.learning_rate,
					"max_steps": args.max_steps,
					"max_scheduling_steps": hparams.max_scheduling_steps,
					"warm_up_steps": hparams.warm_up_steps,
					"eval_interval": args.eval_interval,
					"device": device,
					"gpus": args.gpus,
					"distributed_strategy": args.distributed_strategy,
					"tokenizer": hparams.tokenizer,
					"execution_mode": args.execution_mode,
					"num_transformer_blocks": hparams.num_transformer_blocks,
					"multiheaded_attention_heads": hparams.multiheaded_attention_heads,
					"embedding_dim": hparams.embedding_dim,
					"mcmc_num_steps": hparams.mcmc_num_steps,
					"mcmc_step_size": hparams.mcmc_step_size,
					"mcmc_replay_buffer_size": hparams.mcmc_replay_buffer_size,
					"normalize_initial_condition": hparams.normalize_initial_condition,
					"clamp_futures_grad": hparams.clamp_futures_grad,
					"ckpt_path": args.ckpt_path,
				},
			)
		except Exception as exc:
			print(f"W&B init failed, continuing without logging: {exc}")
			wandb_run = None

	def run_eval(eval_source):
		if args.eval_interval <= 0:
			return None
		if not master_process:
			return None
		model.eval()
		losses = []
		with torch.no_grad():
			eval_chunks = iter_token_chunks(eval_source, tokenizer, hparams.context_length)
			eval_batches = batch_iter(eval_chunks, args.batch_size)
			for idx, eval_batch in enumerate(eval_batches):
				if idx >= 10:
					break
				eval_batch = {"input_ids": eval_batch["input_ids"].to(device)}
				loss = model.forward_loss_wrapper(eval_batch, phase="eval")["loss"]
				losses.append(loss.item())
		model.train()
		if not losses:
			return None
		return sum(losses) / len(losses)

	def save_eval_checkpoint(step_idx, eval_loss):
		if not args.save_path:
			return None
		if not master_process:
			return None
		output_path = Path(args.save_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		ckpt_path = output_path.parent / f"ckpt_eval_step_{step_idx}.pt"
		save_payload = {
			"model": model.state_dict(),
			"config": config,
			"finetune_args": vars(args),
			"step": step_idx,
			"eval_loss": eval_loss,
		}
		torch.save(save_payload, ckpt_path)
		if wandb_run is not None:
			try:
				import wandb
				artifact = wandb.Artifact(f"ckpt_step_{step_idx}", type="model")
				artifact.add_file(str(ckpt_path))
				wandb_run.log_artifact(artifact)
			except Exception as exc:
				print(f"W&B checkpoint logging failed: {exc}")
		return ckpt_path

	step = 0
	for batch in batches:
		batch = {"input_ids": batch["input_ids"].to(device)}
		loss_dict = model.forward_loss_wrapper(batch, phase="train")
		loss = loss_dict["loss"] / gradient_accumulation_steps
		loss.backward()

		if (step + 1) % gradient_accumulation_steps == 0:
			optimizer.step()
			optimizer.zero_grad(set_to_none=True)

		if step % args.log_every == 0:
			loss_value = float(loss.item() * gradient_accumulation_steps)
			if master_process:
				print(f"step {step}: loss {loss_value:.4f}")
			if wandb_run is not None:
				wandb_run.log(
					{
						"step": step,
						"loss": loss_value,
					},
				)

		if args.eval_interval > 0 and step % args.eval_interval == 0:
			eval_loss = run_eval(eval_ds)
			if eval_loss is not None:
				print(f"step {step}: eval loss {eval_loss:.4f}")
				if wandb_run is not None:
					wandb_run.log(
						{
							"step": step,
							"eval_loss": float(eval_loss),
						},
					)
				saved_path = save_eval_checkpoint(step, float(eval_loss))
				if saved_path is not None:
					print(f"Saved eval checkpoint to {saved_path}.")
		step += 1
		if step >= args.max_steps:
			break

	if master_process and args.save_path:
		output_path = Path(args.save_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		save_payload = {
			"model": model.state_dict(),
			"config": config,
			"finetune_args": vars(args),
		}
		torch.save(save_payload, output_path)
		print(f"Saved finetuned checkpoint to {output_path}.")

	if master_process:
		print("Fine-tuning finished.")
	if wandb_run is not None:
		wandb_run.finish()


if __name__ == "__main__":
	main()
