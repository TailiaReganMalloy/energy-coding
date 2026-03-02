"""Interactive CLI for EBT checkpoints.

Example:
  python test_cli.py --ckpt_path Programing/ckpt_iter_910000.pt

  python test_cli.py --ckpt_path out_ebt_instruct/ckpt_iter_480000.pt

  A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. the pans
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "EBT"))

from model.nlp.ebt import EBT_NLP


def normalize_state_dict_keys(state_dict):
	if not any(key.startswith("_orig_mod.") for key in state_dict):
		return state_dict
	return {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}


def sample_top_p(probs, p):
	probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
	probs_sum = torch.cumsum(probs_sort, dim=-1)
	mask = probs_sum - probs_sort > p
	probs_sort[mask] = 0.0
	probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
	next_token = torch.multinomial(probs_sort, num_samples=1)
	next_token = torch.gather(probs_idx, -1, next_token)
	return next_token


def build_hparams(config, args):
	mcmc_num_steps = args.mcmc_num_steps if args.mcmc_num_steps is not None else config.get("mcmc_num_steps", 1)
	if mcmc_num_steps < 1:
		print("mcmc_num_steps must be >= 1 for inference; overriding to 1.")
		mcmc_num_steps = 1
	return SimpleNamespace(
		modality="NLP",
		model_name=config.get("model_name", "ebt"),
		tokenizer=config.get("tokenizer", "gpt2"),
		context_length=config.get("block_size", config.get("context_length", 256)),
		num_transformer_blocks=config.get("n_layer", config.get("num_transformer_blocks", 6)),
		multiheaded_attention_heads=config.get("n_head", config.get("multiheaded_attention_heads", 6)),
		embedding_dim=config.get("n_embd", config.get("embedding_dim", 384)),
		ffn_dim_multiplier=config.get("ffn_dim_multiplier", None),
		batch_size_per_device=1,
		ebt_type=config.get("ebt_type", "default"),
		ebt_norm=config.get("ebt_norm", "rms"),
		ebt_act_func=config.get("ebt_act_func", "silu"),
		dyt_alpha_init=config.get("dyt_alpha_init", 0.5),
		weight_initialization_method=config.get("weight_initialization_method", "xavier"),
		weight_initialization_gain=config.get("weight_initialization_gain", 1.0),
		mcmc_num_steps=mcmc_num_steps,
		mcmc_step_size=args.mcmc_step_size if args.mcmc_step_size is not None else config.get("mcmc_step_size", 60.0),
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
		execution_mode="inference",
		debug_unused_parameters=False,
		infer_ebt_advanced=args.infer_ebt_advanced,
		infer_max_gen_len=args.max_gen_len,
		infer_temp=args.temperature,
		infer_topp=args.top_p,
		infer_logprobs=False,
		infer_echo=False,
		infer_ebt_override_alpha=args.infer_ebt_override_alpha,
		infer_generated_samples=args.infer_generated_samples,
		infer_debug_sample_distances=False,
		infer_langevin_dynamics_noise=args.infer_langevin_dynamics_noise,
		infer_langevin_first_step=args.infer_langevin_first_step,
		infer_energy_sampling_technique=args.infer_energy_sampling_technique,
		infer_alpha_final_landscape=False,
		infer_plot_energy_landscape=False,
	)


def get_logits(hparams, model, input_tokens):
	if hparams.model_name == "ebt":
		if hparams.infer_ebt_advanced:
			outputs = model.ebt_advanced_inference(input_tokens, start_pos=0, learning=False)
			logits = outputs[0]
		else:
			outputs = model.forward(input_tokens, start_pos=0, learning=False, return_raw_logits=True)
			if not outputs[0]:
				raise RuntimeError("Model returned no logits. Ensure mcmc_num_steps >= 1.")
			logits = outputs[0][-1]
	else:
		logits = model.forward(input_tokens, start_pos=0, learning=False, return_raw_logits=True)
	return logits



def apply_repetition_penalty(logits, generated_ids, penalty):
	if penalty <= 1.0 or not generated_ids:
		return logits
	logits = logits.clone()
	unique_ids = set(generated_ids)
	for token_id in unique_ids:
		logits[..., token_id] = logits[..., token_id] / penalty
	return logits


def generate(model, tokenizer, hparams, prompt, device, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stream=False):
	model.eval()
	requires_grad = hparams.model_name == "ebt" and not hparams.infer_ebt_advanced
	context = torch.enable_grad() if requires_grad else torch.inference_mode()
	with context:
		input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
		max_context = hparams.context_length
		if input_ids.shape[1] >= max_context:
			input_ids = input_ids[:, -max_context:]
			prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
			if stream:
				print("[prompt truncated to fit context]\n", end="")
		generated_ids = []
		for _ in range(max_new_tokens):
			logits = get_logits(hparams, model, input_ids)
			next_logits = logits[:, -1, :]
			next_logits = apply_repetition_penalty(next_logits, generated_ids, repetition_penalty)
			if temperature > 0:
				next_logits = next_logits / temperature
				probs = torch.softmax(next_logits, dim=-1)
				if top_k > 0:
					values, indices = torch.topk(probs, top_k)
					probs = torch.zeros_like(probs).scatter_(1, indices, values)
					probs = probs / probs.sum(dim=-1, keepdim=True)
				if top_p < 1.0:
					next_token = sample_top_p(probs, top_p)
				else:
					next_token = torch.multinomial(probs, num_samples=1)
			else:
				next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
			input_ids = torch.cat([input_ids, next_token], dim=-1)
			token_id = next_token.item()
			generated_ids.append(token_id)
			if stream:
				print(tokenizer.decode([token_id], skip_special_tokens=True), end="", flush=True)
				if len(generated_ids) >= 2:
					last_text = tokenizer.decode(generated_ids[-2:], skip_special_tokens=True)
					if "\n\n" in last_text:
						break
			if token_id == tokenizer.eos_token_id:
				break
		if stream:
			print("")
			return tokenizer.decode(generated_ids, skip_special_tokens=True)
		return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
	parser = argparse.ArgumentParser(description="EBT interactive CLI")
	parser.add_argument("--ckpt_path", required=True, help="Path to ckpt_iter_XXXX.pt or ckpt.pt")
	parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
	parser.add_argument("--max_gen_len", type=int, default=128)
	parser.add_argument("--temperature", type=float, default=0.6)
	parser.add_argument("--top_p", type=float, default=0.9)
	parser.add_argument("--top_k", type=int, default=50)
	parser.add_argument("--repetition_penalty", type=float, default=1.1)
	parser.add_argument("--stream", action="store_true", help="Print tokens as they are generated")
	parser.add_argument("--infer_ebt_advanced", action="store_true")
	parser.add_argument("--mcmc_num_steps", type=int, default=None, help="Override mcmc_num_steps for faster inference (must be >= 1)")
	parser.add_argument("--mcmc_step_size", type=float, default=None, help="Override mcmc_step_size for inference")
	parser.add_argument("--infer_ebt_override_alpha", type=float, default=0.0)
	parser.add_argument("--infer_generated_samples", type=int, default=1)
	parser.add_argument("--infer_langevin_dynamics_noise", type=float, default=0.0)
	parser.add_argument("--infer_langevin_first_step", action="store_true")
	parser.add_argument(
		"--infer_energy_sampling_technique",
		choices=["min", "max_gap", "max"],
		default="min",
	)
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

	print("Interactive EBT CLI. Type a prompt and press enter. Type /exit to quit.")
	while True:
		prompt = input("\n> ").strip()
		if prompt.lower() in {"/exit", "exit", "quit"}:
			break
		if not prompt:
			continue
		response = generate(
			model,
			tokenizer,
			hparams,
			prompt,
			device,
			args.max_gen_len,
			args.temperature,
			args.top_p,
			args.top_k,
			args.repetition_penalty,
			stream=args.stream,
		)
		if not args.stream:
			print(response)


if __name__ == "__main__":
	main()
