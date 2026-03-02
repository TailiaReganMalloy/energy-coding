"""Interactive CLI for EBT checkpoints.

Example:
  python test_cli.py --ckpt_path Programing/ckpt_iter_910000.pt

  python test_cli.py --ckpt_path out_ebt_instruct/ckpt_iter_480000.pt

  python ebt_cli.py --ckpt_path ./trained_models/ebt_instruct/ckpt_iter_60000.pt --device mps --temperature 0 --top_k 0 --top_p 1.0

  A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. the pans

[[Question]]: Summarize the paragraph in 2-3 sentences. 

Paragraph: "The city council approved a pilot program to replace diesel buses with electric buses over the next 18 months. Officials estimate a 22% reduction in local transport emissions and lower maintenance costs after year two." 

[[Answer]]:

Detect spam comments in the reddit thread. the task is to predict if a comment is spam or not using some machine learning model (or whatever you think makes sense). output 1 - spam, 0 - not spam. Document: I am a bot. 

50 sample of accuract, using higher inference computer, means less pretraining time, if we cant we should do more, simple mbpp. task 

"""

import argparse
import sys
import time
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


def generate(model, tokenizer, hparams, prompt, device, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stream=False, stop_on_paragraph_break=True):
	model.eval()
	requires_grad = hparams.model_name == "ebt" and not hparams.infer_ebt_advanced
	context = torch.enable_grad() if requires_grad else torch.inference_mode()
	show_progress = not stream
	auto_stop_on_paragraph_break = stop_on_paragraph_break
	with context:
		input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
		max_context = hparams.context_length
		if input_ids.shape[1] >= max_context:
			input_ids = input_ids[:, -max_context:]
			prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
			if stream:
				print("[prompt truncated to fit context]\n", end="")
		generated_ids = []
		generated_text = ""
		if show_progress:
			bar_width = 30
			start_time = time.time()
			print("Generating:", end="", flush=True)
		for _ in range(max_new_tokens):
			if show_progress:
				progress = len(generated_ids)
				filled = int(bar_width * progress / max_new_tokens) if max_new_tokens > 0 else bar_width
				bar = "=" * filled + "-" * (bar_width - filled)
				percent = int(100 * progress / max_new_tokens) if max_new_tokens > 0 else 100
				elapsed = time.time() - start_time
				rate = progress / elapsed if elapsed > 0 else 0.0
				remaining = max_new_tokens - progress
				eta = remaining / rate if rate > 0 else 0.0
				print(
					f"\rGenerating: [{bar}] {percent:3d}% | {elapsed:6.1f}s elapsed | {eta:6.1f}s ETA",
					end="",
					flush=True,
				)
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
			generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
			if stream:
				print(tokenizer.decode([token_id], skip_special_tokens=True), end="", flush=True)
				if len(generated_ids) >= 2:
					last_text = tokenizer.decode(generated_ids[-2:], skip_special_tokens=True)
					if "\n\n" in last_text:
						break
			if auto_stop_on_paragraph_break and "\n\n" in generated_text and len(generated_text.strip()) >= 32:
				break
			if token_id == tokenizer.eos_token_id:
				break
		if stream:
			print("")
		if show_progress:
			elapsed = time.time() - start_time
			print(
				"\rGenerating: [" + "=" * bar_width + f"] 100% | {elapsed:6.1f}s elapsed |   0.0s ETA",
				flush=True,
			)
			return tokenizer.decode(generated_ids, skip_special_tokens=True)
		if show_progress:
			elapsed = time.time() - start_time
			print(
				"\rGenerating: [" + "=" * bar_width + f"] 100% | {elapsed:6.1f}s elapsed |   0.0s ETA",
				flush=True,
			)
		return tokenizer.decode(generated_ids, skip_special_tokens=True)


def read_prompt_with_continuations(first_line):
	lines = [first_line]
	joined = first_line
	requires_answer_tag = "[[Question]]" in first_line and "[[Answer]]" not in first_line
	while requires_answer_tag and "[[Answer]]" not in joined:
		next_line = input().rstrip("\n")
		if next_line.lower() in {"/cancel", "cancel"}:
			return ""
		lines.append(next_line)
		joined = "\n".join(lines)
	return joined


def normalize_prompt_for_answer_generation(prompt):
	for marker in ("[[Answer]]:", "### Response:"):
		if marker in prompt:
			cutoff = prompt.rfind(marker) + len(marker)
			normalized = prompt[:cutoff].rstrip()
			return normalized + "\n"
	return prompt


def main():
	parser = argparse.ArgumentParser(description="EBT interactive CLI")
	parser.add_argument("--ckpt_path", required=True, help="Path to ckpt_iter_XXXX.pt or ckpt.pt")
	parser.add_argument("--device", default="mps", choices=["auto", "cuda", "mps", "cpu"])
	parser.add_argument("--max_gen_len", type=int, default=128)
	parser.add_argument("--temperature", type=float, default=0.8)
	parser.add_argument("--top_p", type=float, default=1.0)
	parser.add_argument("--top_k", type=int, default=5)
	parser.add_argument("--repetition_penalty", type=float, default=1.1)
	parser.add_argument("--stream", action="store_true", help="Print tokens as they are generated")
	parser.add_argument("--no_stop_on_paragraph_break", action="store_true", help="Disable stopping generation at first blank line (\\n\\n)")
	parser.add_argument("--infer_ebt_advanced", action="store_true")
	parser.add_argument("--mcmc_num_steps", type=int, default=5, help="Override mcmc_num_steps for faster inference (must be >= 1)")
	parser.add_argument("--mcmc_step_size", type=float, default=60, help="Override mcmc_step_size for inference")
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

	if args.top_k == 1 and args.temperature > 0:
		print("Warning: top_k=1 with temperature>0 is effectively near-greedy decoding and can degrade output quality.")
		print("Try --temperature 0 (fully greedy) or increase --top_k (e.g. 20-50).")

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
	print("For multi-line question prompts, paste lines and include [[Answer]]:. Type /cancel to abort input.")
	while True:
		prompt = input("\n> ").rstrip("\n")
		if prompt.lower() in {"/exit", "exit", "quit"}:
			break
		if not prompt:
			continue
		prompt = read_prompt_with_continuations(prompt)
		if not prompt:
			continue
		prompt = normalize_prompt_for_answer_generation(prompt)
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
			stop_on_paragraph_break=not args.no_stop_on_paragraph_break,
		)
		if not args.stream:
			print(response)


if __name__ == "__main__":
	main()
