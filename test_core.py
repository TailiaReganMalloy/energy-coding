"""Load an EBT checkpoint and evaluate its CORE metric.

Example:
	python test_core.py --ckpt_path out_ebt_instruct/ckpt_iter_480000.pt
"""

import argparse
import io
import json
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch


def load_env_file(env_path: Path) -> None:
	if not env_path.exists():
		return
	for line in env_path.read_text().splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip().strip('"').strip("'")
		os.environ.setdefault(key, value)


def configure_hf_cache(cache_root: Path | None) -> None:
	if cache_root is None:
		return
	cache_root.mkdir(parents=True, exist_ok=True)
	os.environ["HF_HOME"] = str(cache_root)
	os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
	os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
	os.environ["XDG_CACHE_HOME"] = str(cache_root)


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "EBT"))
sys.path.insert(0, str(PROJECT_ROOT / "nanochat"))

load_env_file(Path("/.env"))
load_env_file(PROJECT_ROOT / ".env")
hf_cache_env = os.environ.get("HF_CACHE_DIR")
hf_cache_dir = Path(hf_cache_env).expanduser() if hf_cache_env else (PROJECT_ROOT / ".hf_cache")
configure_hf_cache(hf_cache_dir)

warnings.filterwarnings(
	"ignore",
	message=r".*torch\.library\.impl_abstract.*",
	category=FutureWarning,
)

from model.nlp.ebt import EBT_NLP
from nanochat.tokenizer import HuggingFaceTokenizer
from scripts.base_eval import evaluate_core


def normalize_state_dict_keys(state_dict):
	if not any(key.startswith("_orig_mod.") for key in state_dict):
		return state_dict
	return {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}


def build_hparams(
	config,
	mcmc_num_steps_override=None,
	temperature_override=None,
	top_p_override=None,
	top_k_override=None,
):
	mcmc_num_steps = mcmc_num_steps_override if mcmc_num_steps_override is not None else config.get("mcmc_num_steps", 1)
	if mcmc_num_steps < 1:
		raise ValueError("mcmc_num_steps must be >= 1")
	temperature = temperature_override if temperature_override is not None else config.get("infer_temp", 0.8)
	top_p = top_p_override if top_p_override is not None else config.get("infer_topp", 1.0)
	top_k = top_k_override if top_k_override is not None else config.get("infer_topk", 5)
	if top_p <= 0.0 or top_p > 1.0:
		raise ValueError("top_p must be in (0, 1]")
	if top_k < 0:
		raise ValueError("top_k must be >= 0")

	return SimpleNamespace(
		modality="NLP",
		model_name=config.get("model_name", "ebt"),
		tokenizer=config.get("tokenizer", "gpt2"),
		context_length=config.get("block_size", 256),
		num_transformer_blocks=config.get("n_layer", 6),
		multiheaded_attention_heads=config.get("n_head", 6),
		embedding_dim=config.get("n_embd", 384),
		ffn_dim_multiplier=config.get("ffn_dim_multiplier", None),
		batch_size_per_device=1,
		ebt_type=config.get("ebt_type", "default"),
		ebt_norm=config.get("ebt_norm", "rms"),
		ebt_act_func=config.get("ebt_act_func", "silu"),
		dyt_alpha_init=config.get("dyt_alpha_init", 0.5),
		weight_initialization_method=config.get("weight_initialization_method", "xavier"),
		weight_initialization_gain=config.get("weight_initialization_gain", 1.0),
		mcmc_num_steps=mcmc_num_steps,
		mcmc_step_size=config.get("mcmc_step_size", 60.0),
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
		infer_temp=temperature,
		infer_topp=top_p,
		infer_topk=top_k,
	)


class EBTCoreModelWrapper:
	def __init__(self, model, max_seq_len=None):
		self.model = model
		self.max_seq_len = max_seq_len

	def __call__(self, input_ids, targets=None, loss_reduction="mean"):
		logits_per_step, _ = self.model.forward(
			input_ids,
			start_pos=0,
			learning=False,
			return_raw_logits=True,
			no_randomness=True,
			return_last_only=True,
		)
		logits = logits_per_step[-1]
		if targets is None:
			return logits
		loss = torch.nn.functional.cross_entropy(
			logits.view(-1, logits.size(-1)),
			targets.view(-1),
			ignore_index=-1,
			reduction=loss_reduction,
		)
		return loss

	def get_device(self):
		return next(self.model.parameters()).device


def resolve_device(device):
	if device != "auto":
		return device
	if torch.cuda.is_available():
		return "cuda"
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return "mps"
	return "cpu"


def load_ebt_for_core(
	ckpt_path,
	device,
	mcmc_num_steps_override=None,
	temperature_override=None,
	top_p_override=None,
	top_k_override=None,
):
	checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
	config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
	if not isinstance(config, dict) or not config:
		raise ValueError("Checkpoint is missing a config dict. Provide a checkpoint saved from train.py.")

	hparams = build_hparams(
		config,
		mcmc_num_steps_override=mcmc_num_steps_override,
		temperature_override=temperature_override,
		top_p_override=top_p_override,
		top_k_override=top_k_override,
	)
	raw_model = EBT_NLP(hparams).to(device)
	state = normalize_state_dict_keys(checkpoint["model"])
	raw_model.load_state_dict(state, strict=False)
	raw_model.eval()

	model = EBTCoreModelWrapper(raw_model, max_seq_len=hparams.context_length)
	tokenizer = HuggingFaceTokenizer.from_pretrained(hparams.tokenizer)
	step = checkpoint.get("iter_num", None)
	return model, tokenizer, step, config


def evaluate_mcmc_sweep(model, tokenizer, device, max_per_task, mcmc_values, model_info):
	try:
		import pandas as pd
	except ModuleNotFoundError as exc:
		raise ModuleNotFoundError(
			"pandas is required for --test_core sweep output. Install with: pip install pandas"
		) from exc

	try:
		from tqdm import tqdm
	except ModuleNotFoundError:
		def tqdm(iterable, **kwargs):
			return iterable

	steps_to_eval = list(mcmc_values)
	if not steps_to_eval:
		raise ValueError("mcmc_values must contain at least one integer")
	if any(step < 1 for step in steps_to_eval):
		raise ValueError("All mcmc_values must be >= 1")
	rows = []

	progress_bar = tqdm(total=0, desc="CORE sweep", unit="eval", dynamic_ncols=True)
	for mcmc_steps in steps_to_eval:
		progress_bar.set_postfix(mcmc_num_steps=mcmc_steps)
		model.model.hparams.mcmc_num_steps = mcmc_steps
		capture_stream = io.StringIO()
		with redirect_stdout(capture_stream), redirect_stderr(capture_stream):
			core = evaluate_core(model, tokenizer, device, max_per_task=max_per_task)

		captured_lines = capture_stream.getvalue().splitlines()
		eval_line_count = sum(1 for line in captured_lines if line.lstrip().startswith("Evaluating:"))
		if eval_line_count > 0 and progress_bar.total == 0:
			progress_bar.total = eval_line_count * len(steps_to_eval)
			progress_bar.refresh()
		progress_bar.update(eval_line_count)
		rows.append(
			{
				"ckpt_path": str(model_info["ckpt_path"]),
				"checkpoint_iter": model_info["checkpoint_iter"],
				"device": device,
				"max_per_task": max_per_task,
				"model_name": model_info["model_name"],
				"tokenizer": model_info["tokenizer"],
				"context_length": model_info["context_length"],
				"n_layer": model_info["n_layer"],
				"n_head": model_info["n_head"],
				"n_embd": model_info["n_embd"],
				"mcmc_num_steps": mcmc_steps,
				"core_metric": core["core_metric"],
				"centered_results_json": json.dumps(core["centered_results"], sort_keys=True),
			}
		)

	progress_bar.close()

	return pd.DataFrame(rows)


def main():
	parser = argparse.ArgumentParser(description="Evaluate CORE metric sweep for an EBT checkpoint")
	parser.add_argument("--ckpt_path", required=True, help="Path to ckpt_iter_XXXX.pt or ckpt.pt")
	parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
	parser.add_argument("--max-per-task", type=int, default=100, help="Max examples per CORE task (default: 100)")
	parser.add_argument(
		"--mcmc-values",
		type=int,
		nargs="+",
		default=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
		help="Explicit MCMC values to sweep, e.g. --mcmc-values 1 10 20",
	)
	parser.add_argument("--temperature", type=float, default=0.8, help="Inference temperature override")
	parser.add_argument("--top_p", type=float, default=0.8, help="Top-p (nucleus) sampling override, in (0, 1]")
	parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling override (>= 0)")
	parser.add_argument("--output-csv", default="core_mcmc_sweep.csv", help="Where to save sweep results as CSV")
	args = parser.parse_args()
	mcmc_values = list(args.mcmc_values)

	device = resolve_device(args.device)
	model, tokenizer, step, config = load_ebt_for_core(
		args.ckpt_path,
		device,
		mcmc_num_steps_override=mcmc_values[0],
		temperature_override=args.temperature,
		top_p_override=args.top_p,
		top_k_override=args.top_k,
	)

	label = f"EBT checkpoint (iter {step})" if step is not None else "EBT checkpoint"
	print(f"Evaluating {label} on device={device}")
	print(
		f"Running CORE sweep for mcmc_num_steps values "
		f"{mcmc_values} "
		f"with max_per_task={args.max_per_task}"
	)

	model_info = {
		"ckpt_path": args.ckpt_path,
		"checkpoint_iter": step,
		"model_name": config.get("model_name", "ebt"),
		"tokenizer": config.get("tokenizer", "gpt2"),
		"context_length": config.get("block_size", 256),
		"n_layer": config.get("n_layer", 6),
		"n_head": config.get("n_head", 6),
		"n_embd": config.get("n_embd", 384),
	}

	df = evaluate_mcmc_sweep(
		model=model,
		tokenizer=tokenizer,
		device=device,
		max_per_task=args.max_per_task,
		mcmc_values=mcmc_values,
		model_info=model_info,
	)

	output_csv = Path(args.output_csv)
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_csv, index=False)

	print("\nSweep results (mcmc_num_steps vs core_metric):")
	print(df[["mcmc_num_steps", "core_metric"]].to_string(index=False))
	best_idx = df["core_metric"].idxmax()
	best_row = df.loc[best_idx]
	print(
		f"\nBest CORE metric: {best_row['core_metric']:.6f} "
		f"at mcmc_num_steps={int(best_row['mcmc_num_steps'])}"
	)
	print(f"Saved full sweep dataframe to: {output_csv}")


if __name__ == "__main__":
	main()
