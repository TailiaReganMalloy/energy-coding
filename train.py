"""
Train an Energy Based Transformer (EBT) on OpenWebText using the same
training loop structure as nanoGPT.

Example (single GPU):
	python train.py --dataset=openwebtext --data_dir=nanoGPT/data/openwebtext

Example (DDP, 4 GPUs):
	torchrun --standalone --nproc_per_node=4 train.py --dataset=openwebtext --data_dir=nanoGPT/data/openwebtext
"""

import argparse
import json
import os
import time
import math
import pickle
import runpy
import warnings
import re
from contextlib import nullcontext
from types import SimpleNamespace

import sys
from pathlib import Path

# Limit CPU thread usage unless explicitly set to avoid oversubscription warnings.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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

load_env_file(Path("/.env"))
load_env_file(PROJECT_ROOT / ".env")
hf_cache_env = os.environ.get("HF_CACHE_DIR")
hf_cache_dir = Path(hf_cache_env).expanduser() if hf_cache_env else (PROJECT_ROOT / ".hf_cache")
configure_hf_cache(hf_cache_dir)

# Silence known third-party FutureWarnings that are already handled upstream.
warnings.filterwarnings(
	"ignore",
	message=r".*torch\.library\.impl_abstract.*",
	category=FutureWarning,
)

from model.nlp.ebt import EBT_NLP

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
wandb_run = None

# -----------------------------------------------------------------------------
# Default config values designed to mirror nanoGPT training loop behavior.
defaults = {
	# I/O
	"out_dir": "out_ebt",
	"eval_interval": 2000,
	"log_interval": 1,
	"eval_iters": 200,
	"eval_only": False,
	"always_save_checkpoint": True,
	"init_from": "scratch",
	"resume_latest": False,
	# data
	"dataset": "openwebtext",
	"data_dir": os.path.join("nanoGPT", "data", "openwebtext"),
	"gradient_accumulation_steps": 5 * 8,
	"batch_size": 4,
	"block_size": 256,
	# train_model.py-compatible aliases
	"accumulate_grad_batches": 5 * 8,
	"batch_size_per_device": 4,
	"context_length": 256,
	# model
	"n_layer": 6,
	"n_head": 6,
	"n_embd": 384,
	"dropout": 0.0,
	# train_model.py-compatible aliases
	"num_transformer_blocks": 6,
	"multiheaded_attention_heads": 6,
	"embedding_dim": 384,
	# optimizer
	"learning_rate": 1e-3,
	"max_iters": 600000,
	"weight_decay": 1e-1,
	"beta1": 0.9,
	"beta2": 0.95,
	"grad_clip": 1.0,
	# train_model.py-compatible aliases
	"peak_learning_rate": 1e-3,
	"max_steps": 600000,
	"gradient_clip_val": 1.0,
	# learning rate decay settings
	"decay_lr": True,
	"warmup_iters": 2000,
	"lr_decay_iters": 600000,
	"min_lr": 6e-5,
	# train_model.py-compatible aliases
	"warm_up_steps": 2000,
	"max_scheduling_steps": 600000,
	# DDP settings
	"backend": "nccl",
	# train_model.py-compatible hardware params
	"gpus": "8",
	"distributed_strategy": "ddp",
	"execution_mode": "pretrain",
	# system
	"device": "auto",
	"dtype": "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
	"force_float32": True,
	"compile": False,
	# Vast.ai serverless (optional)
	"train_vast_serverless": False,
	"vast_endpoint_name": "yhqdfymr",
	"vast_model_name": "Qwen/Qwen3-8B",
	"vast_max_tokens": 256,
	"vast_temperature": 0.7,
	"vast_top_k": 20,
	"vast_top_p": 0.4,
	"vast_stream": False,
	"vast_prompt_max_chars": 1000,
	# memory check
	"check_memory_fit": True,
	"memory_safety_factor": 1.2,
	"auto_reduce_on_oom": True,
	"oom_retries": 3,
	# EBT-specific defaults
	"tokenizer": "gpt2",
	"model_name": "ebt",
	"ebt_type": "default",
	"ebt_norm": "rms",
	"ebt_act_func": "silu",
	"ffn_dim_multiplier": None,
	"dyt_alpha_init": 0.5,
	"weight_initialization_method": "xavier",
	"weight_initialization_gain": 1.0,
	"mcmc_num_steps": 1,
	"mcmc_step_size": 60.0,
	"mcmc_step_size_learnable": False,
	"langevin_dynamics_noise": 0.0,
	"langevin_dynamics_noise_learnable": False,
	"randomize_mcmc_step_size_scale": 1.0,
	"randomize_mcmc_num_steps": 0,
	"randomize_mcmc_num_steps_final_landscape": False,
	"randomize_mcmc_num_steps_min": 0,
	"denoising_initial_condition": "random_noise",
	"gaussian_random_noise_scaling": 1.0,
	"normalize_initial_condition": False,
	"normalize_initial_condition_only_first_step": False,
	"vocab_to_embed_uses_prob_dist": False,
	"num_modality_processing_mlp_layers": 1,
	"learnable_process_memory": False,
	"process_memory_type": None,
	"process_memory_linear_layer": False,
	"clamp_futures_grad": False,
	"clamp_futures_grad_max_change": 9.0,
	"absolute_clamp": 0.0,
	"clamp_max_after_warm_up": 0.0,
	"sharpen_predicted_distribution": 0.0,
	"mcmc_replay_buffer_size": 192,
	"truncate_mcmc": False,
	"no_mcmc_detach": False,
	"contrastive_loss": False,
	"contrastive_loss_coeff": 0.0005,
	"discrete_contrastive_loss_true_logit_val": 0.0,
	"soften_target_prob_dist": 0.0,
	"reconstruction_coeff": 1.0,
}

def _parse_bool(value: str) -> bool:
	value_lower = value.lower()
	if value_lower in ("1", "true", "yes", "y", "on"):
		return True
	if value_lower in ("0", "false", "no", "n", "off"):
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_optional_str(value: str) -> str | None:
	return None if value.lower() == "none" else value


def _arg_type(default_value):
	if isinstance(default_value, bool):
		return _parse_bool
	if default_value is None:
		return _parse_optional_str
	return type(default_value)


def _parse_args(default_values: dict) -> dict:
	parser = argparse.ArgumentParser()
	for key, default_value in default_values.items():
		parser.add_argument(f"--{key}", type=_arg_type(default_value), default=default_value)
	args = parser.parse_args()
	config_values = dict(default_values)
	config_values.update(vars(args))
	for key, value in config_values.items():
		if value != default_values[key]:
			print(f"Overriding: {key} = {value}")
	return config_values


def _parse_gpus(value):
	if isinstance(value, int):
		return value
	if isinstance(value, str):
		value = value.strip()
		if value.startswith("[") and value.endswith("]"):
			return json.loads(value)
		if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
			return int(value)
	return value


config = _parse_args(defaults)
globals().update(config)

# ----------------------------------------------------------------------------
# Normalize train_model.py-style aliases after overrides so both names work.
learning_rate = peak_learning_rate
batch_size = batch_size_per_device
gradient_accumulation_steps = accumulate_grad_batches
block_size = context_length
n_layer = num_transformer_blocks
n_head = multiheaded_attention_heads
n_embd = embedding_dim
max_iters = max_steps
lr_decay_iters = max_scheduling_steps
warmup_iters = warm_up_steps
grad_clip = gradient_clip_val
config.update({k: globals()[k] for k in config.keys()})
# -----------------------------------------------------------------------------

gpus = _parse_gpus(gpus)
if gpus == -1:
	num_gpus_requested = torch.cuda.device_count()
elif isinstance(gpus, list):
	num_gpus_requested = len(gpus)
elif isinstance(gpus, int):
	num_gpus_requested = gpus
else:
	raise ValueError(f"Unsupported gpus value: {gpus}")

# Ensure head_dim is even for rotary embeddings.
if n_embd % n_head != 0:
	raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head}).")
head_dim = n_embd // n_head
if head_dim % 2 != 0:
	new_n_embd = n_head * (head_dim + 1)
	print(
		f"Adjusting n_embd from {n_embd} to {new_n_embd} so head_dim is even for rotary embeddings."
	)
	n_embd = new_n_embd
	config["n_embd"] = n_embd

def find_latest_checkpoint(output_dir: str) -> str | None:
	output_path = Path(output_dir)
	if not output_path.exists():
		return None
	ckpt_iter_paths = []
	for path in output_path.glob("ckpt_iter_*.pt"):
		stem = path.stem
		try:
			iter_str = stem.split("ckpt_iter_", 1)[1]
			iter_num = int(iter_str)
			ckpt_iter_paths.append((iter_num, path))
		except (IndexError, ValueError):
			continue
	if ckpt_iter_paths:
		ckpt_iter_paths.sort(key=lambda item: item[0])
		return str(ckpt_iter_paths[-1][1])
	ckpt_path = output_path / "ckpt.pt"
	return str(ckpt_path) if ckpt_path.exists() else None

data_dir = os.path.abspath(os.path.join(PROJECT_ROOT, data_dir)) if not os.path.isabs(data_dir) else data_dir

# DDP setup
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
	init_process_group(backend=backend)
	ddp_rank = int(os.environ["RANK"])
	ddp_local_rank = int(os.environ["LOCAL_RANK"])
	ddp_world_size = int(os.environ["WORLD_SIZE"])
	device = f"cuda:{ddp_local_rank}"
	torch.cuda.set_device(device)
	master_process = ddp_rank == 0
	seed_offset = ddp_rank
	if gradient_accumulation_steps % ddp_world_size == 0:
		gradient_accumulation_steps //= ddp_world_size
	elif master_process:
		print(
			"Warning: gradient_accumulation_steps is not divisible by world size; "
			"treating it as per-rank accumulation."
		)
else:
	master_process = True
	seed_offset = 0
	ddp_world_size = 1

if ddp:
	if num_gpus_requested != ddp_world_size:
		raise ValueError(
			"Requested gpus does not match torchrun world size. "
			f"gpus={gpus} (num_gpus={num_gpus_requested}) vs WORLD_SIZE={ddp_world_size}. "
			"Update --gpus or --nproc_per_node to match."
		)
else:
	if num_gpus_requested != 1:
		print(
			"Warning: --gpus requests multiple devices but DDP is not active; "
			"using a single process. Use torchrun to enable multi-GPU training."
		)
	if device == "auto" and isinstance(gpus, list) and gpus:
		device = f"cuda:{gpus[0]}"

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
	os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)

if not ddp:
	if device == "auto":
		if torch.cuda.is_available():
			device = "cuda"
		elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
			device = "mps"
		else:
			device = "cpu"
	elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
		raise RuntimeError("MPS requested but not available.")
	elif device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but not available.")
else:
	if "cuda" not in device:
		raise RuntimeError("DDP is only supported on CUDA devices.")

if "cuda" in device and torch.cuda.is_available():
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

if "cuda" in device:
	device_type = "cuda"
elif device == "mps":
	device_type = "mps"
else:
	device_type = "cpu"

if force_float32:
	dtype = "float32"

ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
	nullcontext()
	if device_type in ("cpu", "mps") or dtype == "float32"
	else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

if train_vast_serverless:
	if not os.environ.get("VAST_API_KEY"):
		raise RuntimeError("VAST_API_KEY must be set when train_vast_serverless=True")
	if master_process:
		print("Vast serverless mode enabled for eval-time requests.")
		print(f"VAST endpoint: {vast_endpoint_name}")
		print(f"VAST model: {vast_model_name}")

if master_process and WANDB_API_KEY:
	try:
		import wandb
		wandb_run = wandb.init(
			entity=WANDB_ENTITY,
			project=WANDB_PROJECT,
			config={
				"dataset": dataset,
				"data_dir": data_dir,
				"out_dir": out_dir,
				"batch_size_per_device": batch_size,
				"accumulate_grad_batches": gradient_accumulation_steps,
				"context_length": block_size,
				"num_transformer_blocks": n_layer,
				"multiheaded_attention_heads": n_head,
				"embedding_dim": n_embd,
				"peak_learning_rate": learning_rate,
				"max_steps": max_iters,
				"max_scheduling_steps": lr_decay_iters,
				"warm_up_steps": warmup_iters,
				"gradient_clip_val": grad_clip,
				"weight_decay": weight_decay,
				"beta1": beta1,
				"beta2": beta2,
				"gpus": gpus,
				"distributed_strategy": distributed_strategy,
				"device": device,
				"dtype": dtype,
				"mcmc_num_steps": mcmc_num_steps,
				"mcmc_step_size": mcmc_step_size,
				"mcmc_replay_buffer_size": mcmc_replay_buffer_size,
				"train_vast_serverless": train_vast_serverless,
				"vast_endpoint_name": vast_endpoint_name,
				"vast_model_name": vast_model_name,
			},
		)
	except Exception as exc:
		print(f"W&B init failed, continuing without logging: {exc}")
		wandb_run = None

def ensure_openwebtext_prepared():
	train_bin = os.path.join(data_dir, "train.bin")
	val_bin = os.path.join(data_dir, "val.bin")
	if os.path.exists(train_bin) and os.path.exists(val_bin):
		return

	if dataset != "openwebtext":
		if ddp and not master_process:
			return

		configure_hf_cache(hf_cache_dir)
		os.makedirs(data_dir, exist_ok=True)

		try:
			from huggingface_hub import hf_hub_download
		except Exception as exc:
			raise RuntimeError(
				"Missing dataset binaries and huggingface_hub is unavailable. "
				"Install it with `pip install huggingface_hub` or provide local train.bin/val.bin."
			) from exc

		rel_data_dir = os.path.relpath(data_dir, PROJECT_ROOT).replace("\\", "/")
		prefix_candidates = {"", f"{Path(rel_data_dir).name}/"}
		if rel_data_dir and rel_data_dir != ".":
			prefix_candidates.add(f"{rel_data_dir}/")
			if "data/" in rel_data_dir:
				prefix_candidates.add(f"{rel_data_dir[rel_data_dir.index('data/'):]}/")

		repo_candidates = []
		custom_repo = os.environ.get("HF_BINARIES_REPO")
		if custom_repo:
			repo_candidates.append(custom_repo)
		if isinstance(dataset, str):
			repo_candidates.append(dataset)
			if "/" in dataset:
				owner, repo = dataset.split("/", 1)
				repo_candidates.append(f"{owner}/{repo.replace('_', '-')}")
				repo_candidates.append(f"{owner}/{repo.replace('-', '_')}")

		# De-duplicate while preserving order.
		repo_candidates = list(dict.fromkeys(repo_candidates))
		prefix_candidates = [p for p in dict.fromkeys(prefix_candidates) if p is not None]

		last_error = None
		downloaded = False
		for repo_id in repo_candidates:
			for prefix in prefix_candidates:
				try:
					hf_hub_download(
						repo_id=repo_id,
						filename=f"{prefix}train.bin",
						repo_type="dataset",
						local_dir=data_dir,
					)
					hf_hub_download(
						repo_id=repo_id,
						filename=f"{prefix}val.bin",
						repo_type="dataset",
						local_dir=data_dir,
					)
					try:
						hf_hub_download(
							repo_id=repo_id,
							filename=f"{prefix}meta.pkl",
							repo_type="dataset",
							local_dir=data_dir,
						)
					except Exception:
						pass
					print(f"Downloaded dataset binaries from HF dataset '{repo_id}' (prefix='{prefix}') to {data_dir}.")
					downloaded = True
					break
				except Exception as exc:
					last_error = exc
			if downloaded:
				break

		if os.path.exists(train_bin) and os.path.exists(val_bin):
			return

		# Fallback: if prebuilt binaries are not present on HF, build train.bin/val.bin from text records.
		try:
			import json
			from transformers import AutoTokenizer

			def _record_to_text(record):
				if isinstance(record.get("text"), str) and record["text"].strip():
					return record["text"].strip()
				instruction = (record.get("instruction") or "").strip()
				input_text = (record.get("input") or "").strip()
				output_text = (record.get("output") or "").strip()
				if instruction and output_text:
					if input_text:
						return (
							"### Instruction:\n"
							f"{instruction}\n\n"
							"### Input:\n"
							f"{input_text}\n\n"
							"### Response:\n"
							f"{output_text}"
						)
					return (
						"### Instruction:\n"
						f"{instruction}\n\n"
						"### Response:\n"
						f"{output_text}"
					)
				prompt = (record.get("prompt") or "").strip()
				completion = (record.get("completion") or "").strip()
				if prompt and completion:
					return f"{prompt}{completion}"
				for key in ("content", "document"):
					value = record.get(key)
					if isinstance(value, str) and value.strip():
						return value.strip()
				return None

			print(f"Prebuilt binaries not found on HF; building train.bin/val.bin from records for '{dataset}'...")
			tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer, clean_up_tokenization_spaces=False)

			jsonl_paths = sorted(Path(data_dir).rglob("*.jsonl"))
			records = []
			for jsonl_path in jsonl_paths:
				with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as f:
					for line in f:
						line = line.strip()
						if not line:
							continue
						try:
							record = json.loads(line)
						except json.JSONDecodeError:
							continue
						if isinstance(record, dict):
							records.append(record)

			if not records:
				from datasets import load_dataset as hf_load_dataset
				ds = hf_load_dataset(dataset)
				if "train" in ds:
					records = list(ds["train"])
				else:
					first_split = next(iter(ds.keys()))
					records = list(ds[first_split])

			if len(records) < 2:
				raise RuntimeError("Dataset records are too small to create both train and val binaries.")

			val_size = max(1, int(len(records) * 0.01))
			train_source = records[:-val_size]
			val_source = records[-val_size:]

			def _encode_split(split_records):
				ids = []
				for rec in split_records:
					text = _record_to_text(rec)
					if not text:
						continue
					if tokenizer_obj.eos_token:
						text += tokenizer_obj.eos_token
					chunk_ids = tokenizer_obj.encode(text, add_special_tokens=False)
					if chunk_ids:
						ids.extend(chunk_ids)
				return ids

			train_ids = _encode_split(train_source)
			val_ids = _encode_split(val_source)

			if not train_ids or not val_ids:
				raise RuntimeError("Unable to tokenize dataset into non-empty train/val ids.")

			max_id = max(max(train_ids), max(val_ids))
			if max_id >= 2**16:
				raise RuntimeError(
					f"Token id {max_id} exceeds uint16 range required by this training pipeline."
				)

			np.array(train_ids, dtype=np.uint16).tofile(train_bin)
			np.array(val_ids, dtype=np.uint16).tofile(val_bin)

			meta_path = os.path.join(data_dir, "meta.pkl")
			with open(meta_path, "wb") as f:
				pickle.dump({"vocab_size": int(tokenizer_obj.vocab_size)}, f)

			print(
				f"Built dataset binaries at {data_dir}: "
				f"train tokens={len(train_ids):,}, val tokens={len(val_ids):,}."
			)
			return
		except Exception as build_exc:
			raise FileNotFoundError(
				f"Missing dataset binaries at {data_dir}. Expected train.bin and val.bin. "
				f"Attempted HF download from repo candidates {repo_candidates} with path prefixes {prefix_candidates}. "
				f"Last download error: {last_error}. "
				f"Fallback binary build from HF dataset '{dataset}' also failed: {build_exc}"
			) from build_exc

	prepare_path = os.path.join(PROJECT_ROOT, "nanoGPT", "data", "openwebtext", "prepare.py")
	if not os.path.exists(prepare_path):
		raise FileNotFoundError(
			f"Missing OpenWebText prepare script at {prepare_path}. Cannot build dataset."
		)

	if ddp and not master_process:
		return

	# Ensure HF cache uses the configured directory.
	configure_hf_cache(hf_cache_dir)

	os.environ["OPENWEBTEXT_OUT_DIR"] = data_dir

	print("OpenWebText binaries not found. Running prepare.py to build train.bin/val.bin...")
	try:
		runpy.run_path(prepare_path, run_name="__main__")
	except Exception as exc:
		raise RuntimeError(
			"Failed to prepare OpenWebText. Check HF cache permissions or set HF_HOME/HF_DATASETS_CACHE."
		) from exc


def build_prompt_from_batch(batch, max_chars: int) -> str:
	try:
		import tiktoken
		enc = tiktoken.get_encoding("gpt2")
		ids = batch["input_ids"][0, 0].tolist()
		text = enc.decode(ids)
		return text[:max_chars]
	except Exception:
		return " ".join(str(x) for x in batch["input_ids"][0, 0].tolist())[:max_chars]


def run_vast_serverless_request(prompt: str) -> str:
	import asyncio
	from vastai import Serverless

	async def _call():
		async with Serverless() as client:
			endpoint = await client.get_endpoint(name=vast_endpoint_name)
			payload = {
				"input": {
					"model": vast_model_name,
					"prompt": prompt,
					"max_tokens": vast_max_tokens,
					"temperature": vast_temperature,
					"top_k": vast_top_k,
					"top_p": vast_top_p,
					"stream": vast_stream,
				}
			}
			response = await endpoint.request(
				"/v1/completions",
				payload,
				cost=vast_max_tokens,
				stream=vast_stream,
			)
			if vast_stream:
				stream = response["response"]
				chunks = []
				async for event in stream:
					chunks.append(event["choices"][0].get("text", ""))
				return "".join(chunks)
			return response["response"]["choices"][0]["text"]

	return asyncio.run(_call())


# data loader (openwebtext binary, same as nanoGPT)
def get_batch(split: str):
	split_name = "train" if split == "train" else "val"
	data = np.memmap(os.path.join(data_dir, f"{split_name}.bin"), dtype=np.uint16, mode="r")
	ix = torch.randint(len(data) - (block_size + 1), (batch_size,))
	x = torch.stack([torch.from_numpy((data[i : i + block_size + 1]).astype(np.int64)) for i in ix])
	if device_type == "cuda":
		x = x.pin_memory().to(device, non_blocking=True)
	else:
		x = x.to(device)
	return {"input_ids": x.unsqueeze(1)}

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
if os.path.exists(meta_path) and master_process:
	with open(meta_path, "rb") as f:
		meta = pickle.load(f)
	meta_vocab_size = meta.get("vocab_size", None)
	if meta_vocab_size is not None:
		print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
hparams = SimpleNamespace(
	modality="NLP",
	model_name=model_name,
	tokenizer=tokenizer,
	context_length=block_size,
	num_transformer_blocks=n_layer,
	multiheaded_attention_heads=n_head,
	embedding_dim=n_embd,
	ffn_dim_multiplier=ffn_dim_multiplier,
	batch_size_per_device=batch_size,
	ebt_type=ebt_type,
	ebt_norm=ebt_norm,
	ebt_act_func=ebt_act_func,
	dyt_alpha_init=dyt_alpha_init,
	weight_initialization_method=weight_initialization_method,
	weight_initialization_gain=weight_initialization_gain,
	peak_learning_rate=learning_rate,
	accumulate_grad_batches=gradient_accumulation_steps,
	max_steps=max_iters,
	max_scheduling_steps=lr_decay_iters,
	warm_up_steps=warmup_iters,
	gradient_clip_val=grad_clip,
	weight_decay=weight_decay,
	beta1=beta1,
	beta2=beta2,
	mcmc_num_steps=mcmc_num_steps,
	mcmc_step_size=mcmc_step_size,
	mcmc_step_size_learnable=mcmc_step_size_learnable,
	langevin_dynamics_noise=langevin_dynamics_noise,
	langevin_dynamics_noise_learnable=langevin_dynamics_noise_learnable,
	randomize_mcmc_step_size_scale=randomize_mcmc_step_size_scale,
	randomize_mcmc_num_steps=randomize_mcmc_num_steps,
	randomize_mcmc_num_steps_final_landscape=randomize_mcmc_num_steps_final_landscape,
	randomize_mcmc_num_steps_min=randomize_mcmc_num_steps_min,
	denoising_initial_condition=denoising_initial_condition,
	gaussian_random_noise_scaling=gaussian_random_noise_scaling,
	normalize_initial_condition=normalize_initial_condition,
	normalize_initial_condition_only_first_step=normalize_initial_condition_only_first_step,
	vocab_to_embed_uses_prob_dist=vocab_to_embed_uses_prob_dist,
	num_modality_processing_mlp_layers=num_modality_processing_mlp_layers,
	learnable_process_memory=learnable_process_memory,
	process_memory_type=process_memory_type,
	process_memory_linear_layer=process_memory_linear_layer,
	gpus=gpus,
	distributed_strategy=distributed_strategy,
	clamp_futures_grad=clamp_futures_grad,
	clamp_futures_grad_max_change=clamp_futures_grad_max_change,
	absolute_clamp=absolute_clamp,
	clamp_max_after_warm_up=clamp_max_after_warm_up,
	sharpen_predicted_distribution=sharpen_predicted_distribution,
	truncate_mcmc=truncate_mcmc,
	no_mcmc_detach=no_mcmc_detach,
	contrastive_loss=contrastive_loss,
	contrastive_loss_coeff=contrastive_loss_coeff,
	discrete_contrastive_loss_true_logit_val=discrete_contrastive_loss_true_logit_val,
	soften_target_prob_dist=soften_target_prob_dist,
	reconstruction_coeff=reconstruction_coeff,
	mcmc_replay_buffer=False,
	mcmc_replay_buffer_size=mcmc_replay_buffer_size,
	execution_mode=execution_mode,
	debug_unused_parameters=False,
)

model = EBT_NLP(hparams).to(device)

def count_parameters(model_to_count):
	return sum(p.numel() for p in model_to_count.parameters())


def estimate_training_memory_bytes(param_count: int) -> int:
	# Params + grads + Adam states (m, v) in fp32
	bytes_per_param = 4
	param_bytes = param_count * bytes_per_param
	grad_bytes = param_bytes
	adam_bytes = param_bytes * 2
	# Rough activation estimate (very approximate, assumes fp16/bf16 activations)
	act_bytes = batch_size * block_size * n_embd * 2 * 2
	return int((param_bytes + grad_bytes + adam_bytes + act_bytes) * memory_safety_factor)


def get_available_memory_bytes() -> int | None:
	if device_type == "cuda" and torch.cuda.is_available():
		free_bytes, _ = torch.cuda.mem_get_info()
		return int(free_bytes)
	# For MPS/CPU, fall back to system available memory
	try:
		page_size = os.sysconf("SC_PAGE_SIZE")
		avail_pages = os.sysconf("SC_AVPHYS_PAGES")
		return int(page_size * avail_pages)
	except (ValueError, OSError, AttributeError):
		return None


def handle_oom(error: RuntimeError) -> bool:
	global batch_size, block_size, tokens_per_iter
	message = str(error).lower()
	if "out of memory" not in message and "mps backend out of memory" not in message:
		return False

	if not auto_reduce_on_oom:
		return False

	if device_type == "cuda" and torch.cuda.is_available():
		torch.cuda.empty_cache()
	elif device_type == "mps" and hasattr(torch, "mps"):
		torch.mps.empty_cache()

	if batch_size > 1:
		batch_size = max(1, batch_size // 2)
		print(f"OOM detected. Reducing batch_size to {batch_size} and retrying...")
	elif block_size > 64:
		block_size = max(64, block_size // 2)
		print(f"OOM detected. Reducing block_size to {block_size} and retrying...")
	else:
		return False

	tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
	print(f"tokens per iteration will be: {tokens_per_iter:,}")
	return True

if compile:
	print("compiling the model... (takes a ~minute)")
	model = torch.compile(model)

if ddp:
	model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model
train_model = raw_model

if master_process:
	param_count = count_parameters(raw_model)
	print(f"model parameters: {param_count:,} ({param_count/1e6:.2f}M)")
	if check_memory_fit:
		est_bytes = estimate_training_memory_bytes(param_count)
		avail_bytes = get_available_memory_bytes()
		print(f"estimated training memory: {est_bytes/1024**3:.2f} GiB")
		if avail_bytes is not None:
			print(f"available memory: {avail_bytes/1024**3:.2f} GiB")
			if est_bytes > avail_bytes:
				raise RuntimeError(
					"Estimated memory exceeds available memory. Reduce batch size, context length, or model size."
				)

# optimizer
optimizer = torch.optim.AdamW(
	raw_model.parameters(),
	lr=hparams.peak_learning_rate,
	weight_decay=weight_decay,
	betas=(beta1, beta2),
)

def get_uncompiled_model(model_to_use):
	if hasattr(model_to_use, "_orig_mod"):
		return model_to_use._orig_mod
	return model_to_use


def get_model_state_dict(model_to_use):
	return get_uncompiled_model(model_to_use).state_dict()


def normalize_state_dict_keys(state_dict):
	if not any(key.startswith("_orig_mod.") for key in state_dict):
		return state_dict
	return {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}


def parse_iter_from_ckpt_path(ckpt_path: str | os.PathLike | None) -> int | None:
	if not ckpt_path:
		return None
	stem = Path(ckpt_path).stem
	if not stem.startswith("ckpt_iter_"):
		return None
	match = re.fullmatch(r"ckpt_iter_(\d+)", stem)
	if not match:
		return None
	return int(match.group(1))

# resume
iter_num = 0
resume_iter = 0
best_val_loss = 1e9
resume_ckpt_path = None
if resume_latest:
	init_from = "resume"
	resume_ckpt_path = find_latest_checkpoint(out_dir)
	if resume_ckpt_path is None:
		print(f"No checkpoint found in {out_dir}. Starting a new run.")
		init_from = "scratch"
if init_from == "resume":
	ckpt_path = resume_ckpt_path or os.path.join(out_dir, "ckpt.pt")
	checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
	model_state = normalize_state_dict_keys(checkpoint["model"])
	load_target = get_uncompiled_model(raw_model)
	load_target.load_state_dict(model_state, strict=False)
	optimizer.load_state_dict(checkpoint["optimizer"])
	iter_num = checkpoint.get("iter_num", 0)
	iter_from_path = parse_iter_from_ckpt_path(ckpt_path)
	resume_iter = iter_num
	if iter_from_path is not None:
		resume_iter = max(resume_iter, iter_from_path)
		iter_num = resume_iter
	best_val_loss = checkpoint.get("best_val_loss", 1e9)
	if resume_latest and isinstance(checkpoint.get("config"), dict):
		ckpt_config = checkpoint["config"]
		max_iters = ckpt_config.get("max_iters", max_iters)
		lr_decay_iters = ckpt_config.get("lr_decay_iters", lr_decay_iters)
		config["max_iters"] = max_iters
		config["lr_decay_iters"] = lr_decay_iters

scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16" and device_type == "cuda"))

loss_log_path = os.path.join(out_dir, "losses.pkl")
loss_log = {"eval": [], "train": []}
if master_process and os.path.exists(loss_log_path):
	try:
		with open(loss_log_path, "rb") as f:
			loaded = pickle.load(f)
		if isinstance(loaded, dict):
			loss_log.update({k: list(v) for k, v in loaded.items() if k in loss_log})
	except Exception:
		pass

def save_loss_log() -> None:
	if not master_process:
		return
	with open(loss_log_path, "wb") as f:
		pickle.dump(loss_log, f)

@torch.no_grad()
def estimate_loss():
	out = {}
	train_model.eval()
	for split in ["train", "val"]:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			batch = get_batch(split)
			with ctx:
				loss_dict = train_model.forward_loss_wrapper(batch, phase="valid")
				loss = loss_dict["loss"]
			losses[k] = loss.item()
		out[split] = losses.mean()
	train_model.train()
	return out

def get_lr(it):
	if it < warmup_iters:
		return hparams.peak_learning_rate * (it + 1) / (warmup_iters + 1)
	if it > lr_decay_iters:
		return min_lr
	decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
	assert 0 <= decay_ratio <= 1
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return min_lr + coeff * (hparams.peak_learning_rate - min_lr)

if ddp and torch.distributed.is_initialized():
	if master_process:
		ensure_openwebtext_prepared()
	torch.distributed.barrier()
else:
	ensure_openwebtext_prepared()

# training loop
batch = get_batch("train")
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0

stop_training = False

while True:
	retries_left = oom_retries
	while True:
		try:
			global_iter = resume_iter + local_iter_num
			lr = get_lr(global_iter) if decay_lr else hparams.peak_learning_rate
			for param_group in optimizer.param_groups:
				param_group["lr"] = lr

			if global_iter % eval_interval == 0 and master_process:
				losses = estimate_loss()
				print(f"step {global_iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
				loss_log["eval"].append(
					{
						"iter": global_iter,
						"train": float(losses["train"]),
						"val": float(losses["val"]),
					}
				)
				if train_vast_serverless:
					try:
						prompt = build_prompt_from_batch(batch, vast_prompt_max_chars)
						completion = run_vast_serverless_request(prompt)
						print("VAST completion sample:")
						print(completion)
						if wandb_run is not None:
							wandb_run.log(
								{
									"iter": iter_num,
									"vast_completion": completion,
								},
							)
					except Exception as exc:
						print(f"VAST serverless request failed: {exc}")
				if wandb_run is not None:
					wandb_run.log(
						{
							"iter": global_iter,
							"train_loss": float(losses["train"]),
							"val_loss": float(losses["val"]),
							"lr": lr,
						},
					)
				save_loss_log()
				if losses["val"] < best_val_loss or always_save_checkpoint:
					best_val_loss = losses["val"]
					if global_iter > 0:
						checkpoint = {
							"model": get_model_state_dict(raw_model),
							"optimizer": optimizer.state_dict(),
							"iter_num": global_iter,
							"best_val_loss": best_val_loss,
							"config": config,
						}
						print(f"saving checkpoint to {out_dir}")
						torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
						ckpt_iter_path = os.path.join(out_dir, f"ckpt_iter_{global_iter}.pt")
						torch.save(checkpoint, ckpt_iter_path)
			if global_iter == 0 and eval_only:
				stop_training = True
				break

			for micro_step in range(gradient_accumulation_steps):
				if ddp:
					model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
				with ctx:
					loss_dict = train_model.forward_loss_wrapper(batch, phase="train")
					loss = loss_dict["loss"] / gradient_accumulation_steps
				batch = get_batch("train")
				scaler.scale(loss).backward()

			if grad_clip != 0.0:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad(set_to_none=True)

			break
		except RuntimeError as exc:
			if retries_left > 0 and handle_oom(exc):
				retries_left -= 1
				batch = get_batch("train")
				continue
			raise

	if stop_training:
		break

	t1 = time.time()
	dt = t1 - t0
	t0 = t1
	global_iter = resume_iter + local_iter_num
	if global_iter % log_interval == 0 and master_process:
		lossf = loss.item() * gradient_accumulation_steps
		if local_iter_num >= 5:
			running_mfu = running_mfu if running_mfu != -1.0 else 0.0
		print(f"iter {global_iter}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
		loss_log["train"].append({"iter": global_iter, "loss": float(lossf)})
		if wandb_run is not None:
			wandb_run.log(
				{
					"iter": global_iter,
					"loss": float(lossf),
					"time_ms": dt * 1000.0,
					"mfu": running_mfu * 100.0,
					"lr": lr,
				},
			)
		save_loss_log()

	local_iter_num += 1

	if global_iter >= max_iters:
		break

if ddp:
	destroy_process_group()

if wandb_run is not None:
	wandb_run.finish()
