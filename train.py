"""
Train an Energy Based Transformer (EBT) on OpenWebText using the same
training loop structure as nanoGPT.

Example (single GPU):
	python train.py --dataset=openwebtext --data_dir=nanoGPT/data/openwebtext

Example (DDP, 4 GPUs):
	torchrun --standalone --nproc_per_node=4 train.py --dataset=openwebtext --data_dir=nanoGPT/data/openwebtext
"""

import os
import time
import math
import pickle
import runpy
from contextlib import nullcontext
from types import SimpleNamespace

import sys
from pathlib import Path

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
	os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
	os.environ["XDG_CACHE_HOME"] = str(cache_root)


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "EBT"))

load_env_file(PROJECT_ROOT / ".env")
hf_cache_env = os.environ.get("HF_CACHE_DIR")
hf_cache_dir = Path(hf_cache_env).expanduser() if hf_cache_env else (PROJECT_ROOT / ".hf_cache")
configure_hf_cache(hf_cache_dir)

from model.nlp.ebt import EBT_NLP

# -----------------------------------------------------------------------------
# default config values designed to mirror nanoGPT training loop behavior
# I/O
out_dir = "out_ebt"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"  # 'scratch' or 'resume'

# data
dataset = "openwebtext"
data_dir = os.path.join("nanoGPT", "data", dataset)
gradient_accumulation_steps = 5 * 8
batch_size = 4  # micro-batch size (smaller for local machines)
block_size = 256

# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0

# optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP settings
backend = "nccl"

# system
device = "auto"  # 'auto', 'cuda', 'mps', or 'cpu'
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = False

# memory check
check_memory_fit = True
memory_safety_factor = 1.2
auto_reduce_on_oom = True
oom_retries = 3

# EBT-specific defaults
tokenizer = "gpt2"
model_name = "ebt"
ebt_type = "default"
ebt_norm = "rms"
ebt_act_func = "silu"
ffn_dim_multiplier = None
dyt_alpha_init = 0.5
weight_initialization_method = "xavier"
weight_initialization_gain = 1.0

mcmc_num_steps = 1
mcmc_step_size = 60.0
mcmc_step_size_learnable = False
langevin_dynamics_noise = 0.0
langevin_dynamics_noise_learnable = False
randomize_mcmc_step_size_scale = 1.0
randomize_mcmc_num_steps = 0
randomize_mcmc_num_steps_final_landscape = False
randomize_mcmc_num_steps_min = 0
denoising_initial_condition = "random_noise"
gaussian_random_noise_scaling = 1.0
normalize_initial_condition = False
normalize_initial_condition_only_first_step = False
vocab_to_embed_uses_prob_dist = False
num_modality_processing_mlp_layers = 1
learnable_process_memory = False
process_memory_type = None
process_memory_linear_layer = False
clamp_futures_grad = False
clamp_futures_grad_max_change = 9.0
absolute_clamp = 0.0
clamp_max_after_warm_up = 0.0
sharpen_predicted_distribution = 0.0
truncate_mcmc = False
no_mcmc_detach = False
contrastive_loss = False
contrastive_loss_coeff = 0.0005
discrete_contrastive_loss_true_logit_val = 0.0
soften_target_prob_dist = 0.0
reconstruction_coeff = 1.0

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join("nanoGPT", "configurator.py")).read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

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
	assert gradient_accumulation_steps % ddp_world_size == 0
	gradient_accumulation_steps //= ddp_world_size
else:
	master_process = True
	seed_offset = 0
	ddp_world_size = 1

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

ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type in ("cpu", "mps") else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

def ensure_openwebtext_prepared():
	train_bin = os.path.join(data_dir, "train.bin")
	val_bin = os.path.join(data_dir, "val.bin")
	if os.path.exists(train_bin) and os.path.exists(val_bin):
		return

	if dataset != "openwebtext":
		raise FileNotFoundError(
			f"Missing dataset binaries at {data_dir}. Expected train.bin and val.bin."
		)

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
	execution_mode="pretrain",
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
train_model = model

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
optimizer = torch.optim.AdamW(raw_model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

# resume
iter_num = 0
best_val_loss = 1e9
if init_from == "resume":
	ckpt_path = os.path.join(out_dir, "ckpt.pt")
	checkpoint = torch.load(ckpt_path, map_location=device)
	raw_model.load_state_dict(checkpoint["model"])
	optimizer.load_state_dict(checkpoint["optimizer"])
	iter_num = checkpoint.get("iter_num", 0)
	best_val_loss = checkpoint.get("best_val_loss", 1e9)

scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16" and device_type == "cuda"))

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
		return learning_rate * (it + 1) / (warmup_iters + 1)
	if it > lr_decay_iters:
		return min_lr
	decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
	assert 0 <= decay_ratio <= 1
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return min_lr + coeff * (learning_rate - min_lr)

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
			lr = get_lr(iter_num) if decay_lr else learning_rate
			for param_group in optimizer.param_groups:
				param_group["lr"] = lr

			if iter_num % eval_interval == 0 and master_process:
				losses = estimate_loss()
				print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
				if losses["val"] < best_val_loss or always_save_checkpoint:
					best_val_loss = losses["val"]
					if iter_num > 0:
						checkpoint = {
							"model": raw_model.state_dict(),
							"optimizer": optimizer.state_dict(),
							"iter_num": iter_num,
							"best_val_loss": best_val_loss,
							"config": config,
						}
						print(f"saving checkpoint to {out_dir}")
						torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
			if iter_num == 0 and eval_only:
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
	if iter_num % log_interval == 0 and master_process:
		lossf = loss.item() * gradient_accumulation_steps
		if local_iter_num >= 5:
			running_mfu = running_mfu if running_mfu != -1.0 else 0.0
		print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

	iter_num += 1
	local_iter_num += 1

	if iter_num > max_iters:
		break

if ddp:
	destroy_process_group()
