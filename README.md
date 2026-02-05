# Energy-Based Transformer (EBT) Training Hub

This repository brings together **Energy-Based Transformers (EBT)** with the training workflows and practical tooling from **nanochat** and **nanoGPT**. The goal is to train EBT models using the scalable, minimal, and reproducible training approaches proven in those projects, while preserving EBT’s core promise: **generalizable reasoning and System 2 thinking across modalities**.

Key ingredients combined here:
- **EBT architecture + training/inference stack** (from EBT).
- **Practical LLM training harness** (from nanochat).
- **Minimal, hackable GPT training loop** (from nanoGPT).

## Overview (What you can do here)
- Train EBTs for **NLP, image, and video** modalities.
- Run minimal EBT vs Transformer++ baselines.
- Reuse nanochat/nanoGPT-style workflows for data prep, training, evaluation, and inference.

## Visuals

**EBT architecture**

<img src="EBT/assets/model.png" alt="EBT Autoregressive Model" width="100%" />

**nanochat branding & scaling context**

<img src="nanochat/dev/nanochat.png" alt="nanochat logo" width="50%" />
<img src="nanochat/dev/scaling_laws_jan26.png" alt="nanochat scaling laws" width="100%" />

**nanoGPT reference loss curve**

<img src="nanoGPT/assets/gpt2_124M_loss.png" alt="nanoGPT GPT-2 124M loss" width="100%" />

## Install (combined from submodules)

### EBT environment (recommended)
Use the EBT setup for the core EBT training stack:

```bash
conda create -n ebt python=3.11
conda activate ebt
pip install --upgrade pip
pip install -r EBT/requirements.txt
```

> **GPU note (Linux, CUDA 12.1):** PyTorch GPU wheels live on a separate index. Either set `export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121` before running install, or append `--extra-index-url https://download.pytorch.org/whl/cu121` to the `pip install` command. The CUDA helper wheels listed in the EBT requirements resolve only on Linux/x86_64 and are skipped on macOS/Windows.

> **Heads-up:** Don’t use `conda install --file requirements.txt` for EBT. The file uses PEP 508 environment markers that Conda can’t parse. Always install via `pip install -r`.

Optional alternatives:
- EBT `EBT/gh200_requirements.txt` for GH200s.
- EBT `EBT/loose_requirements.txt` without NVIDIA/PyTorch/Triton pins.
- Cross‑platform `EBT/environment.yml`.

### nanochat environment
nanochat uses a Python/uv workflow (see [nanochat/README.md](nanochat/README.md)). If you want to run its scripts directly, set up its env separately, then activate when needed.

### nanoGPT dependencies (minimal)
If you only need nanoGPT-style training loop dependencies:

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## API keys & tokens (VAST, Hugging Face, Weights & Biases)

This repo reads environment variables from a project .env file at the repo root (and also from /.env if you keep one there). These are used for dataset caching, experiment tracking, and (optionally) VAST API access.

### Hugging Face (datasets + models)
Where to get it:
- Create a token at https://huggingface.co/settings/tokens

How it’s used:
- `HF_TOKEN` lets datasets/models download when required.
- `HF_CACHE_DIR`, `HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE`, `XDG_CACHE_HOME` control where caches are stored.
- The OpenWebText prep step still builds `train.bin`/`val.bin` from the HF cache (the cache contains raw shards, not the .bin files).

### Weights & Biases (training metrics)
Where to get it:
- Create an API key at https://wandb.ai/settings

How it’s used:
- If `WANDB_API_KEY` is set, training in [train.py](train.py) automatically starts a W&B run.
- `WANDB_ENTITY` and `WANDB_PROJECT` select the destination workspace and project.

### VAST API (optional)
Where to get it:
- Create an API key in your VAST account: https://cloud.vast.ai/

How it’s used:
- `VAST_API_KEY` is picked up by the VAST SDK.
- You can verify connectivity with [test_vast.py](test_vast.py).

### Example .env
Place this in the repo root as .env:

```dotenv
HF_TOKEN=hf_...your_token...
HF_CACHE_DIR=./hf_cache
HF_HOME=./hf_cache
HF_DATASETS_CACHE=./hf_cache/datasets
HF_HUB_CACHE=./hf_cache/hub
XDG_CACHE_HOME=./hf_cache

WANDB_API_KEY=wandb_...your_key...
WANDB_ENTITY=your_team_or_user
WANDB_PROJECT=energy-coder

VAST_API_KEY=vast_...your_key...
```

### Test VAST API setup
Once `VAST_API_KEY` is set, run:

```bash
python test_vast.py
```

## Training scripts (how to run)

### EBT pretraining
EBT scripts live under `EBT/job_scripts/` per modality.

Example (NLP System 1 pretraining):

```bash
bash EBT/job_scripts/nlp/pretrain/ebt_s1.sh
```

Optional (HPC / slurm):

```bash
bash EBT/slurm_executor.sh reference_a100 EBT/job_scripts/nlp/pretrain/ebt_s1.sh
```

Important knobs in these scripts:
- **RUN_NAME**
- **MODEL_NAME**
- **MODEL_SIZE** (sets layers/heads/embed dims automatically)

### EBT inference
Use modality-specific inference scripts under `EBT/inference/` and `EBT/job_scripts/*/inference/`.

Example:

```bash
bash EBT/job_scripts/nlp/inference/ebt.sh
```

Key flags:
- `--only_test_model_ckpt`
- `--only_test`
- `--execution_mode "inference"`

### Minimal EBT vs Transformer++ training loop
This is a compact training loop for quick experimentation:

```bash
python EBT/example_code/minimal_nlp_training_loop.py
```

### nanochat speedrun (GPT‑2‑grade baseline)
For GPT‑2‑grade LLM training + chat UI in nanochat:

```bash
bash nanochat/runs/speedrun.sh
python -m nanochat/scripts.chat_web
```

> Use a `screen` or `tmux` session; the speedrun takes a few hours on 8xH100.

### nanoGPT quick start (Shakespeare)
Character‑level GPT on a tiny dataset:

```bash
python nanoGPT/data/shakespeare_char/prepare.py
python nanoGPT/train.py nanoGPT/config/train_shakespeare_char.py
python nanoGPT/sample.py --out_dir=out-shakespeare-char
```

## How this repo is organized
- **EBT/** — core Energy‑Based Transformer codebase, job scripts, inference, and datasets.
- **nanochat/** — end‑to‑end LLM training harness (tokenization → pretrain → SFT/RL → eval → chat UI).
- **nanoGPT/** — minimal GPT training loop, data prep, and sampling.
- Root-level scripts (e.g., [train.py](train.py)) can be used for local experiments.

## References
- EBT paper and resources: https://energy-based-transformers.github.io/
- nanochat: https://github.com/karpathy/nanochat
- nanoGPT: https://github.com/karpathy/nanoGPT

## Notes
- For video datasets and inference in EBT, see [EBT/data/vid/README.md](EBT/data/vid/README.md) and [EBT/inference/vid/README.md](EBT/inference/vid/README.md).
- For EBT internals and code flow, see [EBT/CODE_INFO.md](EBT/CODE_INFO.md).
