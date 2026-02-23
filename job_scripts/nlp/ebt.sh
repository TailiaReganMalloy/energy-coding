export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens7

torchrun --standalone --nproc_per_node=8  train.py \
    --dataset=openwebtext \
    --data_dir=nanoGPT/data/openwebtext \
    --out_dir=out_ebt_openwebtext \
    --resume_latest=True \
    --max_steps=1000000 \
    --max_scheduling_steps=1000000 \
    --warm_up_steps=2000 \
    --eval_interval=5000 \
    --batch_size_per_device=1 \
    --accumulate_grad_batches=1 \
    --context_length=512 \
    --num_transformer_blocks=24 \
    --multiheaded_attention_heads=16 \
    --embedding_dim=1024 \
    --tokenizer=gpt2 \
    --gpus=8 \
    --distributed_strategy=ddp \
    --mcmc_num_steps=2 \
    --mcmc_replay_buffer_size=48 \
    --mcmc_step_size=2.0 \
    --normalize_initial_condition=True \
    --clamp_futures_grad=True
else
  torchrun --standalone --nproc_per_node=4 train.py \
    --dataset=openwebtext \
    --data_dir=nanoGPT/data/openwebtext \
    --out_dir=out_ebt_openwebtext \
    --resume_latest=True \
    --max_iters=500000 \
    --lr_decay_iters=500000 \
    --warmup_iters=2000 \
    --eval_interval=500 \
    --batch_size=2 \
    --gradient_accumulation_steps=4 \
    --block_size=512 \
    --n_layer=8 \
    --n_head=8 \
    --n_embd=512 \
    --tokenizer=gpt2 \
    --compile=True \
    --mcmc_num_steps=2 \
    --mcmc_step_size=16.0 \
    --normalize_initial_condition=True \
    --clamp_futures_grad=True
fi

# scp ~/models/my_model.pt tailiamalloy_gmail_com@instance-20260217-135005:~/energy-coding/models/

# scp ./Programing/ckpt_iter_910000.pt tailiamalloy_gmail_com@34.28.128.97:~/energy-coding/out_ebt_openwebtext/

# gcloud compute disks resize instance-20260217-135005 --size=1000 --zone=us-central1-a 