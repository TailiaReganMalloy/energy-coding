torchrun --standalone --nproc_per_node=4 train.py \
    --dataset=openwebtext \
    --data_dir=nanoGPT/data/openwebtext \
    --out_dir=out_ebt_openwebtext \
    --resume_latest=True \
    --max_steps=1000000 \
    --max_scheduling_steps=1000000 \
    --warm_up_steps=2000 \
    --eval_interval=5000 \
    --batch_size_per_device=2 \
    --accumulate_grad_batches=4 \
    --context_length=512 \
    --num_transformer_blocks=24 \
    --multiheaded_attention_heads=32 \
    --embedding_dim=2048 \
    --tokenizer=gpt2 \
    --gpus=8 \
    --distributed_strategy=ddp \
    --compile=True \
    --mcmc_num_steps=2 \
    --mcmc_replay_buffer_size=192 \
    --mcmc_step_size=2.0 \
    --normalize_initial_condition=True \
    --clamp_futures_grad=True

# --lr 0002
