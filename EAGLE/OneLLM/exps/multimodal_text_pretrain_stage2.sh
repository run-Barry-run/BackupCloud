#!/bin/bash

LLAMA_7B_PATH="./weights/"
OUTPUT_DIR="./output"
IMAGE_TEXT_MODEL="./weights/"

torchrun pretrain_single.py \
--epochs 1 --dataset audio \
--batch_size 4 --accum_iter 1 \
--model_parallel_size 1 \
--data_parallel sdp \
--save_consolidated \
--llama_type onellm \
--llama_ckpt_dir ${LLAMA_7B_PATH} \
--llama_config config/llama2/7B.json \
--tokenizer_path config/llama2/tokenizer.model \
--init_from ${IMAGE_TEXT_MODEL} \
--init_from_image \
--auto_resume \
--weight_decay 0.1 --output_dir ${OUTPUT_DIR} \
--warmup_iters 2000 --lr_decay_iters 400000 --lr 1e-5 --min_lr 5e-6 --clip_grad 2 \
--save_freq 1000 \
2>&1 >> ~/Documents/OneLLM/logs/pretrain_stage2.log