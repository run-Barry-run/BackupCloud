MODEL_PATH=./OneLLM/weights/consolidated.00-of-01.pth
MODEL_NAME=onellm
TASKS=mme

CUDA_VISIBLE_DEVICES='0' python evaluate_lmms_eval.py \
           --model onellm \
           --model_args pretrained=${MODEL_PATH} \
           --tasks ${TASKS} \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix ${MODEL_NAME}_mmbench \
           --output_path ./output/eval/ 
echo $MODEL_PATH
echo $TASKS
