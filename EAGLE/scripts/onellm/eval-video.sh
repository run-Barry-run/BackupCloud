MODEL_PATH=./OneLLM/weights/consolidated.00-of-01.pth
MODEL_NAME=onellm
# TASKS=mvbench
# TASKS=activitynetqa
TASKS=youcook2_val

CUDA_VISIBLE_DEVICES='3' python evaluate_lmms_eval.py \
           --model onellm \
           --model_args pretrained=${MODEL_PATH},modality='video' \
           --tasks ${TASKS} \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix ${MODEL_NAME}_mmbench \
           --output_path ./output/eval/ 
echo $MODEL_PATH
echo $TASKS
