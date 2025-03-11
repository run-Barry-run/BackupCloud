# MODEL_PATH=./checkpoints/final_result/video/video_finetune_1epoch
# MODEL_PATH=./checkpoints/Baseline/Video/pr_llm/video_llama3.2-1b-finetune_pr_llm/checkpoint-2000
# MODEL_PATH=./checkpoints/Baseline/Video/en_pr/finetune-video-llama3.2-1b-ori-token
MODEL_PATH=./checkpoints/Baseline/Video/pr/finetune-video-llama3.2-1b-ori-token
MODEL_NAME=eagle
# CONV_MODE=vicuna_v1
CONV_MODE=llama3

# TASKS=mvbench
TASKS=activitynetqa
# TASKS=librispeech
# TASKS=clotho_aqa_test
# TASKS=youcook2_val

CUDA_VISIBLE_DEVICES='0' accelerate launch --num_processes=1 \
           evaluate_lmms_eval.py \
           --model eagle \
           --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
           --tasks ${TASKS} \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix ${MODEL_NAME} \
           --output_path ./output/eval/ 
echo $MODEL_PATH
echo $TASKS
# CUDA_VISIBLE_DEVICES='1' python evaluate_lmms_eval.py \
#            --model eagle \
#            --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
#            --tasks ${TASKS} \
#            --batch_size 1 \
#            --log_samples \
#            --log_samples_suffix ${MODEL_NAME}_mmbench \
#            --output_path ./output/eval/ 
        #    --verbosity=DEBUG
# echo $MODEL_PATH
# echo $TASKS