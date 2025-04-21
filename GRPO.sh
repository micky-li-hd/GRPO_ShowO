# accelerate launch --main_process_port 1234 --config_file llava/configs/accelerate_configs/zero3.yaml \
#     --num_processes=7 llava/train/llava_trainer_grpo.py \
#     --config llava/configs/config_grpo.yaml \
#     --data_path /path_to_annotation_file
export PYTHONPATH=/home/v-haodongli/SimpleAR:$PYTHONPATH
accelerate launch --main_process_port 1234 --config_file simpar/configs/accelerate_configs/zero3.yaml \
    --num_processes=1 simpar/train/llava_trainer_grpo.py \
    --config simpar/configs/config_grpo_clip.yaml \
    --data_path simpar/data/generation_prompts.txt