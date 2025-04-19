accelerate launch --main_process_port 1234 --config_file llava/configs/accelerate_configs/zero3.yaml \
    --num_processes=7 llava/train/llava_trainer_grpo.py \
    --config llava/configs/config_grpo.yaml \
    --data_path /path_to_annotation_file

    