#!/bin/bash

python train.py --domain_name franka-FrankaBinPickSmall_v2d --cameras 8 10 --demo_model_dir demos/bin_pick --warmup_cpc 1600 --reward_type sparse  --frame_stack 1 --num_updates 1 --observation_type hybrid --encoder_type pixel --work_dir ./data/FetchPickAndPlace-v1 --pre_transform_image_size 100 --image_size 84 --agent rad_sac --seed -1 --critic_lr 0.001 --actor_lr 0.001 --eval_freq 1000 --batch_size 128  --num_train_steps 200000 --save_tb --save_video --demo_model_step 195000 --demo_samples 500  --warmup_cpc_ema  --change_model
