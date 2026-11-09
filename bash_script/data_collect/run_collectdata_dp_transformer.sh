#!/bin/bash

GPU=0

NUM_EPISODES=500

ALL_CONFIGS=(
  "datacollect_diffusion_policy_transformer_lowdim_lift_mh_abs.yaml"
  "datacollect_diffusion_policy_transformer_lowdim_can_mh_abs.yaml"
  "datacollect_diffusion_policy_transformer_lowdim_square_mh_abs.yaml"
  "datacollect_diffusion_policy_transformer_lowdim_transport_mh_abs.yaml"
  "datacollect_diffusion_policy_transformer_lowdim_tool_hang_ph_abs.yaml"
)

for CONFIG_NAME in "${ALL_CONFIGS[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --config-dir=diffusion_policy/config/data_collect/diffusion_policy_transformer \
    --config-name=$CONFIG_NAME \
    name='datacollect_diffusion_transformer_lowdim' \
    collecting.num_episodes=${NUM_EPISODES} \
    collecting.render_image=True \
    task.env_runner.render_hw="[224, 224]" \
    hydra.run.dir='logs/collect_data/${task_name}/${name}_${task_name}'
done
