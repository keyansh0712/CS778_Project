#!/bin/bash
num_iter=${1}

policy_model="models/online_410m_iter2"             # Use latest online model
ref_model="models/offline_pythia410m_iter1"         # Stable reference
train_file="./data/data_with_rewards_iter${num_iter}_mixp.json"

echo ">>> [DPO TRAINING - MixP] Iter ${num_iter}"
echo ">>> Policy model: ${policy_model}"
echo ">>> Reference model: ${ref_model}"
echo ">>> Train file: ${train_file}"

CUDA_VISIBLE_DEVICES=0 python ./dpo_iteration/run_dpo.py \
  --run_name "mixp_410m_iter${num_iter}" \
  --output_dir "./models/mixp_410m_iter${num_iter}" \
  --model_name_or_path ${policy_model} \
  --ref_model ${ref_model} \
  --train_dir ${train_file} \
  --eval_dir ${train_file} \
  --learning_rate 5e-7 \
  --max_steps 400 \
  --choose_type max_min \
  --loss_type rev_kl \
  --lr_scheduler_type cosine \
  --max_length 192 \
  --max_prompt_length 128 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --margin_scale 4 \
  --report_to wandb \
  --flash_attention False

echo ">>> [TRAINING COMPLETE] → ./models/mixp_410m_iter${num_iter}"
