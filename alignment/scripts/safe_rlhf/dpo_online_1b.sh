#!/bin/bash
num_iter=$1

if [ "$num_iter" -eq 1 ]; then
  model_path="EleutherAI/pythia-410m"
else
  model_path="./models/online_410m_iter$((num_iter-1))"
fi

ref_model="EleutherAI/pythia-410m"
train_file="./data/data_with_rewards_iter${num_iter}_online.json"
output_dir="./models/online_410m_iter${num_iter}"

echo ">>> [DPO TRAINING] Iter ${num_iter}"
echo "Using policy model: ${model_path}"
echo "Train file: ${train_file}"
echo "Saving to: ${output_dir}"

CUDA_VISIBLE_DEVICES=0 python ./dpo_iteration/run_dpo.py \
  --run_name "online_410m_iter${num_iter}" \
  --output_dir "${output_dir}" \
  --model_name_or_path "${model_path}" \
  --ref_model "${ref_model}" \
  --train_dir "${train_file}" \
  --eval_dir "${train_file}" \
  --learning_rate 1e-6 \
  --max_steps 300 \
  --loss_type sigmoid \
  --choose_type max_min \
  --lr_scheduler_type cosine \
  --max_length 192 \
  --max_prompt_length 128 \
  --gradient_checkpointing True \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --flash_attention False \
  --report_to wandb

echo ">>> [TRAINING COMPLETE] → ${output_dir}"
