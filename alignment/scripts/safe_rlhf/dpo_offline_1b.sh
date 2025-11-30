#!/bin/bash
num_iter=${1:-1}

model_path="EleutherAI/pythia-410m"
train_dir="./data/data_with_rewards_iter${num_iter}_offline.json"
eval_dir="./data/data_with_rewards_iter${num_iter}_offline.json"

echo "🔥 Running DPO 410M with aggressive memory optimization"
echo "📂 Dataset: ${train_dir}"

accelerate launch dpo_iteration/run_dpo.py \
  --run_name "offline_pythia410m_iter${num_iter}" \
  --output_dir "./models/offline_pythia410m_iter${num_iter}" \
  --model_name_or_path $model_path \
  --ref_model $model_path \
  --train_dir $train_dir \
  --eval_dir $eval_dir \
  --learning_rate 1e-6 \
  --max_steps 1500 \
  --choose_type max_min \
  --loss_type sigmoid \
  --lr_scheduler_type cosine \
  --max_length 128 \
  --max_prompt_length 64 \
  --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 1 \
  --num_train_epochs 1 \
  --report_to wandb \
  --gradient_checkpointing True \
  --flash_attention True \
  --optimizer_type paged_adamw_8bit
