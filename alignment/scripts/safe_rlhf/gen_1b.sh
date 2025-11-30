#!/bin/bash
num_iter=${1}         # iteration number
K=${2:-3}             # Number of samples to generate
type=${3:-online}     # Type: online / mixp (optional)
max_input_length=64
max_new_tokens=64
temperature=0.7
seed=$num_iter

# get_hf2.py writes into ./data/data_0.json
output_dir="./data"

# 🎯 Model Selection (Hybrid Strategy)
if [ "$num_iter" -eq 1 ]; then
  infer_model="EleutherAI/pythia-1.4b"        # Use 1.4B only for the FIRST generation
else
  infer_model="models/${type}_410m_iter$((num_iter-1))"  # Online policy afterward
fi

prompt_dir="PKU-Alignment/PKU-SafeRLHF-prompt"

echo ">>> [GENERATION] Iter $num_iter using model: $infer_model"

# Cleanup previous generation (corrected filename)
rm -f ${output_dir}/data_0.json

CUDA_VISIBLE_DEVICES=0 python ./generation/safe_rlhf/get_hf2.py \
  --model_name_or_path ${infer_model} \
  --dataset_name_or_path ${prompt_dir} \
  --output_dir ${output_dir} \
  --local_index 0 --my_world_size 1 \
  --K ${K} \
  --temperature ${temperature} \
  --eos_ids 2 \
  --max_input_length ${max_input_length} \
  --max_new_tokens ${max_new_tokens} \
  --num_iter ${num_iter} \
  --seed ${seed}

# Rename the generated file (corrected path)
raw_file="${output_dir}/data_0.json"
final_file="./data/gen_data_iter${num_iter}_${type}.json"

if [ -f "$raw_file" ]; then
  mv "$raw_file" "$final_file"
  echo ">>> [GENERATION COMPLETE] → ${final_file}"
else
  echo "❌ ERROR: Expected ${raw_file} but it was not created."
fi
