#!/bin/bash

num_iter=${1}          # MixP iteration number (e.g., 1)
K=${2:-4}              # must be even
type="mixp"
max_input_length=64
max_new_tokens=64
temperature=0.7
seed=$num_iter
output_dir="./data"

# 🔍 Ensure K is even
if [ $((K % 2)) -ne 0 ]; then
  echo "❌ ERROR: K must be even! Provided K=$K"
  exit 1
fi

# 🎯 Policy model for MixP → best online so far
# For MixP – use last online model (iter2) as policy
policy_model="models/online_410m_iter2"


# 🧠 Reference model → offline SFT baseline
ref_model="models/offline_pythia410m_iter1"

prompt_dir="PKU-Alignment/PKU-SafeRLHF-prompt"

echo ">>> [MixP GENERATION] Iter $num_iter"
echo ">>> Policy model:     $policy_model"
echo ">>> Reference model:  $ref_model"

# 🚨 Validate model paths
if [ ! -d "$policy_model" ]; then
  echo "❌ ERROR: Policy model not found at $policy_model"
  exit 1
fi
if [ ! -d "$ref_model" ]; then
  echo "❌ ERROR: Reference model not found at $ref_model"
  exit 1
fi

# 🧹 Cleanup
rm -f ${output_dir}/gen_data0.json \
      ${output_dir}/gen_data0_policy.json \
      ${output_dir}/gen_data0_ref.json

# 🔹 Generate from policy model
CUDA_VISIBLE_DEVICES=0 python ./generation/safe_rlhf/get_hf2.py \
  --model_name_or_path ${policy_model} \
  --dataset_name_or_path ${prompt_dir} \
  --output_dir ${output_dir} \
  --local_index 0 --my_world_size 1 \
  --K $((K / 2)) \
  --max_input_length ${max_input_length} \
  --max_new_tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --eos_ids 2 \
  --num_iter ${num_iter} \
  --seed ${seed}

mv ${output_dir}/gen_data0.json ${output_dir}/gen_data0_policy.json

# 🔸 Generate from baseline reference model
CUDA_VISIBLE_DEVICES=0 python ./generation/safe_rlhf/get_hf2.py \
  --model_name_or_path ${ref_model} \
  --dataset_name_or_path ${prompt_dir} \
  --output_dir ${output_dir} \
  --local_index 0 --my_world_size 1 \
  --K $((K / 2)) \
  --max_input_length ${max_input_length} \
  --max_new_tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --eos_ids 2 \
  --num_iter ${num_iter} \
  --seed ${seed}

mv ${output_dir}/gen_data0.json ${output_dir}/gen_data0_ref.json

# 🔁 Merge them
python ./generation/merge_mixp.py \
  --policy ./data/gen_data0_policy.json \
  --ref ./data/gen_data0_ref.json \
  --output ./data/gen_data_iter${num_iter}_${type}.json

echo ">>> [GENERATION COMPLETE] → ./data/gen_data_iter${num_iter}_${type}.json"
