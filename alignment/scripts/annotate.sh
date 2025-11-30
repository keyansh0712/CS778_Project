#!/bin/bash
num_iter=${1}           # iteration number
type=${2}               # mixp / online / offline
K=${3:-3}               # default to 3 if not provided

# GPU selection (optional)
export CUDA_VISIBLE_DEVICES=0

echo ">>> [ANNOTATION] Iter ${num_iter} (${type})"
echo ">>> Input:  ./data/gen_data_iter${num_iter}_${type}.json"
echo ">>> Output: ./data/data_with_rewards_iter${num_iter}_${type}.json"
echo ">>> Using K=${K}, Reward Model=OpenAssistant/reward-model-deberta-v3-large"

python ./annotate_data/get_rewards.py \
    --dataset_name_or_path "./data/gen_data_iter${num_iter}_${type}.json" \
    --output_dir "./data/data_with_rewards_iter${num_iter}_${type}.json" \
    --K ${K} \
    --reward_name_or_path "OpenAssistant/reward-model-deberta-v3-large"

echo ">>> [ANNOTATION COMPLETE] → ./data/data_with_rewards_iter${num_iter}_${type}.json"
