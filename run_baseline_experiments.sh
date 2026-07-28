#!/bin/bash

# Submit lightweight baseline experiments.
#
# Usage:
#   bash run_baseline_experiments.sh baseline_experiment_configs.sh

CONFIG_FILE=${1:-baseline_experiment_configs.sh}
PREDEFINED_SEEDS=(42 123 456 789 1011 1213 1415 1617 1819 2021)

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

source "$CONFIG_FILE"

if [[ ${#BASELINE_EXPERIMENTS[@]} -eq 0 ]]; then
    echo "No baseline experiments found in $CONFIG_FILE" >&2
    exit 1
fi

get_seed_for_experiment() {
    local exp_name=$1
    if [[ "$exp_name" =~ _([0-9]+)$ ]]; then
        local per_config_exp_num=${BASH_REMATCH[1]}
        local seed_index=$(( (per_config_exp_num - 1) % ${#PREDEFINED_SEEDS[@]} ))
        echo "${PREDEFINED_SEEDS[$seed_index]}"
    else
        echo "${PREDEFINED_SEEDS[0]}"
    fi
}

for config in "${BASELINE_EXPERIMENTS[@]}"; do
    read -r exp_name baseline <<< "$config"
    seed=$(get_seed_for_experiment "$exp_name")

    echo "Submitting baseline: EXP=$exp_name BASELINE=$baseline SEED=$seed"
    EXP="$exp_name" BASELINE="$baseline" SEED="$seed" sbatch -J "$exp_name" baseline_job.sh
done
