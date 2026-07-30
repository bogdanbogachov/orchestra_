#!/bin/bash

# Baseline experiment configurations.
#
# Format:
#   "EXP_NAME BASELINE"
#
# EXP_NAME follows the existing convention:
#   base_name_global_exp_num_per_config_exp_num

BASELINE_EXPERIMENTS=()

add_baseline_experiments() {
    local base_name=$1
    local baseline=$2
    local global_exp_num=$3
    local per_config_start=$4
    local per_config_end=$5

    for i in $(seq "$per_config_start" "$per_config_end"); do
        BASELINE_EXPERIMENTS+=("${base_name}_${global_exp_num}_${i} ${baseline}")
    done
}

# Edit global experiment numbers before submitting if needed.
#add_baseline_experiments "clinc_noise_sbert_linear" "sbert_linear" 76 1 10
#add_baseline_experiments "clinc_noise_distilbert_cls" "distilbert_cls" 76 1 10
add_baseline_experiments "bank_distilbert_attention" "distilbert_attention" 78 1 10
