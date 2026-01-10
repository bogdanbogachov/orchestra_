#!/bin/bash

# Experiment configurations for run_experiments.sh
# 
# Format: "EXP_NAME EVAL_HEAD CUSTOM POOL FFT DSTYLE D_INF C_INF"
# 
# Where:
#   EXP_NAME  - Unique experiment name (used for job name and output files)
#   EVAL_HEAD - "custom_head" or "default_head"
#   CUSTOM    - "True" or "False" (use custom head)
#   POOL      - Pooling strategy: "mean", "max", "attention", or "last"
#   FFT       - "True" or "False" (use FFT filtering)
#   D_INF     - "True" or "False" (run default inference)
#   C_INF     - "True" or "False" (run custom inference)

# Experiment configurations - 9 core experiments
# Using loops to generate repetitive entries

EXPERIMENTS=()

# Helper function to add experiments with two-level numbering
# Format: base_name_global_exp_num_per_config_exp_num
# Example: 36_l_default_1_7 (global exp 1, per-config exp 7)
add_experiments() {
    local base_name=$1
    local eval_head=$2
    local custom=$3
    local pool=$4
    local fft=$5
    local d_inf=$6
    local c_inf=$7
    local global_exp_num=$8      # Global experiment number (first number)
    local per_config_start=$9     # Start of per-config experiment numbers (second number)
    local per_config_end=${10}    # End of per-config experiment numbers (second number)
    
    for i in $(seq $per_config_start $per_config_end); do
        EXPERIMENTS+=("${base_name}_${global_exp_num}_${i} ${eval_head} ${custom} ${pool} ${fft} ${d_inf} ${c_inf}")
    done
}

# Format: add_experiments "base_name" "eval_head" "custom" "pool" "fft" "d_inf" "c_inf" global_exp_num per_config_start per_config_end
#add_experiments "35_l_default" "default_head" "False" "mean" "False" "True" "False" 8 1 3
add_experiments "35_l_custom_last" "custom_head" "True" "last" "False" "False" "True" 9 6 6
#add_experiments "35_l_custom_max" "custom_head" "True" "max" "False" "False" "True" 8 1 3
#add_experiments "35_l_custom_mean" "custom_head" "True" "mean" "False" "False" "True" 8 1 3
#add_experiments "35_l_custom_attention" "custom_head" "True" "attention" "False" "False" "True" 8 1 3
#add_experiments "35_l_custom_fft_last" "custom_head" "True" "last" "True" "False" "True" 8 1 3
#add_experiments "35_l_custom_fft_max" "custom_head" "True" "max" "True" "False" "True" 8 1 3
#add_experiments "35_l_custom_fft_mean" "custom_head" "True" "mean" "True" "False" "True" 8 1 3
#add_experiments "35_l_custom_fft_attention" "custom_head" "True" "attention" "True" "False" "True" 8 1 3
