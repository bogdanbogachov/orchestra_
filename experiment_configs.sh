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

# Helper function to add experiments with number range
add_experiments() {
    local base_name=$1
    local eval_head=$2
    local custom=$3
    local pool=$4
    local fft=$5
    local d_inf=$6
    local c_inf=$7
    local start_num=$8
    local end_num=$9
    
    for i in $(seq $start_num $end_num); do
        EXPERIMENTS+=("${base_name}_${i} ${eval_head} ${custom} ${pool} ${fft} ${d_inf} ${c_inf}")
    done
}

add_experiments "35_l_default" "default_head" "False" "mean" "False" "True" "False" 1 2

add_experiments "35_l_custom_last" "custom_head" "True" "last" "False" "False" "True" 1 2
#
#add_experiments "35_l_custom_max" "custom_head" "True" "max" "False" "False" "True" 1 15
#
#add_experiments "35_l_custom_mean" "custom_head" "True" "mean" "False" "False" "True" 1 15
#
#add_experiments "35_l_custom_attention" "custom_head" "True" "attention" "False" "False" "True" 1 15
#
#add_experiments "35_l_custom_fft_last" "custom_head" "True" "last" "True" "False" "True" 1 15
#
#add_experiments "35_l_custom_fft_max" "custom_head" "True" "max" "True" "False" "True" 1 15
#
#add_experiments "35_l_custom_fft_mean" "custom_head" "True" "mean" "True" "False" "True" 1 15
#
#add_experiments "35_l_custom_fft_attention" "custom_head" "True" "attention" "True" "False" "True" 1 15
