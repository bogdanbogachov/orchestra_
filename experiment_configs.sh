#!/bin/bash

# Experiment configurations for run_experiments.sh
# 
# Format: "EXP_NAME EVAL_HEAD CUSTOM POOL FFT D_INF C_INF"
# 
# Where:
#   EXP_NAME  - Unique experiment name (used for job name and output files)
#   EVAL_HEAD - "custom_head" or "default_head"
#   CUSTOM    - "True" or "False" (use custom head)
#   POOL      - Pooling strategy: "mean", "max", "attention", or "last"
#   FFT       - "True" or "False" (use FFT filtering)
#   D_INF     - "True" or "False" (run default inference)
#   C_INF     - "True" or "False" (run custom inference)

# Experiment configurations - 9 experiments

EXPERIMENTS=(
    "35_1_default_501 default_head False mean False True False"
    "35_1_default_502 default_head False mean False True False"
    "35_1_default_503 default_head False mean False True False"
    "35_1_default_504 default_head False mean False True False"
    "35_1_default_505 default_head False mean False True False"
    "35_1_custom_last_501 custom_head True last False False True"
    "35_1_custom_last_502 custom_head True last False False True"
    "35_1_custom_last_503 custom_head True last False False True"
    "35_1_custom_last_504 custom_head True last False False True"
    "35_1_custom_last_505 custom_head True last False False True"
    "35_1_custom_max_501 custom_head True max False False True"
    "35_1_custom_max_502 custom_head True max False False True"
    "35_1_custom_max_503 custom_head True max False False True"
    "35_1_custom_max_504 custom_head True max False False True"
    "35_1_custom_max_505 custom_head True max False False True"
    "35_l_custom_mean_501 custom_head True mean False False True"
    "35_l_custom_mean_502 custom_head True mean False False True"
    "35_l_custom_mean_503 custom_head True mean False False True"
    "35_l_custom_mean_504 custom_head True mean False False True"
    "35_l_custom_mean_505 custom_head True mean False False True"
    "35_l_custom_attention_501 custom_head True attention False False True"
    "35_l_custom_attention_502 custom_head True attention False False True"
    "35_l_custom_attention_503 custom_head True attention False False True"
    "35_l_custom_attention_504 custom_head True attention False False True"
    "35_l_custom_attention_505 custom_head True attention False False True"
    "35_l_custom_fft_last_501 custom_head True last True False True"
    "35_l_custom_fft_last_502 custom_head True last True False True"
    "35_l_custom_fft_last_503 custom_head True last True False True"
    "35_l_custom_fft_last_504 custom_head True last True False True"
    "35_l_custom_fft_last_505 custom_head True last True False True"
    "35_l_custom_fft_max_501 custom_head True max True False True"
    "35_l_custom_fft_max_502 custom_head True max True False True"
    "35_l_custom_fft_max_503 custom_head True max True False True"
    "35_l_custom_fft_max_504 custom_head True max True False True"
    "35_l_custom_fft_max_505 custom_head True max True False True"
    "35_l_custom_fft_mean_501 custom_head True mean True False True"
    "35_l_custom_fft_mean_502 custom_head True mean True False True"
    "35_l_custom_fft_mean_503 custom_head True mean True False True"
    "35_l_custom_fft_mean_504 custom_head True mean True False True"
    "35_l_custom_fft_mean_505 custom_head True mean True False True"
    "35_l_custom_fft_attention_501 custom_head True attention True False True"
    "35_l_custom_fft_attention_502 custom_head True attention True False True"
    "35_l_custom_fft_attention_503 custom_head True attention True False True"
    "35_l_custom_fft_attention_504 custom_head True attention True False True"
    "35_l_custom_fft_attention_505 custom_head True attention True False True"
)

