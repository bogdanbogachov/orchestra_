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
#   DSTYLE    - "True" or "False" (use default head style: apply linear to all tokens first, then select last token)
#   D_INF     - "True" or "False" (run default inference)
#   C_INF     - "True" or "False" (run custom inference)

# Experiment configurations - 9 experiments

EXPERIMENTS=(
    "35_1_default_223 default_head False mean False False True False"
#    "35_1_default_602 default_head False mean False False True False"
#    "35_1_default_603 default_head False mean False False True False"
#    "35_1_default_604 default_head False mean False False True False"
#    "35_1_default_605 default_head False mean False False True False"
    "35_1_custom_last_223 custom_head True last False False False True"
#    "35_1_custom_last_602 custom_head True last False False True"
#    "35_1_custom_last_603 custom_head True last False False True"
#    "35_1_custom_last_604 custom_head True last False False True"
#    "35_1_custom_last_605 custom_head True last False False True"
#    "35_1_custom_max_601 custom_head True max False False True"
#    "35_1_custom_max_602 custom_head True max False False True"
#    "35_1_custom_max_603 custom_head True max False False True"
#    "35_1_custom_max_604 custom_head True max False False True"
#    "35_1_custom_max_605 custom_head True max False False True"
#    "35_l_custom_mean_601 custom_head True mean False False True"
#    "35_l_custom_mean_602 custom_head True mean False False True"
#    "35_l_custom_mean_603 custom_head True mean False False True"
#    "35_l_custom_mean_604 custom_head True mean False False True"
#    "35_l_custom_mean_605 custom_head True mean False False True"
#    "35_l_custom_attention_601 custom_head True attention False False True"
#    "35_l_custom_attention_602 custom_head True attention False False True"
#    "35_l_custom_attention_603 custom_head True attention False False True"
#    "35_l_custom_attention_604 custom_head True attention False False True"
#    "35_l_custom_attention_605 custom_head True attention False False True"
#    "35_l_custom_fft_last_601 custom_head True last True False True"
#    "35_l_custom_fft_last_602 custom_head True last True False True"
#    "35_l_custom_fft_last_603 custom_head True last True False True"
#    "35_l_custom_fft_last_604 custom_head True last True False True"
#    "35_l_custom_fft_last_605 custom_head True last True False True"
#    "35_l_custom_fft_max_601 custom_head True max True False True"
#    "35_l_custom_fft_max_602 custom_head True max True False True"
#    "35_l_custom_fft_max_603 custom_head True max True False True"
#    "35_l_custom_fft_max_604 custom_head True max True False True"
#    "35_l_custom_fft_max_605 custom_head True max True False True"
#    "35_l_custom_fft_mean_601 custom_head True mean True False True"
#    "35_l_custom_fft_mean_602 custom_head True mean True False True"
#    "35_l_custom_fft_mean_603 custom_head True mean True False True"
#    "35_l_custom_fft_mean_604 custom_head True mean True False True"
#    "35_l_custom_fft_mean_605 custom_head True mean True False True"
#    "35_l_custom_fft_attention_601 custom_head True attention True False True"
#    "35_l_custom_fft_attention_602 custom_head True attention True False True"
#    "35_l_custom_fft_attention_603 custom_head True attention True False True"
#    "35_l_custom_fft_attention_604 custom_head True attention True False True"
#    "35_l_custom_fft_attention_605 custom_head True attention True False True"
)
