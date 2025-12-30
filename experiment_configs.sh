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

# Experiment configurations - 9 experiments

EXPERIMENTS=(
    "35_1_default_231 default_head False mean False True False"
    "35_1_default_232 default_head False mean False True False"
    "35_1_default_233 default_head False mean False True False"
    "35_1_default_234 default_head False mean False True False"
    "35_1_default_235 default_head False mean False True False"
    "35_1_custom_last_231 custom_head True last False False True"
    "35_1_custom_last_232 custom_head True last False False True"
    "35_1_custom_last_233 custom_head True last False False True"
    "35_1_custom_last_234 custom_head True last False False True"
    "35_1_custom_last_235 custom_head True last False False True"
    "35_1_custom_max_231 custom_head True max False False True"
    "35_1_custom_max_232 custom_head True max False False True"
    "35_1_custom_max_233 custom_head True max False False True"
    "35_1_custom_max_234 custom_head True max False False True"
    "35_1_custom_max_235 custom_head True max False False True"
    "35_l_custom_mean_231 custom_head True mean False False True"
    "35_l_custom_mean_232 custom_head True mean False False True"
    "35_l_custom_mean_233 custom_head True mean False False True"
    "35_l_custom_mean_234 custom_head True mean False False True"
    "35_l_custom_mean_235 custom_head True mean False False True"
    "35_l_custom_attention_231 custom_head True attention False False True"
    "35_l_custom_attention_232 custom_head True attention False False True"
    "35_l_custom_attention_233 custom_head True attention False False True"
    "35_l_custom_attention_234 custom_head True attention False False True"
    "35_l_custom_attention_235 custom_head True attention False False True"
    "35_l_custom_fft_last_231 custom_head True last True False True"
    "35_l_custom_fft_last_232 custom_head True last True False True"
    "35_l_custom_fft_last_233 custom_head True last True False True"
    "35_l_custom_fft_last_234 custom_head True last True False True"
    "35_l_custom_fft_last_235 custom_head True last True False True"
    "35_l_custom_fft_max_231 custom_head True max True False True"
    "35_l_custom_fft_max_232 custom_head True max True False True"
    "35_l_custom_fft_max_233 custom_head True max True False True"
    "35_l_custom_fft_max_234 custom_head True max True False True"
    "35_l_custom_fft_max_235 custom_head True max True False True"
    "35_l_custom_fft_mean_231 custom_head True mean True False True"
    "35_l_custom_fft_mean_232 custom_head True mean True False True"
    "35_l_custom_fft_mean_233 custom_head True mean True False True"
    "35_l_custom_fft_mean_234 custom_head True mean True False True"
    "35_l_custom_fft_mean_235 custom_head True mean True False True"
    "35_l_custom_fft_attention_231 custom_head True attention True False True"
    "35_l_custom_fft_attention_232 custom_head True attention True False True"
    "35_l_custom_fft_attention_233 custom_head True attention True False True"
    "35_l_custom_fft_attention_234 custom_head True attention True False True"
    "35_l_custom_fft_attention_235 custom_head True attention True False True"
)
