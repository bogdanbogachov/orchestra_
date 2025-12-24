#!/bin/bash

# Script to run multiple experiments sequentially
# Each job will wait for the previous one to start and run for a specified time before submitting the next
#
# Usage:
#   ./run_experiments.sh [config_file] [wait_minutes]
#
# Arguments:
#   config_file  - Optional: Path to a bash file containing EXPERIMENTS array (default: define in script)
#   wait_minutes - Optional: Minutes to wait after job starts (default: 1)
#
# Example:
#   ./run_experiments.sh experiment_configs.sh 1
#   ./run_experiments.sh 2  # Wait 2 minutes, use default config in script

# Default wait time (in minutes)
WAIT_MINUTES=${2:-1}
if [[ -n "$1" && "$1" =~ ^[0-9]+$ ]]; then
    # If first arg is a number, treat it as wait_minutes
    WAIT_MINUTES="$1"
    CONFIG_FILE=""
else
    CONFIG_FILE="$1"
fi

# Function to check if a job is running
is_job_running() {
    local job_id=$1
    local status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
    # Check for both "RUNNING" (full) and "R" (short) formats
    # Also handle case-insensitive matching
    status=$(echo "$status" | tr '[:lower:]' '[:upper:]')
    if [[ "$status" == "RUNNING" || "$status" == "R" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to get job start time (in seconds since epoch)
get_job_start_time() {
    local job_id=$1
    # Try to get start time from sacct first (more reliable)
    local start_time=$(sacct -j "$job_id" -n -o Start --format=Start --noheader 2>/dev/null | head -1 | tr -d ' ')
    
    if [[ -n "$start_time" && "$start_time" != "Unknown" && "$start_time" != "N/A" ]]; then
        # Try Linux date format first
        local epoch=$(date -d "$start_time" +%s 2>/dev/null)
        if [[ -n "$epoch" ]]; then
            echo "$epoch"
            return 0
        fi
        # Try macOS/BSD date format
        epoch=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$start_time" +%s 2>/dev/null)
        if [[ -n "$epoch" ]]; then
            echo "$epoch"
            return 0
        fi
        # Try with different format (if timezone included)
        epoch=$(date -d "${start_time%.*}" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "${start_time%.*}" +%s 2>/dev/null)
        if [[ -n "$epoch" ]]; then
            echo "$epoch"
            return 0
        fi
    fi
    
    # Fallback: try to get from squeue
    local start_time_str=$(squeue -j "$job_id" -h -o "%S" 2>/dev/null | tr -d ' ')
    if [[ -n "$start_time_str" && "$start_time_str" != "N/A" && "$start_time_str" != "Unknown" ]]; then
        local epoch=$(date -d "$start_time_str" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S" "$start_time_str" +%s 2>/dev/null)
        if [[ -n "$epoch" ]]; then
            echo "$epoch"
            return 0
        fi
    fi
    
    # If all else fails, return empty
    echo ""
    return 1
}

# Function to wait for job to start and run for specified minutes
wait_for_job_runtime() {
    local job_id=$1
    local wait_minutes=${2:-1}
    local wait_seconds=$((wait_minutes * 60))
    
    echo "Waiting for job $job_id to start..."
    
    # Wait for job to start (max 10 minutes)
    local max_wait=600
    local waited=0
    while ! is_job_running "$job_id"; do
        # Get current status for logging
        local current_status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null || echo "UNKNOWN")
        if [[ $((waited % 30)) -eq 0 ]]; then
            echo "  Job $job_id status: $current_status (waited ${waited}s / ${max_wait}s max)"
        fi
        sleep 5
        waited=$((waited + 5))
        if [[ $waited -ge $max_wait ]]; then
            echo "Warning: Job $job_id did not start within $max_wait seconds (current status: $current_status). Proceeding anyway..."
            return 1
        fi
    done
    
    echo "Job $job_id is now running. Waiting for it to run for $wait_minutes minute(s)..."
    
    # Get start time
    local start_time=$(get_job_start_time "$job_id")
    if [[ -z "$start_time" ]]; then
        echo "Warning: Could not get start time for job $job_id. Waiting fixed $wait_minutes minute(s)..."
        sleep "$wait_seconds"
        return 0
    fi
    
    # Wait until job has run for the specified time
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -ge $wait_seconds ]]; then
            echo "Job $job_id has been running for at least $wait_minutes minute(s). Proceeding to next job."
            return 0
        fi
        
        # Check if job is still running
        if ! is_job_running "$job_id"; then
            echo "Warning: Job $job_id is no longer running. Proceeding to next job."
            return 0
        fi
        
        sleep 10
    done
}

# Function to submit a job with given environment variables
submit_job() {
    local exp_name=$1
    local eval_head=$2
    local custom=$3
    local pool=$4
    local fft=$5
    local d_inf=$6
    local c_inf=$7
    
    # Export environment variables
    export EXP="$exp_name"
    export EVAL="$eval_head"
    export CUSTOM="$custom"
    export POOL="$pool"
    export FFT="$fft"
    export D_INF="$d_inf"
    export C_INF="$c_inf"
    
    echo "=========================================="
    echo "Submitting job with configuration:"
    echo "  EXP=$EXP"
    echo "  EVAL=$EVAL"
    echo "  CUSTOM=$CUSTOM"
    echo "  POOL=$POOL"
    echo "  FFT=$FFT"
    echo "  D_INF=$D_INF"
    echo "  C_INF=$C_INF"
    echo "=========================================="
    
    # Submit the job
    local output=$(sbatch --job-name="$EXP" --output="_err_out/${EXP}.out" --error="_err_out/${EXP}.err" job.sh 2>&1)
    local job_id=$(echo "$output" | grep -oP '\d+' | head -1)
    
    if [[ -z "$job_id" ]]; then
        echo "Error: Failed to submit job. Output: $output"
        return 1
    fi
    
    echo "Submitted job $job_id: $EXP"
    echo "$job_id"
}

# ============================================================================
# CONFIGURATION: Define your experiment configurations here
# ============================================================================
# Format: Each line represents one experiment with the following fields:
# EXP_NAME EVAL_HEAD CUSTOM POOL FFT D_INF C_INF
# 
# Option 1: Define experiments directly here (uncomment and modify):
# EXPERIMENTS=(
#     "exp1 custom_head True mean True False False"
#     "exp2 custom_head True max False False False"
#     "exp3 default_head False mean False False False"
# )

# Option 2: Source from external config file
if [[ -n "$CONFIG_FILE" && -f "$CONFIG_FILE" ]]; then
    echo "Loading experiment configurations from: $CONFIG_FILE"
    source "$CONFIG_FILE"
elif [[ -z "${EXPERIMENTS+x}" ]]; then
    # Default: define experiments here if not already set
    EXPERIMENTS=(
        # "exp1 custom_head True mean True False False"
        # "exp2 custom_head True max False False False"
        # Add more experiments here...
    )
fi

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Check if EXPERIMENTS array is empty
if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    echo "Error: No experiments configured!"
    echo "Please edit this script and add your experiment configurations to the EXPERIMENTS array."
    exit 1
fi

# Create output directory for job logs if it doesn't exist
mkdir -p _err_out

# Track the last submitted job ID
LAST_JOB_ID=""

# Process each experiment
for exp_config in "${EXPERIMENTS[@]}"; do
    # Parse configuration
    read -r exp_name eval_head custom pool fft d_inf c_inf <<< "$exp_config"
    
    # Submit the job
    job_id=$(submit_job "$exp_name" "$eval_head" "$custom" "$pool" "$fft" "$d_inf" "$c_inf")
    
    if [[ -z "$job_id" ]]; then
        echo "Failed to submit job for $exp_name. Skipping..."
        continue
    fi
    
    LAST_JOB_ID="$job_id"
    
    # Wait for this job to start and run for 1 minute (unless it's the last one)
    # Check if there are more experiments after this one
    local is_last=true
    local found_current=false
    for check_config in "${EXPERIMENTS[@]}"; do
        if [[ "$found_current" == true ]]; then
            is_last=false
            break
        fi
        if [[ "$check_config" == "$exp_config" ]]; then
            found_current=true
        fi
    done
    
    if [[ "$is_last" == false ]]; then
        wait_for_job_runtime "$job_id" "$WAIT_MINUTES"
    else
        echo "This was the last job. No need to wait."
    fi
done

echo "=========================================="
echo "All jobs submitted!"
echo "Last job ID: $LAST_JOB_ID"
echo "Wait time per job: $WAIT_MINUTES minute(s)"
echo "Monitor jobs with: squeue -u \$USER"
echo "Check job details with: sacct -j \$JOB_ID"
echo "=========================================="

