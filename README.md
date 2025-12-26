# orchestra_

## 1. Overview

## 2. Set up

### 2.1. Environment requirements
Python 3.13 and higher (maybe lower, TBD).

Create a virtual environment and install dependencies:
```bash
virtualenv -p /usr/bin/python venv
source venv/bin/activate
pip install -r requirements.txt
```

All scripts expect a model at `downloaded_models/downloaded_3_2_1b`. You'll need to download a model there first or modify the `model_path`.

### 2.2. Download Llama 3.2 1B
```bash
# Create directory
mkdir -p downloaded_models/downloaded_3_2_1b

# Login to HuggingFace. Only needed once. To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
hf auth login

# Download using huggingface-cli (you may need to login first)
hf download meta-llama/Llama-3.2-1B --local-dir downloaded_models/downloaded_3_2_1b
```

## 3. Running Experiments on Compute Canada Clusters

### 3.1 Cluster Submission Process

For each experiment:

job.sh setup:
- Enter your email to receive notifications about your jobs
- Estimate time required for a specific job
- Choose appropriate resources (for llama 2 fine-tuning GPU partitioning is desired)

   ```bash
   export EXP=experiment_name                              
   export EVAL=eval_head_type # default_head | custom_head
   export CUSTOM=boolean_head_type # True | False
   export POOL=pooling_strategy # mean | max | attention | last
   export FFT=boolean # True | False
   export DSTYLE=boolean # True | False (use default head style: apply linear to all tokens first, then select last token)
   export D_INF=boolean # True | False
   export C_INF=boolean # True | False
   sbatch --job-name="$EXP" --output="_err_out/${EXP}.out" --error="_err_out/${EXP}.err" job.sh
   ```

**Important**: Environment variables override respective variables in `config.yaml`.

### 3.2 Running Multiple Experiments Sequentially

To run multiple experiments automatically, use the `run_experiments.sh` script. This script submits jobs sequentially, waiting for each job to start and run for a specified time before submitting the next one.

#### 3.2.1 Configuring Experiments

Edit `experiment_configs.sh` to define your experiments. Each experiment is defined on a single line with the following format:

```
"EXP_NAME EVAL_HEAD CUSTOM POOL FFT D_INF C_INF"
```

**Parameters:**
- `EXP_NAME` - Unique experiment name (used for job name and output files)
- `EVAL_HEAD` - `"custom_head"` or `"default_head"`
- `CUSTOM` - `"True"` or `"False"` (use custom head)
- `POOL` - Pooling strategy: `"mean"`, `"max"`, `"attention"`, or `"last"`
- `FFT` - `"True"` or `"False"` (use FFT filtering)
- `D_INF` - `"True"` or `"False"` (run default inference)
- `C_INF` - `"True"` or `"False"` (run custom inference)

**Example configuration:**
```bash
EXPERIMENTS=(
    "exp1 custom_head True mean True False False False"
    "exp2 custom_head True max False False False False"
    "exp3 custom_head True last False True False False"
    "exp4 default_head False mean False False False False"
)
```

#### 3.2.2 Running the Script

**Basic usage:**
```bash
./run_experiments.sh experiment_configs.sh 2
```

This will:
- Load experiments from `experiment_configs.sh`
- Submit each job sequentially
- Wait for each job to start (checks every 5 seconds)
- Wait 2 minutes after each job starts before submitting the next
- Save all output files to `_err_out/` directory

**Options:**
```bash
# Use default config in script (if defined)
./run_experiments.sh 2

# Use custom config file with 1 minute wait (default)
./run_experiments.sh experiment_configs.sh

# Run in background and save output to log file
nohup ./run_experiments.sh experiment_configs.sh 2 > run_experiments.log 2>&1 &
```

**Arguments:**
- First argument: Config file path (optional) or wait time in minutes
- Second argument: Wait time in minutes (default: 1)

#### 3.2.3 How It Works

1. **Job Submission**: The script submits the first job with the specified environment variables
2. **Wait for Start**: Waits for the job to start running (checks status every 5 seconds, logs every 30 seconds)
3. **Wait for Runtime**: Once running, waits for the specified time (e.g., 2 minutes) to ensure environment variables are captured
4. **Next Job**: Submits the next experiment with new environment variables
5. **Repeat**: Continues until all experiments are submitted

**Note**: The script runs on the login node (not submitted as a job itself). Use `screen` or `tmux` for long-running sessions, or run it in the background with `nohup`.

#### 3.2.4 Monitoring Jobs

While the script is running, monitor your jobs in another terminal:

```bash
# View all your jobs
squeue -u $USER

# View specific job details
squeue -j JOB_ID

# View job accounting information
sacct -j JOB_ID

# View job output (after completion)
cat _err_out/EXP_NAME.out
cat _err_out/EXP_NAME.err
```

**Important**: 
- The script automatically creates the `_err_out/` directory if it doesn't exist
- All job output and error files are saved to `_err_out/`
- The script handles job status checking and will proceed if a job completes quickly or fails
