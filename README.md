# orchestra_

## 1. Overview

This repository implements an end-to-end text classification pipeline on Llama 3.2 1B, including data preprocessing, optional noise injection, fine-tuning with default or custom heads, evaluation, and aggregation of results (with efficiency and Green AI metrics).

## 2. Set up

### 2.1. Environment requirements
Python 3.10.12 (tested).

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

## 3. Experiment pipeline with `main.py`

This repository exposes all major steps of the experimental pipeline through `main.py`.
Each step can be run independently or combined in a single call via Boolean flags.

### 3.1. Core configuration (`config.yaml` and environment overrides)

- **Configuration file**: All defaults live in `config.yaml`.
  - **experiment**: Unique experiment identifier (also used in output paths).
  - **model**: Choice of head (`use_custom_head`), pooling strategy (`pooling_strategy`), FFT usage (`use_fft`), number of labels, dtype, and device map.
  - **paths**: Location of the base model, experiment directory, and data files:
    - `paths.data.training`: raw labeled dataset (JSON list of `{ "text": ..., "label": ... }`).
    - `paths.data.train` / `paths.data.test`: preprocessed train/test splits used by training and inference.
  - **training**: Hyperparameters (epochs, batch sizes, LR, scheduler, early stopping, fp16, etc.).
  - **data_processing**: Train–test split parameters (`test_size`, `random_state`, `stratify`).
  - **evaluation**: Which head to evaluate by default (`default_head` or `custom_head`).

- **Environment variable overrides** (handled in `config.py`):
  - `EXP` → overrides `experiment`.
  - `EVAL` → overrides `evaluation.head`.
  - `CUSTOM` → overrides `model.use_custom_head` (`true` / `false`).
  - `POOL` → overrides `model.pooling_strategy` (e.g., `mean`, `max`, `attention`, `last`).
  - `FFT` → overrides `model.use_fft` (`true` / `false`).

These overrides are what the cluster scripts use to sweep across configurations while keeping a single `config.yaml` in version control.

### 3.2. Step 1 – Data preprocessing (optional)

**Goal**: When using a manually crafted dataset (not Banking77/CLINC150 from HuggingFace), create a train/test split from a single raw labeled file.

- Input (only for manual datasets): `paths.data.training` (default: `training_data.json`).
- Outputs:
  - `paths.data.train` (default: `train_data.json`)
  - `paths.data.test` (default: `test_data.json`)

Run:

```bash
python main.py --preprocess_data true
```

This step is **optional** and only needed if you start from a single raw JSON file rather than predefined HuggingFace datasets.
It will:
- Stratify labels if `data_processing.stratify: true`.
- Use `data_processing.test_size` and `data_processing.random_state`.
- Log label distributions for original/train/test splits.

### 3.3. Step 2 – Optional noise injection

**Goal**: Add realistic textual noise to both train and test splits to study robustness.

- Inputs:
  - `paths.data.train`
  - `paths.data.test`
- Output: The same files are modified **in place** with noisy versions of the texts.
- `--noise_type` controls the noise template family:
  - `banking77`
  - `clinc150`

Run (example for Banking77-style noise):

```bash
python main.py --add_noise true --noise_type banking77
```

Notes:
- Different templates are used for train vs. test to induce a subtle distribution shift.
- Typos are applied only to injected noise fragments, not to the original utterance.

To replicate a particular noisy condition, record both:
- The `noise_type`, and
- The exact commit of this repository (noise templates live in `commands/data_processing/add_noise.py`).

### 3.4. Step 3 – Fine-tuning

**Goal**: Fine-tune either the default classification head or a custom LLaMA-based head (with optional FFT pooling and LoRA adapters).

Run:

```bash
python main.py --finetune true
```

Key details:
- Uses `paths.data.train` as input.
- Reads all model/training settings from `config.yaml` (plus environment overrides).
- Uses LoRA settings from `lora` in `config.yaml`.
- Supports optional reproducibility via the `training.seed` field or `SEED` environment variable.
- Internally constructs an experiment directory under `paths.experiments`, typically:
  - `experiments/<experiment_name>/<head_type>` for simple names, or
  - `experiments/<global_exp_num>/<base_name>_<per_config_exp_num>/<head_type>` for the Compute Canada sweeps (see cluster section below).

Outputs (per head type):
- Fine-tuned model weights (either full head or PEFT adapters + classifier).
- Tokenizer.
- Training logs and training metrics (including FLOPs, memory, and Green AI metrics).

### 3.5. Step 4 – Inference (default vs. custom head)

**Goal**: Run dataset-level inference and record predictions + efficiency metrics.

You can enable one or both heads in a single call.

Default-head inference:

```bash
python main.py --infer_default true
```

Custom-head inference:

```bash
python main.py --infer_custom true
```

Behavior:
- Input dataset: `paths.data.test` by default.
- Output files (under the experiment directory for each head):
  - `default_head/test_predictions.json`
  - `custom_head/test_predictions.json`
- Each JSON contains:
  - `pred` (argmax class),
  - `probs` (class probabilities),
  - per-example latency (ms),
  - aggregated FLOPs, peak memory, and energy/carbon metrics for the entire run.

These prediction files are the only inputs required for the evaluation step.

### 3.6. Step 5 – Evaluation

**Goal**: Compute accuracy, precision, recall, F1, and latency statistics for one head.

Run:

```bash
python main.py --evaluate true
```

Head selection:
- By default, `evaluation.head` in `config.yaml` (or `EVAL` env var) chooses:
  - `"default_head"` or `"custom_head"`.

Evaluation reads:
- Ground-truth labels from `paths.data.test`.
- Predictions from the corresponding `test_predictions.json` (for the chosen head).

Outputs:
- `experiments/.../<head>/evaluation_results.json` containing:
  - Standard classification metrics (accuracy, precision, recall, F1).
  - Latency statistics (mean ± std, excluding the first “warmup” example).
  - Training and inference FLOPs and memory statistics (if available).
  - Energy and carbon metrics for training and inference (Green AI metrics).

### 3.7. Step 6 – Aggregating results across runs

**Goal**: Aggregate metrics across multiple runs of the same configuration and produce tables + charts suitable for publication.

Run for a specific global experiment ID:

```bash
python main.py --aggregate_results true --global_exp_num 35
```

Where:
- `global_exp_num` corresponds to the numeric subdirectory under `paths.experiments` used by your sweep (e.g., `experiments/35/...`).

Aggregator behavior:
- Collects all `evaluation_results.json` files under `experiments/<global_exp_num>/**`.
- Groups runs by experiment “base name” (e.g., pooling/FFT/head configuration).
- Computes mean ± standard deviation for all key metrics across runs.

Outputs (per `global_exp_num`):
- `aggregations/<global_exp_num>/aggregated_metrics.csv`
- `aggregations/<global_exp_num>/aggregated_metrics.tex` (LaTeX table).
- `aggregations/<global_exp_num>/charts/*.png` (per-metric line plots with error bars).
- `aggregations/<global_exp_num>/detailed_statistics.json` (full statistics for all metrics).

### 3.8. Step 7 – F1 bar chart across aggregations

**Goal**: Compare F1 scores across different aggregations (e.g., different global experiment IDs) in a single bar chart with error bars.

Prerequisite:
- For each aggregation ID you want to plot, you must already have an `aggregated_metrics.csv` in `aggregations/<id>/` from the previous step.

Run, for example:

```bash
python main.py \
  --charts true \
  --aggregation_nums 35 36 \
  --aggregation_names "Baseline" "Noisy (Banking77)"
```

Flags:
- `--aggregation_nums`: list of aggregation IDs (integers) to include.
- `--aggregation_names`: optional human-readable labels (must match in length).

Output:
- `charts/f1_scores_comparison.png` – a publication-quality bar chart comparing F1 means ± std across all specified aggregations and experiment configurations.

This step is typically the last one you run when preparing figures for a paper or report.

## 4. Running Experiments on Compute Canada Clusters

### 4.1 Cluster Submission Process

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
   export D_INF=boolean # True | False
   export C_INF=boolean # True | False
   sbatch --job-name="$EXP" --output="_err_out/${EXP}.out" --error="_err_out/${EXP}.err" job.sh
   ```

**Important**: Environment variables override respective variables in `config.yaml`.

### 4.2 Running Multiple Experiments Sequentially

To run multiple experiments automatically, use the `run_experiments.sh` script. This script submits jobs sequentially, waiting for each job to start and run for a specified time before submitting the next one.

#### 4.2.1 Configuring Experiments

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

#### 4.2.2 Running the Script

**Basic usage:**
```bash
./run_experiments.sh experiment_configs.sh 2
```

This will:
- Load experiments from `experiment_configs.sh`
- Submit each job sequentially
- Wait for each job to start (checks every 5 seconds)
- Wait x minutes after each job starts before submitting the next
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

#### 4.2.3 How It Works

1. **Job Submission**: The script submits the first job with the specified environment variables
2. **Wait for Start**: Waits for the job to start running (checks status every 5 seconds, logs every 30 seconds)
3. **Wait for Runtime**: Once running, waits for the specified time (e.g., 2 minutes) to ensure environment variables are captured
4. **Next Job**: Submits the next experiment with new environment variables
5. **Repeat**: Continues until all experiments are submitted

**Note**: The script runs on the login node (not submitted as a job itself). Use `screen` or `tmux` for long-running sessions, or run it in the background with `nohup`.

#### 4.2.4 Monitoring Jobs

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
