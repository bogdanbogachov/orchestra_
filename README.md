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
   export D_INF=boolean # True | False
   export C_INF=boolean # True | False
   sbatch --job-name="$EXP" --output="${EXP}.out" --error="${EXP}.err" job.sh
   ```

**Important**: Environment variables override respective variables in `config.yaml`.
