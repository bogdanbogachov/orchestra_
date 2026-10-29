#!/bin/bash

#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-adml2021
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --array=0-2
#SBATCH --output=_err_out/rapid_pilot_%A_%a.out

TASKS=(site gis dba)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

module load python/3.11.5
module load gcc cuda/12.2
module load scipy-stack
module load gcc arrow

# Activate venv
source venv/bin/activate

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

RUN_DIR=rapid_pilot/runs/${TASK}
mkdir -p "${RUN_DIR}"

python -m rapid_pilot.train --task_dir rapid_pilot/data/${TASK} --out "${RUN_DIR}" && \
python -m rapid_pilot.analyze --task_dir rapid_pilot/data/${TASK} --run_dir "${RUN_DIR}"
