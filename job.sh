#!/bin/bash

#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-adml2021
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gpus=h100_80gb:1

module load python/3.11.5
module load gcc cuda/12.2
module load scipy-stack
module load gcc arrow

# Activate venv
source venv/bin/activate

# Run the Python script
python main.py --finetune=True --infer_default="${D_INF:-False}" --infer_custom="${C_INF:-False}" --evaluate=True
