#!/bin/bash

#SBATCH --mail-user=bogdan.bogachov@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-adml2021
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=h100_80gb:1

module load python/3.11.5
module load gcc cuda/12.2
module load scipy-stack
module load gcc arrow

source venv/bin/activate

case "${BASELINE}" in
  sbert_linear)
    python baselines/run_sbert_linear.py \
      --model-name "downloaded_models/all-MiniLM-L6-v2" \
      --batch-size "${BASELINE_BATCH_SIZE:-64}" \
      --max-length "${BASELINE_MAX_LENGTH:-128}"
    ;;
  distilbert_cls)
    python baselines/run_distilbert_cls.py \
      --model-name "downloaded_models/distilbert-base-uncased" \
      --batch-size "${BASELINE_BATCH_SIZE:-32}" \
      --max-length "${BASELINE_MAX_LENGTH:-128}" \
      --num-train-epochs "${DISTILBERT_EPOCHS:-10}" \
      --learning-rate "${DISTILBERT_LR:-0.00002}"
    ;;
  distilbert_attention)
    python baselines/run_distilbert_attention.py \
      --model-name "downloaded_models/distilbert-base-uncased" \
      --batch-size "${BASELINE_BATCH_SIZE:-32}" \
      --max-length "${BASELINE_MAX_LENGTH:-128}" \
      --num-train-epochs "${DISTILBERT_EPOCHS:-10}" \
      --learning-rate "${DISTILBERT_LR:-0.00002}"
    ;;
  *)
    echo "Unknown BASELINE='${BASELINE}'. Use BASELINE=sbert_linear, distilbert_cls, or distilbert_attention." >&2
    exit 2
    ;;
esac
