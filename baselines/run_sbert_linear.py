#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commands.baselines.sbert_linear import build_parser, run_sbert_linear


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_sbert_linear(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        classifier_max_iter=args.classifier_max_iter,
        classifier_c=args.classifier_c,
    )
