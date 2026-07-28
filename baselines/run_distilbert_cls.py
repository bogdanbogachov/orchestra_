#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from commands.baselines.distilbert_cls import build_parser, run_distilbert_cls


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_distilbert_cls(
        model_name=args.model_name,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
