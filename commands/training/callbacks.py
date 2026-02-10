from transformers import EarlyStoppingCallback

from commands.training.metrics_callback import TrainingMetricsCallback
from commands.training.accuracy_milestone_callback import AccuracyMilestoneCallback
from logger_config import logger


def build_callbacks(training_config: dict, output_dir: str):
    early_patience = training_config.get("early_stopping_patience", 3)
    early_threshold = training_config.get("early_stopping_threshold", 0.0)

    early = EarlyStoppingCallback(
        early_stopping_patience=early_patience,
        early_stopping_threshold=early_threshold,
    )
    logger.info(f"✓ Early stopping enabled with patience={early_patience} evaluation steps")

    metrics_cb = TrainingMetricsCallback(output_dir=output_dir)
    logger.info("✓ Training metrics tracking enabled (FLOPs and memory)")

    acc_threshold = training_config.get("accuracy_threshold", 0.95)
    milestone_cb = AccuracyMilestoneCallback(
        output_dir=output_dir,
        accuracy_threshold=acc_threshold,
        metrics_callback=metrics_cb,
    )
    logger.info(f"✓ Accuracy milestone tracking enabled ({acc_threshold*100:.1f}% threshold)")

    return [early, metrics_cb, milestone_cb], metrics_cb, milestone_cb, acc_threshold
