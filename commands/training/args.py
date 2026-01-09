import torch
from transformers import TrainingArguments

from logger_config import logger


def compute_gradient_accumulation_steps(training_config: dict) -> int:
    per_device = training_config.get("per_device_train_batch_size", 4)
    effective = training_config.get("effective_batch_size", 48)
    steps = max(1, effective // per_device)

    logger.info(
        f"✓ Batch configuration: per_device={per_device}, "
        f"gradient_accumulation={steps}, effective_batch_size={per_device * steps}"
    )
    return steps


def build_training_arguments(
    training_config: dict,
    output_dir: str,
    seed,
    gradient_accumulation_steps: int,
) -> TrainingArguments:
    eval_steps = training_config.get("eval_steps", 25)
    save_steps = training_config.get("save_steps", None) or eval_steps

    per_device = training_config.get("per_device_train_batch_size", 4)

    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": training_config.get("num_train_epochs", 20),
        "per_device_train_batch_size": per_device,
        "per_device_eval_batch_size": training_config.get("per_device_eval_batch_size", per_device),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": training_config["learning_rate"],
        "lr_scheduler_type": training_config.get("lr_scheduler_type", "linear"),
        "warmup_ratio": training_config.get("warmup_ratio", 0.1),

        "logging_dir": f"{output_dir}/logs",
        "logging_steps": training_config.get("logging_steps", 10),
        "eval_steps": eval_steps,
        "eval_strategy": training_config.get("eval_strategy", "steps"),
        "save_strategy": training_config.get("save_strategy", "steps"),
        "save_steps": save_steps,
        "save_total_limit": training_config.get("save_total_limit", 3),
        "load_best_model_at_end": training_config.get("load_best_model_at_end", True),
        "metric_for_best_model": training_config.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": training_config.get("greater_is_better", False),

        "fp16": training_config.get("fp16", True) and torch.cuda.is_available(),
        "report_to": "none",

        # Reduce checkpoint size
        "save_only_model": True,
        "save_safetensors": True,
    }

    if seed is not None:
        kwargs["seed"] = seed

    return TrainingArguments(**kwargs)
