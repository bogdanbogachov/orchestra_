import os
from logger_config import logger
from commands.inference.custom import run_infer_custom
from commands.inference.default import run_infer_default
from .trainer_patches import find_checkpoint_for_step


def save_final_artifacts(model, tokenizer, trainer, output_dir: str, use_custom_head: bool, training_args):
    if use_custom_head:
        model.save_pretrained(output_dir, safe_serialization=training_args.save_safetensors)
        logger.info(f"✓ Saved custom model (PEFT adapters + classifier) to {output_dir}")
    else:
        trainer.save_model()
        logger.info(f"✓ Saved default head model with finetuned score layer to {output_dir}")

    tokenizer.save_pretrained(output_dir)
    logger.info(f"✓ Saved tokenizer to {output_dir}")


def run_milestone_inference(
    trainer,
    milestone_callback,
    output_dir: str,
    use_custom_head: bool,
    accuracy_threshold: float,
):
    threshold_percent = int(accuracy_threshold * 100)
    logger.info("=" * 100)
    logger.info(f"Running inference on test set for {threshold_percent}% milestone model")
    logger.info("=" * 100)

    checkpoint_info = milestone_callback.get_checkpoint_info()
    first_milestone_step = milestone_callback.get_first_milestone_checkpoint()

    if first_milestone_step is None:
        key = f"first_{threshold_percent}_percent"
        info = checkpoint_info.get(key)
        if info is not None:
            first_milestone_step = info.get("step")
            logger.info(f"Found {threshold_percent}% milestone at step {first_milestone_step} from checkpoint_info")

    if first_milestone_step is None:
        logger.info(f"{threshold_percent}% accuracy threshold not reached, skipping milestone model inference")
        return

    run_dir = trainer.args.output_dir
    milestone_dir = find_checkpoint_for_step(run_dir, first_milestone_step)

    if milestone_dir is None or not os.path.exists(milestone_dir):
        logger.warning(f"{threshold_percent}% milestone reached at step {first_milestone_step}, but checkpoint not found")
        return

    logger.info(f"Running inference on {threshold_percent}% milestone model (checkpoint: {milestone_dir})...")
    out_path = os.path.join(output_dir, f"test_predictions_{threshold_percent}percent.json")

    if use_custom_head:
        run_infer_custom(adapter_path=milestone_dir, output_path=out_path)
    else:
        run_infer_default(adapter_path=milestone_dir, output_path=out_path)

    logger.info(f"✓ Saved {threshold_percent}% milestone model predictions to {out_path}")
