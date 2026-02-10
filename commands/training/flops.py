from logger_config import logger
from commands.utils.metrics import calculate_flops_for_transformer


def precompute_flops(trainer, metrics_callback):
    try:
        dl = trainer.get_train_dataloader()
        first = next(iter(dl))
        input_ids = first["input_ids"]
        attn = first.get("attention_mask")

        metrics_callback.flops_per_sample = calculate_flops_for_transformer(trainer.model, input_ids, attn)
        metrics_callback.flops_calculated = True
        logger.info(f"  Precomputed forward FLOPs per sample: {metrics_callback.flops_per_sample:,}")
    except Exception as e:
        logger.warning(f"  Could not precompute FLOPs: {e}")
