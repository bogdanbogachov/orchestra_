import os
import glob
import torch
from transformers.trainer_utils import SaveStrategy, PREFIX_CHECKPOINT_DIR

from logger_config import logger


def patch_peft_adapter_only_save_load(trainer, model, training_args):
    """
    Overrides:
      - trainer._save
      - trainer._load_best_model
    Only apply for custom head models.
    """

    original_save = trainer._save

    def custom_save(output_dir=None, state_dict=None):
        if output_dir is None:
            output_dir = trainer.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        try:
            from peft import get_peft_model_state_dict
            import safetensors.torch

            active_adapter = getattr(model.base_model, "active_adapter", "default")
            if isinstance(active_adapter, list):
                active_adapter = active_adapter[0] if active_adapter else "default"

            peft_state_dict = get_peft_model_state_dict(model.base_model, adapter_name=active_adapter)

            prefixed = {f"base_model.model.{k}": v for k, v in peft_state_dict.items()}

            if training_args.save_safetensors:
                safetensors.torch.save_file(
                    prefixed,
                    os.path.join(output_dir, "adapter_model.safetensors"),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(prefixed, os.path.join(output_dir, "adapter_model.bin"))

            torch.save(model.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))

            if hasattr(model.base_model, "peft_config") and active_adapter in model.base_model.peft_config:
                model.base_model.peft_config[active_adapter].save_pretrained(output_dir)

        except Exception as e:
            logger.warning(f"Optimized save failed, falling back to save_pretrained: {e}")
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(output_dir, safe_serialization=training_args.save_safetensors)
            else:
                original_save(output_dir, state_dict=state_dict)
                return

        torch.save(trainer.args, os.path.join(output_dir, "training_args.bin"))
        if trainer.processing_class is not None:
            trainer.processing_class.save_pretrained(output_dir)
        elif (
            trainer.data_collator is not None
            and hasattr(trainer.data_collator, "tokenizer")
            and trainer.data_collator.tokenizer is not None
        ):
            trainer.data_collator.tokenizer.save_pretrained(output_dir)

    trainer._save = custom_save
    logger.info("✓ Patched Trainer._save for adapter-only checkpoints")

    original_load_best = trainer._load_best_model

    def custom_load_best_model():
        import safetensors.torch
        from peft import set_peft_model_state_dict

        best_dir = trainer.state.best_model_checkpoint
        if best_dir is None:
            logger.warning("No best model checkpoint found, skipping load")
            return

        logger.info(f"Loading best model from {best_dir} (score: {trainer.state.best_metric}).")

        adapter_st = os.path.join(best_dir, "adapter_model.safetensors")
        adapter_bin = os.path.join(best_dir, "adapter_model.bin")

        if os.path.exists(adapter_st):
            adapter_state = safetensors.torch.load_file(adapter_st)
        elif os.path.exists(adapter_bin):
            adapter_state = torch.load(adapter_bin, map_location="cpu")
        else:
            logger.warning(f"Adapter file not found in {best_dir}, trying original load method")
            original_load_best()
            return

        peft_state = {}
        for k, v in adapter_state.items():
            if k.startswith("base_model.model."):
                peft_state[k.replace("base_model.model.", "")] = v
        set_peft_model_state_dict(model.base_model, peft_state)
        logger.info("✓ Loaded PEFT adapters from best checkpoint")

        classifier_path = os.path.join(best_dir, "classifier.pt")
        if os.path.exists(classifier_path):
            device = next(model.classifier.parameters()).device
            classifier_state = torch.load(classifier_path, map_location=device)
            model.classifier.load_state_dict(classifier_state)
            logger.info("✓ Loaded classifier from best checkpoint")
        else:
            logger.warning(f"Classifier file not found in {best_dir}")

    trainer._load_best_model = custom_load_best_model
    logger.info("✓ Patched Trainer._load_best_model for adapter+classifier loading")

    return trainer


def patch_accuracy_in_checkpoint_names(trainer):
    """
    Overrides trainer._save_checkpoint to include eval_accuracy in checkpoint folder names.
    """
    original_save_checkpoint = trainer._save_checkpoint

    def custom_save_checkpoint(model, trial=None):
        eval_accuracy = None
        if hasattr(trainer.state, "log_history") and trainer.state.log_history:
            for entry in reversed(trainer.state.log_history):
                if "eval_accuracy" in entry:
                    eval_accuracy = entry["eval_accuracy"]
                    break

        if eval_accuracy is not None:
            accuracy_str = f"acc{eval_accuracy:.4f}".replace(".", "p")
            ckpt_folder = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}-{accuracy_str}"
        else:
            ckpt_folder = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}"

        run_dir = trainer._get_output_dir(trial=trial)
        out_dir = os.path.join(run_dir, ckpt_folder)

        try:
            trainer.save_model(out_dir, _internal_call=True)

            if trainer.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and trainer.state.best_global_step:
                best_acc = None
                for entry in trainer.state.log_history:
                    if entry.get("step") == trainer.state.best_global_step and "eval_accuracy" in entry:
                        best_acc = entry["eval_accuracy"]
                        break

                if best_acc is not None:
                    best_str = f"acc{best_acc:.4f}".replace(".", "p")
                    best_folder = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}-{best_str}"
                else:
                    best_folder = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}"

                best_dir = os.path.join(run_dir, best_folder)
                if os.path.exists(best_dir):
                    trainer.state.best_model_checkpoint = best_dir

            if not trainer.args.save_only_model:
                trainer._save_optimizer_and_scheduler(out_dir)
                trainer._save_scaler(out_dir)
                trainer._save_rng_state(out_dir)

            if trainer.args.should_save:
                from transformers.trainer import ExportableState
                for cb in [cb for cb in trainer.callback_handler.callbacks + [trainer.control] if isinstance(cb, ExportableState)]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(trainer.state.stateful_callbacks[cb_name], list):
                        trainer.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        trainer.state.stateful_callbacks[cb_name] = cb_state
                trainer.state.save_to_json(os.path.join(out_dir, "trainer_state.json"))

        except Exception as e:
            logger.warning(f"Custom checkpoint save failed, falling back to original: {e}")
            original_save_checkpoint(model, trial)

    trainer._save_checkpoint = custom_save_checkpoint
    logger.info("✓ Patched Trainer._save_checkpoint to include eval_accuracy in folder names")
    return trainer


def find_checkpoint_for_step(run_dir: str, step: int):
    pattern = os.path.join(run_dir, f"checkpoint-{step}*")
    matches = glob.glob(pattern)
    if not matches:
        return None
    # prefer acc-suffixed
    for m in matches:
        if "acc" in os.path.basename(m):
            return m
    return matches[0]
