import torch
import json
import os
from typing import Optional
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from transformers.trainer_utils import SaveStrategy, PREFIX_CHECKPOINT_DIR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from config import CONFIG
from commands.training.dataset import ClassificationDataset
from commands.training.model import load_model_and_tokenizer, setup_lora
from commands.training.seed_utils import set_seed
from commands.training.metrics_callback import TrainingMetricsCallback
from commands.training.accuracy_milestone_callback import AccuracyMilestoneCallback
from commands.utils.metrics import calculate_flops_for_transformer
from logger_config import logger


def load_data(data_path: str):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    return texts, labels


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}


def extract_global_experiment_number(experiment_name: str) -> Optional[int]:
    """
    Extract global experiment number from experiment name.
    
    Format: base_name_global_exp_num_per_config_exp_num
    Example: "35_l_default_8_1" -> 8
    Example: "35_L_custom_last_8_1" -> 8
    """
    import re
    # Pattern: base_name_global_exp_num_per_config_exp_num
    # Extract the second-to-last number (global experiment number)
    # Match: any characters, then _number_number at the end
    match = re.match(r'^(.+)_(\d+)_(\d+)$', experiment_name)
    if match:
        return int(match.group(2))  # global_exp_num is the second-to-last number
    
    # Try alternative pattern in case format is slightly different
    # Look for pattern: ..._number_number at the end
    parts = experiment_name.split('_')
    if len(parts) >= 3:
        try:
            # Try to parse last two parts as numbers
            last_num = int(parts[-1])
            second_last_num = int(parts[-2])
            # If both are numbers, return the second-to-last
            return second_last_num
        except (ValueError, IndexError):
            pass
    
    return None


def run_finetune():
    training_config = CONFIG['training']
    data_config = CONFIG['data_processing']
    paths_config = CONFIG['paths']
    experiment_name = CONFIG.get('experiment', 'orchestra')
    experiments_dir = paths_config['experiments']
    
    # Extract global experiment number and create nested directory structure
    # ALWAYS use nested structure when global number can be extracted
    global_exp_num = extract_global_experiment_number(experiment_name)
    if global_exp_num is not None:
        # New structure: experiments/global_exp_num/experiment_name/head_type
        output_base = os.path.join(experiments_dir, str(global_exp_num), experiment_name)
        logger.info(f"Using nested directory structure: experiments/{global_exp_num}/{experiment_name}")
    else:
        # Fallback to old structure ONLY if global number truly cannot be extracted
        # This should be rare - most experiment names follow the pattern
        output_base = os.path.join(experiments_dir, experiment_name)
        logger.warning(f"Could not extract global experiment number from '{experiment_name}', using flat structure")
        logger.warning(f"Expected format: base_name_global_exp_num_per_config_exp_num (e.g., '35_l_default_8_1')")
    
    # Use random seed if specified, otherwise use None for non-deterministic training
    seed = training_config.get('seed', None)
    if seed is not None:
        set_seed(seed)
        logger.info(f"✓ Set random seed to {seed} for reproducible training")
    else:
        logger.info("✓ Using non-deterministic training (no fixed seed)")
    
    model, tokenizer, use_custom_head = load_model_and_tokenizer()
    model = setup_lora(model, use_custom_head)
    
    head_type = "custom_head" if use_custom_head else "default_head"
    output_dir = os.path.join(output_base, head_type)
    
    # Ensure we're using nested structure - never create in root if we have global_exp_num
    if global_exp_num is not None:
        # Verify the path includes the global experiment number
        expected_global_exp_path = os.path.join(experiments_dir, str(global_exp_num))
        if not output_dir.startswith(expected_global_exp_path):
            raise ValueError(f"Output directory {output_dir} does not include global experiment number {global_exp_num}. "
                           f"Expected path to start with {expected_global_exp_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    max_length = training_config['max_length']
    
    logger.info(f"Training {head_type} - output directory: {output_dir}")
    
    # Verify output_dir is correct (should be nested when global_exp_num exists)
    if global_exp_num is not None:
        actual_path = os.path.abspath(output_dir)
        expected_components = [experiments_dir, str(global_exp_num), experiment_name, head_type]
        expected_path = os.path.abspath(os.path.join(*expected_components))
        if actual_path != expected_path:
            logger.error(f"Path mismatch! Actual: {actual_path}, Expected: {expected_path}")
            raise ValueError(f"Output directory path verification failed. Check directory structure.")
    
    all_texts, all_labels = load_data(paths_config['data']['train'])
    # Use random state only if seed is set, otherwise use None for random splits
    split_random_state = data_config.get('random_state') if seed is not None else None
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels, 
        test_size=data_config['test_size'],
        random_state=split_random_state,
        stratify=all_labels if data_config['stratify'] else None
    )
    
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_length=max_length)

    logger.info(f"✓ Split data: {len(train_texts)} train, {len(val_texts)} validation ({data_config['test_size']*100:.1f}%)")

    # Calculate gradient accumulation steps to achieve effective batch size of 16
    per_device_batch_size = training_config.get('per_device_train_batch_size', 4)
    effective_batch_size = training_config.get('effective_batch_size', 48)
    gradient_accumulation_steps = max(1, effective_batch_size // per_device_batch_size)
    
    logger.info(f"✓ Batch configuration: per_device={per_device_batch_size}, "
                f"gradient_accumulation={gradient_accumulation_steps}, "
                f"effective_batch_size={per_device_batch_size * gradient_accumulation_steps}")
    
    # Set save_steps to match eval_steps if not specified
    eval_steps = training_config.get('eval_steps', 25)
    save_steps = training_config.get('save_steps', None)
    if save_steps is None:
        save_steps = eval_steps
    
    # Build training arguments - conditionally include seed if not None
    training_kwargs = {
        'output_dir': output_dir,
        'num_train_epochs': training_config.get('num_train_epochs', 20),  # Max epochs for early stopping
        'per_device_train_batch_size': per_device_batch_size,
        'per_device_eval_batch_size': training_config.get('per_device_eval_batch_size', per_device_batch_size),
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'learning_rate': training_config['learning_rate'],
        # Learning rate schedule with warmup
        'lr_scheduler_type': training_config.get('lr_scheduler_type', 'linear'),
        'warmup_ratio': training_config.get('warmup_ratio', 0.1),  # 10% of training steps for warmup
        # Early stopping configuration
        'logging_dir': f"{output_dir}/logs",
        'logging_steps': training_config.get('logging_steps', 10),
        'eval_steps': eval_steps,
        'eval_strategy': training_config.get('eval_strategy', 'steps'),
        'save_strategy': training_config.get('save_strategy', 'steps'),
        'save_steps': save_steps,
        'save_total_limit': training_config.get('save_total_limit', 3),  # Keep only best 3 checkpoints
        'load_best_model_at_end': training_config.get('load_best_model_at_end', True),
        'metric_for_best_model': training_config.get('metric_for_best_model', 'eval_loss'),
        'greater_is_better': training_config.get('greater_is_better', False),
        'fp16': training_config.get('fp16', True) and torch.cuda.is_available(),
        'report_to': 'none',
        # Reduce checkpoint size: only save model weights (skip optimizer/scheduler/RNG states)
        'save_only_model': True,
        'save_safetensors': True,  # Use safetensors format (more efficient and secure)
    }
    
    # Only include seed if it's not None
    if seed is not None:
        training_kwargs['seed'] = seed
    
    training_args = TrainingArguments(**training_kwargs)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Early stopping callback with industry-standard patience
    early_stopping_patience = training_config.get('early_stopping_patience', 3)
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=training_config.get('early_stopping_threshold', 0.0)
    )
    logger.info(f"✓ Early stopping enabled with patience={early_stopping_patience} evaluation steps")

    # Training metrics callback for FLOPs and memory tracking
    metrics_callback = TrainingMetricsCallback(output_dir=output_dir)
    logger.info("✓ Training metrics tracking enabled (FLOPs and memory)")
    
    # Accuracy milestone callback to track accuracy milestone
    accuracy_threshold = training_config.get('accuracy_threshold', 0.95)
    accuracy_milestone_callback = AccuracyMilestoneCallback(
        output_dir=output_dir,
        accuracy_threshold=accuracy_threshold,
        metrics_callback=metrics_callback
    )
    logger.info(f"✓ Accuracy milestone tracking enabled ({accuracy_threshold*100:.1f}% threshold)")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback, metrics_callback, accuracy_milestone_callback],
    )

    # Override Trainer's _save method for custom head models with optimized direct save
    # This ensures checkpoints only save PEFT adapters, not the full 5GB base model
    # Optimized to avoid save_pretrained overhead for faster checkpoint saving
    if use_custom_head:
        original_save = trainer._save
        
        def custom_save(output_dir=None, state_dict=None):
            """Optimized save that directly extracts PEFT adapters + classifier state dict."""
            if output_dir is None:
                output_dir = trainer.args.output_dir
            
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                # Directly get PEFT adapter state dict (faster than full save_pretrained)
                from peft import get_peft_model_state_dict
                import safetensors.torch
                
                # Get only PEFT adapter weights (not full base model)
                active_adapter = getattr(model.base_model, 'active_adapter', "default")
                if isinstance(active_adapter, list):
                    active_adapter = active_adapter[0] if active_adapter else "default"
                
                peft_state_dict = get_peft_model_state_dict(
                    model.base_model,
                    adapter_name=active_adapter
                )
                
                # Save PEFT adapter state dict directly (faster than save_pretrained)
                # Add base_model.model prefix for PEFT format compatibility
                prefixed_peft_dict = {}
                for key, value in peft_state_dict.items():
                    prefixed_peft_dict[f"base_model.model.{key}"] = value
                
                if training_args.save_safetensors:
                    safetensors.torch.save_file(
                        prefixed_peft_dict,
                        os.path.join(output_dir, "adapter_model.safetensors"),
                        metadata={"format": "pt"}
                    )
                else:
                    torch.save(prefixed_peft_dict, os.path.join(output_dir, "adapter_model.bin"))
                
                # Save classifier separately as classifier.pt (matches inference loading)
                classifier_path = os.path.join(output_dir, "classifier.pt")
                torch.save(model.classifier.state_dict(), classifier_path)
                
                # Save PEFT config (minimal overhead)
                if hasattr(model.base_model, 'peft_config') and active_adapter in model.base_model.peft_config:
                    peft_config = model.base_model.peft_config[active_adapter]
                    peft_config.save_pretrained(output_dir)
                
            except Exception as e:
                logger.warning(f"Optimized save failed, falling back to save_pretrained: {e}")
                # Fallback to model.save_pretrained if direct extraction fails
                if hasattr(model, 'save_pretrained'):
                    model.save_pretrained(
                        output_dir,
                        safe_serialization=training_args.save_safetensors
                    )
                else:
                    original_save(output_dir, state_dict=state_dict)
                    return
            
            # Save training args and tokenizer (same as original)
            torch.save(trainer.args, os.path.join(output_dir, "training_args.bin"))
            if trainer.processing_class is not None:
                trainer.processing_class.save_pretrained(output_dir)
            elif (trainer.data_collator is not None and 
                  hasattr(trainer.data_collator, "tokenizer") and 
                  trainer.data_collator.tokenizer is not None):
                trainer.data_collator.tokenizer.save_pretrained(output_dir)
        
        trainer._save = custom_save
        logger.info("✓ Overrode Trainer save method with optimized PEFT adapter-only checkpoints")
        
        # Also override _load_best_model to handle adapter + classifier loading
        original_load_best = trainer._load_best_model
        
        def custom_load_best_model():
            """Custom load that handles PEFT adapters + classifier for custom head models."""
            import safetensors.torch
            
            best_checkpoint_dir = trainer.state.best_model_checkpoint
            if best_checkpoint_dir is None:
                logger.warning("No best model checkpoint found, skipping load")
                return
            
            logger.info(f"Loading best model from {best_checkpoint_dir} (score: {trainer.state.best_metric}).")
            
            # Load PEFT adapters
            adapter_path = os.path.join(best_checkpoint_dir, "adapter_model.safetensors")
            adapter_bin_path = os.path.join(best_checkpoint_dir, "adapter_model.bin")
            
            if os.path.exists(adapter_path):
                # Load adapter state dict
                adapter_state_dict = safetensors.torch.load_file(adapter_path)
                # Remove base_model.model prefix for loading
                peft_state_dict = {}
                for key, value in adapter_state_dict.items():
                    if key.startswith("base_model.model."):
                        peft_state_dict[key.replace("base_model.model.", "")] = value
                
                # Load into PEFT model
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(model.base_model, peft_state_dict)
                logger.info("✓ Loaded PEFT adapters from best checkpoint")
            elif os.path.exists(adapter_bin_path):
                adapter_state_dict = torch.load(adapter_bin_path, map_location="cpu")
                peft_state_dict = {}
                for key, value in adapter_state_dict.items():
                    if key.startswith("base_model.model."):
                        peft_state_dict[key.replace("base_model.model.", "")] = value
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(model.base_model, peft_state_dict)
                logger.info("✓ Loaded PEFT adapters from best checkpoint")
            else:
                logger.warning(f"Adapter file not found in {best_checkpoint_dir}, trying original load method")
                original_load_best()
                return
            
            # Load classifier
            classifier_path = os.path.join(best_checkpoint_dir, "classifier.pt")
            if os.path.exists(classifier_path):
                # Get device from classifier
                device = next(model.classifier.parameters()).device
                classifier_state = torch.load(classifier_path, map_location=device)
                model.classifier.load_state_dict(classifier_state)
                logger.info("✓ Loaded classifier from best checkpoint")
            else:
                logger.warning(f"Classifier file not found in {best_checkpoint_dir}")
        
        trainer._load_best_model = custom_load_best_model
        logger.info("✓ Overrode Trainer load_best_model method for custom head models")
    
    # Override _save_checkpoint to include eval_accuracy in checkpoint folder names
    original_save_checkpoint = trainer._save_checkpoint
    
    def custom_save_checkpoint(model, trial=None):
        """Save checkpoint with eval_accuracy in folder name."""
        # Get latest eval_accuracy from state if available
        eval_accuracy = None
        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            # Find the most recent eval_accuracy from log history
            for log_entry in reversed(trainer.state.log_history):
                if 'eval_accuracy' in log_entry:
                    eval_accuracy = log_entry['eval_accuracy']
                    break
        
        # Build checkpoint folder name with accuracy
        if eval_accuracy is not None:
            accuracy_str = f"acc{eval_accuracy:.4f}".replace('.', 'p')
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}-{accuracy_str}"
        else:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.global_step}"
        
        run_dir = trainer._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        
        try:
            # Save model
            trainer.save_model(output_dir, _internal_call=True)
            
            # Handle best checkpoint tracking
            if trainer.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and trainer.state.best_global_step:
                # Get accuracy for best checkpoint too
                best_eval_accuracy = None
                for log_entry in trainer.state.log_history:
                    if log_entry.get('step') == trainer.state.best_global_step and 'eval_accuracy' in log_entry:
                        best_eval_accuracy = log_entry['eval_accuracy']
                        break
                
                if best_eval_accuracy is not None:
                    best_accuracy_str = f"acc{best_eval_accuracy:.4f}".replace('.', 'p')
                    best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}-{best_accuracy_str}"
                else:
                    best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{trainer.state.best_global_step}"
                
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)
                if os.path.exists(best_checkpoint_dir):
                    trainer.state.best_model_checkpoint = best_checkpoint_dir
            
            # Save optimizer, scheduler, RNG if needed
            if not trainer.args.save_only_model:
                trainer._save_optimizer_and_scheduler(output_dir)
                trainer._save_scaler(output_dir)
                trainer._save_rng_state(output_dir)
            
            # Save trainer state
            if trainer.args.should_save:
                from transformers.trainer import ExportableState
                for cb in [cb for cb in trainer.callback_handler.callbacks + [trainer.control] if isinstance(cb, ExportableState)]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(trainer.state.stateful_callbacks[cb_name], list):
                        trainer.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        trainer.state.stateful_callbacks[cb_name] = cb_state
                
                trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
            
        except Exception as e:
            logger.warning(f"Custom checkpoint save failed, falling back to original: {e}")
            original_save_checkpoint(model, trial)
    
    trainer._save_checkpoint = custom_save_checkpoint
    logger.info("✓ Overrode Trainer _save_checkpoint to include eval_accuracy in folder names")

    # Precompute FLOPs once from first train batch (reliable)
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

    trainer.train()

    if use_custom_head:
        # Use the custom save_pretrained method which saves only PEFT adapters + classifier
        model.save_pretrained(output_dir, safe_serialization=training_args.save_safetensors)
        logger.info(f"✓ Saved custom model (PEFT adapters + classifier) to {output_dir}")
    else:
        trainer.save_model()
        logger.info(f"✓ Saved default head model with finetuned score layer to {output_dir}")
    
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✓ Saved tokenizer to {output_dir}")
    
    # Run inference on both best model and milestone model
    threshold_percent = int(accuracy_threshold * 100)
    logger.info("=" * 100)
    logger.info(f"Running inference on test set for both best model and {threshold_percent}% milestone model")
    logger.info("=" * 100)
    
    # Get checkpoint info
    checkpoint_info = accuracy_milestone_callback.get_checkpoint_info()
    first_milestone_step = accuracy_milestone_callback.get_first_95_percent_checkpoint()
    
    # Find best model checkpoint
    best_checkpoint_dir = trainer.state.best_model_checkpoint if hasattr(trainer.state, 'best_model_checkpoint') else None
    
    # Find milestone checkpoint
    milestone_checkpoint_dir = None
    if first_milestone_step is not None:
        # Find checkpoint folder that matches the step (may have accuracy in name)
        run_dir = trainer._get_output_dir()
        import glob
        # Search for checkpoint folders matching the step (with or without accuracy suffix)
        pattern = os.path.join(run_dir, f"checkpoint-{first_milestone_step}*")
        matches = glob.glob(pattern)
        if matches:
            # Prefer checkpoint with accuracy in name, otherwise use first match
            milestone_checkpoint_dir = matches[0]
            for match in matches:
                if "acc" in os.path.basename(match):
                    milestone_checkpoint_dir = match
                    break
            logger.info(f"Found {threshold_percent}% milestone checkpoint: {milestone_checkpoint_dir}")
        else:
            logger.warning(f"Could not find checkpoint folder for step {first_milestone_step} in {run_dir}")
    
    # Run inference on best model (default behavior)
    logger.info("Running inference on best model...")
    if use_custom_head:
        from commands.inference.custom import run_infer_custom
        # For custom head, checkpoint contains adapter_model.safetensors and classifier.pt
        # Use the checkpoint directory directly
        best_adapter_path = best_checkpoint_dir if best_checkpoint_dir else output_dir
        best_output_path = os.path.join(output_dir, "test_predictions_best.json")
        run_infer_custom(adapter_path=best_adapter_path, output_path=best_output_path)
    else:
        from commands.inference.default import run_infer_default
        # For default head, checkpoint contains adapter files
        best_adapter_path = best_checkpoint_dir if best_checkpoint_dir else output_dir
        best_output_path = os.path.join(output_dir, "test_predictions_best.json")
        run_infer_default(adapter_path=best_adapter_path, output_path=best_output_path)
    logger.info(f"✓ Saved best model predictions to {best_output_path}")
    
    # Run inference on milestone model if it exists
    if milestone_checkpoint_dir is not None and os.path.exists(milestone_checkpoint_dir):
        logger.info(f"Running inference on {threshold_percent}% milestone model (checkpoint: {milestone_checkpoint_dir})...")
        milestone_output_path = os.path.join(output_dir, f"test_predictions_{threshold_percent}percent.json")
        if use_custom_head:
            run_infer_custom(adapter_path=milestone_checkpoint_dir, output_path=milestone_output_path)
        else:
            run_infer_default(adapter_path=milestone_checkpoint_dir, output_path=milestone_output_path)
        logger.info(f"✓ Saved {threshold_percent}% milestone model predictions to {milestone_output_path}")
    else:
        if first_milestone_step is not None:
            logger.warning(f"{threshold_percent}% milestone reached at step {first_milestone_step}, but checkpoint not found")
            # Try to find it by searching all checkpoint directories
            run_dir = trainer._get_output_dir()
            import glob
            pattern = os.path.join(run_dir, f"checkpoint-{first_milestone_step}*")
            matches = glob.glob(pattern)
            if matches:
                milestone_checkpoint_dir = matches[0]
                logger.info(f"Found {threshold_percent}% milestone checkpoint at {milestone_checkpoint_dir}, running inference...")
                milestone_output_path = os.path.join(output_dir, f"test_predictions_{threshold_percent}percent.json")
                if use_custom_head:
                    run_infer_custom(adapter_path=milestone_checkpoint_dir, output_path=milestone_output_path)
                else:
                    run_infer_default(adapter_path=milestone_checkpoint_dir, output_path=milestone_output_path)
                logger.info(f"✓ Saved {threshold_percent}% milestone model predictions to {milestone_output_path}")
        else:
            logger.info(f"{threshold_percent}% accuracy threshold not reached, skipping milestone model inference")
