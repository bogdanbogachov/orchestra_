import torch
import json
import os
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from config import CONFIG
from commands.training.dataset import ClassificationDataset
from commands.training.model import load_model_and_tokenizer, setup_lora
from commands.training.seed_utils import set_seed
from commands.training.metrics_callback import TrainingMetricsCallback
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


def run_finetune():
    training_config = CONFIG['training']
    data_config = CONFIG['data_processing']
    paths_config = CONFIG['paths']
    model_config = CONFIG['model']
    experiment_name = CONFIG.get('experiment', 'orchestra')
    experiments_dir = paths_config['experiments']
    
    # Extract global_exp_num and restructure path
    # Format: base_name_global_exp_num_per_config_exp_num
    # Example: "35_l_default_9_10" -> global_exp_num=9, base_with_per_config="35_l_default_10"
    import re
    match = re.match(r'^(.+)_(\d+)_(\d+)$', experiment_name)
    if match:
        base_name = match.group(1)
        global_exp_num = match.group(2)
        per_config_exp_num = match.group(3)
        # New structure: experiments/global_exp_num/base_name_per_config_exp_num/head_type
        base_with_per_config = f"{base_name}_{per_config_exp_num}"
        experiment_base_dir = os.path.join(experiments_dir, global_exp_num, base_with_per_config)
    else:
        # Fallback for non-standard experiment names
        experiment_base_dir = os.path.join(experiments_dir, experiment_name)
    
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
    output_dir = os.path.join(experiment_base_dir, head_type)
    os.makedirs(output_dir, exist_ok=True)
    max_length = training_config['max_length']
    
    logger.info(f"Training {head_type} - output directory: {output_dir}")
    
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback, metrics_callback],
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

    # Override optimizer creation to use separate learning rate for FFT adaptive network
    if use_custom_head and model_config.get('use_fft') and model_config.get('fft_adaptive'):
        from torch.optim import AdamW
        
        base_lr = training_config['learning_rate']
        fft_lr_multiplier = model_config.get('fft_learning_rate_multiplier', 10.0)
        fft_lr = base_lr * fft_lr_multiplier
        
        def create_optimizer_with_fft_lr():
            """Create optimizer with separate learning rate for FFT adaptive network."""
            # First, get decay parameters (for weight decay)
            decay_parameters = trainer.get_decay_parameter_names(trainer.model_wrapped if hasattr(trainer, 'model_wrapped') else trainer.model)
            
            # Separate parameters into groups
            fft_params_decay = []
            fft_params_no_decay = []
            other_params_decay = []
            other_params_no_decay = []
            
            for name, param in trainer.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                if 'fft_adaptive_network' in name:
                    if name in decay_parameters:
                        fft_params_decay.append(param)
                    else:
                        fft_params_no_decay.append(param)
                else:
                    if name in decay_parameters:
                        other_params_decay.append(param)
                    else:
                        other_params_no_decay.append(param)
            
            # Create parameter groups with different learning rates
            param_groups = []
            
            # Get weight decay value safely
            weight_decay = getattr(training_args, 'weight_decay', 0.0)
            
            # FFT parameters with weight decay
            if fft_params_decay:
                param_groups.append({
                    "params": fft_params_decay,
                    "lr": fft_lr,
                    "weight_decay": weight_decay,
                })
            
            # FFT parameters without weight decay
            if fft_params_no_decay:
                param_groups.append({
                    "params": fft_params_no_decay,
                    "lr": fft_lr,
                    "weight_decay": 0.0,
                })
            
            # Other parameters with weight decay
            if other_params_decay:
                param_groups.append({
                    "params": other_params_decay,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                })
            
            # Other parameters without weight decay
            if other_params_no_decay:
                param_groups.append({
                    "params": other_params_no_decay,
                    "lr": base_lr,
                    "weight_decay": 0.0,
                })
            
            # Create optimizer with custom parameter groups
            optimizer = AdamW(
                param_groups,
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                eps=training_args.adam_epsilon,
            )
            
            trainer.optimizer = optimizer
            logger.info(f"✓ Created optimizer with separate learning rates:")
            logger.info(f"  Base LR: {base_lr:.6f} (for all other parameters)")
            logger.info(f"  FFT LR: {fft_lr:.6f} (for FFT adaptive network, {fft_lr_multiplier}x multiplier)")
            logger.info(f"  FFT parameters: {len(fft_params_decay) + len(fft_params_no_decay)}")
            logger.info(f"  Other parameters: {len(other_params_decay) + len(other_params_no_decay)}")
            
            return optimizer
        
        trainer.create_optimizer = create_optimizer_with_fft_lr
        logger.info("✓ Overrode optimizer creation to use separate FFT learning rate")

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
