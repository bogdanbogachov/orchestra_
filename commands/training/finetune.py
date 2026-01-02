import torch
import json
import os
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
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


def run_finetune():
    training_config = CONFIG['training']
    data_config = CONFIG['data_processing']
    paths_config = CONFIG['paths']
    experiment_name = CONFIG.get('experiment', 'orchestra')
    experiments_dir = paths_config['experiments']
    
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
    output_dir = os.path.join(experiments_dir, experiment_name, head_type)
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
        'report_to': 'none'
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
        callbacks=[early_stopping_callback, metrics_callback],
    )

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
        model.base_model.save_pretrained(output_dir)
        classifier_path = f"{output_dir}/classifier.pt"
        torch.save(model.classifier.state_dict(), classifier_path)
        logger.info(f"✓ Saved custom classifier to {classifier_path}")
    else:
        trainer.save_model()
        logger.info(f"✓ Saved default head model with finetuned score layer to {output_dir}")
    
    tokenizer.save_pretrained(output_dir)
    logger.info(f"✓ Saved tokenizer to {output_dir}")
