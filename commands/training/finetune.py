from config import CONFIG
from logger_config import logger

from commands.training.model import load_model_and_tokenizer, setup_lora
from commands.training.seed_utils import set_seed

from .data import load_data, make_splits, build_datasets
from .paths import resolve_output_dirs, verify_output_dir
from .args import compute_gradient_accumulation_steps, build_training_arguments
from .callbacks import build_callbacks
from .trainer_factory import build_trainer
from .trainer_patches import patch_peft_adapter_only_save_load, patch_accuracy_in_checkpoint_names
from .flops import precompute_flops
from .post_train import save_final_artifacts, run_milestone_inference


def run_finetune():
    training_config = CONFIG['training']
    data_config = CONFIG['data_processing']
    paths_config = CONFIG['paths']
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
    # Check environment variable first, then config file
    seed = os.getenv('SEED')
    if seed is not None:
        try:
            seed = int(seed)
        except ValueError:
            logger.warning(f"Invalid SEED environment variable: {seed}. Using config value.")
            seed = training_config.get('seed', None)
    else:
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

    output_base, output_dir, head_type, global_exp_num = resolve_output_dirs(
        experiments_dir=experiments_dir,
        experiment_name=experiment_name,
        use_custom_head=use_custom_head,
    )
    verify_output_dir(experiments_dir, experiment_name, head_type, output_dir, global_exp_num)

    logger.info(f"Training {head_type} - output directory: {output_dir}")

    max_length = training_config["max_length"]

    texts, labels = load_data(paths_config["data"]["train"])
    train_texts, val_texts, train_labels, val_labels = make_splits(texts, labels, data_config, seed)
    train_dataset, val_dataset = build_datasets(
        train_texts, train_labels, val_texts, val_labels, tokenizer, max_length=max_length
    )
    logger.info(f"✓ Split data: {len(train_texts)} train, {len(val_texts)} validation ({data_config['test_size']*100:.1f}%)")

    grad_accum = compute_gradient_accumulation_steps(training_config)
    training_args = build_training_arguments(training_config, output_dir, seed, grad_accum)

    callbacks, metrics_cb, milestone_cb, acc_threshold = build_callbacks(training_config, output_dir)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        callbacks=callbacks,
    )

    if use_custom_head:
        trainer = patch_peft_adapter_only_save_load(trainer, model, training_args)

    trainer = patch_accuracy_in_checkpoint_names(trainer)

    precompute_flops(trainer, metrics_cb)

    trainer.train()

    save_final_artifacts(model, tokenizer, trainer, output_dir, use_custom_head, training_args)

    run_milestone_inference(
        trainer=trainer,
        milestone_callback=milestone_cb,
        output_dir=output_dir,
        use_custom_head=use_custom_head,
        accuracy_threshold=acc_threshold,
    )
