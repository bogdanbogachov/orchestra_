import torch
import json
import os
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from config import CONFIG
from commands.training.dataset import ClassificationDataset
from commands.training.model import load_model_and_tokenizer, setup_lora, CustomClassificationModel
from logging import logger

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
    
    model, tokenizer, use_custom_head = load_model_and_tokenizer()
    model = setup_lora(model, use_custom_head)
    
    head_type = "custom_head" if use_custom_head else "default_head"
    output_dir = os.path.join(experiments_dir, experiment_name, head_type)
    os.makedirs(output_dir, exist_ok=True)
    max_length = training_config['max_length']
    
    logger.info(f"Training {head_type} - output directory: {output_dir}")
    
    all_texts, all_labels = load_data(paths_config['data']['train'])
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels, 
        test_size=data_config['test_size'],
        random_state=data_config['random_state'],
        stratify=all_labels if data_config['stratify'] else None
    )
    
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_length=max_length)

    logger.info(f"✓ Split data: {len(train_texts)} train, {len(val_texts)} validation ({data_config['test_size']*100:.1f}%)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        learning_rate=training_config['learning_rate'],
        logging_dir=f"{output_dir}/logs",
        logging_steps=training_config['logging_steps'],
        eval_steps=training_config['eval_steps'],
        eval_strategy=training_config['eval_strategy'],
        save_strategy=training_config['save_strategy'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        fp16=training_config['fp16'] and torch.cuda.is_available(),
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

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
