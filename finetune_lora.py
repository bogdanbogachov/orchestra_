"""
Minimal LoRA fine-tuning script for Llama classification.
Supports both default and custom classification heads.
"""

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.llama.modeling_llama import LlamaClassificationHead
from peft import LoraConfig, get_peft_model, TaskType
import json
from typing import Optional, List, Dict


class ClassificationDataset(Dataset):
    """Simple dataset for text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class CustomClassificationModel(torch.nn.Module):
    """Wrapper for custom classification head with base model."""
    
    def __init__(self, base_model, classifier):
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.config = base_model.config  # Trainer expects config attribute
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get hidden states from base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Classify using custom head
        result = self.classifier(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Return in format expected by Trainer
        return SequenceClassifierOutput(
            logits=result['logits'],
            loss=result['loss']
        )


def load_model_and_tokenizer(
    model_path: str,
    num_labels: int,
    use_custom_head: bool = False,
    pooling_strategy: str = "mean"
):
    """Load model and tokenizer for fine-tuning."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_custom_head:
        # Load base model and create custom classification head
        base_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        
        classifier = LlamaClassificationHead(
            config=base_model.config,
            num_labels=num_labels,
            pooling_strategy=pooling_strategy
        ).to(base_model.device)
        
        model = CustomClassificationModel(base_model, classifier)
        print(f"✓ Loaded base model with custom classification head (pooling: {pooling_strategy})")
    else:
        # Load default classification model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        print("✓ Loaded model with default classification head")
    
    return model, tokenizer, use_custom_head


def setup_lora(model, train_classifier: bool = True, use_custom_head: bool = False):
    """Setup LoRA adapters on the model."""
    
    # For custom head, apply LoRA to base_model first, then wrap
    if use_custom_head and isinstance(model, CustomClassificationModel):
        # Determine which modules to save
        modules_to_save = []
        if train_classifier:
            modules_to_save = ["classifier"]
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # Use FEATURE_EXTRACTION for base model
            modules_to_save=modules_to_save if modules_to_save else None
        )
        
        # Apply LoRA to base_model
        model.base_model = get_peft_model(model.base_model, lora_config)
        
        # If classifier should be trainable, make it so
        if train_classifier:
            for param in model.classifier.parameters():
                param.requires_grad = True
    else:
        # For default head, apply LoRA normally
        modules_to_save = []
        if train_classifier and hasattr(model, 'score'):
            modules_to_save = ["score"]
        
        lora_config = LoraConfig(
            r=8,  # LoRA rank
            lora_alpha=16,  # LoRA alpha scaling
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            modules_to_save=modules_to_save if modules_to_save else None
        )
        
        model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA configured")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total parameters: {total_params:,}")
    if train_classifier:
        print(f"  Classification head is trainable")
    
    return model


def load_data(data_path: str):
    """Load training data from JSON file.
    
    Expected format: [{"text": "...", "label": 0}, ...]
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama with LoRA")
    parser.add_argument("--model_path", type=str, default="downloaded_models/downloaded_3_2_1b",
                       help="Path to model directory")
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data JSON file")
    parser.add_argument("--val_data", type=str, default=None,
                       help="Path to validation data JSON file (optional)")
    parser.add_argument("--output_dir", type=str, default="./lora_checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--num_labels", type=int, default=5,
                       help="Number of classification labels")
    parser.add_argument("--use_custom_head", action="store_true",
                       help="Use custom classification head instead of default")
    parser.add_argument("--pooling_strategy", type=str, default="mean",
                       choices=["mean", "max", "last", "attention"],
                       help="Pooling strategy for custom head (only used if --use_custom_head)")
    parser.add_argument("--train_classifier", action="store_true",
                       help="Make classification head trainable (not just LoRA adapters)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LoRA Fine-tuning Setup")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer, use_custom_head = load_model_and_tokenizer(
        model_path=args.model_path,
        num_labels=args.num_labels,
        use_custom_head=args.use_custom_head,
        pooling_strategy=args.pooling_strategy
    )
    
    # Setup LoRA
    model = setup_lora(model, train_classifier=args.train_classifier, use_custom_head=use_custom_head)
    
    # Load data
    print(f"\nLoading training data from {args.train_data}...")
    train_texts, train_labels = load_data(args.train_data)
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, args.max_length)
    print(f"✓ Loaded {len(train_dataset)} training examples")
    
    val_dataset = None
    if args.val_data:
        print(f"Loading validation data from {args.val_data}...")
        val_texts, val_labels = load_data(args.val_data)
        val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, args.max_length)
        print(f"✓ Loaded {len(val_dataset)} validation examples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=100,
        eval_steps=100 if val_dataset else None,
        evaluation_strategy="steps" if val_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="loss" if val_dataset else None,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print("✓ Training complete!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Model saved to: {args.output_dir}")
    print(f"Total training steps: {trainer.state.global_step}")
    if val_dataset:
        print(f"Best validation loss: {trainer.state.best_metric:.4f}")


if __name__ == "__main__":
    main()
