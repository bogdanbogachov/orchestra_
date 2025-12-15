import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from custom_llama_classification import LlamaClassificationHead
from peft import LoraConfig, get_peft_model, TaskType
import json
from typing import List
from sklearn.model_selection import train_test_split


class ClassificationDataset(Dataset):
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
    def __init__(self, base_model, classifier):
        super().__init__()
        self.base_model = base_model
        self.classifier = classifier
        self.config = base_model.config  # Trainer expects config attribute
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        result = self.classifier(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            labels=labels
        )

        return SequenceClassifierOutput(
            logits=result['logits'],
            loss=result['loss']
        )


def load_model_and_tokenizer(
    model_path: str,
    num_labels: int,
    use_custom_head: bool = False,
    pooling_strategy: str = "mean",
    use_fft: bool = False
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|reserved_special_token_15|>"

    if use_custom_head:
        base_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        
        classifier = LlamaClassificationHead(
            config=base_model.config,
            num_labels=num_labels,
            pooling_strategy=pooling_strategy,
            use_fft=use_fft
        ).to(base_model.device)
        
        model = CustomClassificationModel(base_model, classifier)
        fft_status = "enabled" if use_fft else "disabled"
        print(f"✓ Loaded base model with custom classification head (pooling: {pooling_strategy}, FFT: {fft_status})")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        # Set pad_token_id in model config to match the tokenizer
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        print("✓ Loaded model with default classification head")

    return model, tokenizer, use_custom_head


def setup_lora(model, use_custom_head: bool = False):
    if use_custom_head:
        module_name = "classifier"
        task_type = TaskType.FEATURE_EXTRACTION
        target_model = model.base_model
    else:
        module_name = "score"
        task_type = TaskType.SEQ_CLS
        target_model = model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=task_type,
        modules_to_save=[module_name]
    )

    if use_custom_head:
        model.base_model = get_peft_model(target_model, lora_config)
    else:
        model = get_peft_model(target_model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA configured")
    print(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Classification head is trainable")

    return model


def load_data(data_path: str):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    
    return texts, labels


def main():

    output_dir = "default_head"
    use_custom_head = False

    model, tokenizer, use_custom_head = load_model_and_tokenizer(
        model_path="downloaded_models/downloaded_3_2_1b",
        num_labels=50,
        use_custom_head=use_custom_head,
        pooling_strategy="mean",
        use_fft=True
    )

    model = setup_lora(model, use_custom_head=use_custom_head)

    all_texts, all_labels = load_data("train_data.json")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels, 
        test_size=0.2,
        random_state=42,
        stratify=all_labels  # Maintains class distribution in both splits
    )
    
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer)

    print(f"✓ Split data: {len(train_texts)} train, {len(val_texts)} validation ({0.2*100:.1f}%)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_steps=25,
        eval_strategy="steps",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
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

    if isinstance(model, CustomClassificationModel):
        model.base_model.save_pretrained(output_dir)
        classifier_path = f"{output_dir}/classifier.pt"
        torch.save(model.classifier.state_dict(), classifier_path)
    else:
        trainer.save_model()
    
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
