import json
import torch
from collections import Counter
from typing import List, Tuple, Optional
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from logger_config import logger


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


def load_data(data_path: str) -> Tuple[List[str], List[int]]:
    with open(data_path, "r") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    return texts, labels


def make_splits(
    texts: List[str],
    labels: List[int],
    data_config: dict,
    seed: Optional[int],
):
    split_random_state = data_config.get("random_state") if seed is not None else None
    stratify_labels = labels if data_config.get("stratify") else None
    if stratify_labels is not None:
        label_counts = Counter(labels)
        least_populated_count = min(label_counts.values()) if label_counts else 0
        if least_populated_count < 2:
            logger.warning(
                "Disabling stratified validation split because at least one label has fewer than 2 examples "
                f"(min_count={least_populated_count})."
            )
            stratify_labels = None

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=data_config["test_size"],
        random_state=split_random_state,
        stratify=stratify_labels,
    )
    return train_texts, val_texts, train_labels, val_labels


def build_datasets(
    train_texts, train_labels,
    val_texts, val_labels,
    tokenizer,
    max_length: int
):
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_length=max_length)
    return train_dataset, val_dataset
