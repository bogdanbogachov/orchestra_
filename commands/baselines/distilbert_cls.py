import argparse
import os
import time
from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from commands.baselines.common import (
    data_paths,
    load_json_dataset,
    num_labels_from_data,
    prepare_seed,
    reset_memory,
    resolve_baseline_output_dir,
    save_run_metadata,
    timed_energy,
    verify_model_available,
    write_evaluation_results,
    write_predictions,
)
from commands.training.data import ClassificationDataset, make_splits
from logger_config import logger


DEFAULT_MODEL = "distilbert-base-uncased"


def run_distilbert_cls(
    model_name: Optional[str] = None,
    max_length: int = 128,
    num_train_epochs: float = 10.0,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
) -> None:
    seed = prepare_seed()
    baseline_name = "distilbert_cls"
    model_name = model_name or os.getenv("DISTILBERT_MODEL") or DEFAULT_MODEL
    experiment_name, _output_base, output_dir = resolve_baseline_output_dir(baseline_name)

    train_path, test_path = data_paths()
    texts, labels = load_json_dataset(train_path)
    test_texts, test_labels = load_json_dataset(test_path)
    num_labels = num_labels_from_data(labels, test_labels)

    logger.info(f"Running {baseline_name}: model={model_name}, labels={num_labels}, output={output_dir}")

    verify_model_available(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    data_config = {
        **CONFIG_DATA_PROCESSING_DEFAULTS,
    }
    try:
        from config import CONFIG
        data_config.update(CONFIG.get("data_processing", {}))
    except Exception:
        pass

    train_texts, val_texts, train_labels, val_labels = make_splits(texts, labels, data_config, seed)
    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed if seed is not None else 42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    reset_memory()
    with timed_energy(output_dir, experiment_name, "distilbert_cls_training") as train_timer:
        train_output = trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    preds = []
    latencies_ms = []
    with timed_energy(output_dir, experiment_name, "distilbert_cls_inference") as infer_timer:
        for text in test_texts:
            started = time.perf_counter()
            encoded = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
            preds.append(int(torch.argmax(logits, dim=-1).item()))
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

    write_predictions(os.path.join(output_dir, "test_predictions.json"), test_texts, test_labels, preds, latencies_ms)
    write_evaluation_results(
        output_dir=output_dir,
        experiment_name=experiment_name,
        baseline_name=baseline_name,
        model_name=model_name,
        train_path=train_path,
        test_path=test_path,
        test_labels=test_labels,
        test_preds=preds,
        latencies_ms=latencies_ms,
        train_samples=len(train_texts),
        train_seconds=train_timer.seconds,
        inference_seconds=infer_timer.seconds,
        training_energy=train_timer.energy,
        inference_energy=infer_timer.energy,
        extra={
            "pooling": "DistilBERT sequence classification head over [CLS] representation",
            "max_length": max_length,
            "num_train_epochs": num_train_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "train_loss": float(train_output.training_loss),
        },
    )
    save_run_metadata(
        output_dir,
        {
            "baseline": baseline_name,
            "model": model_name,
            "seed": seed,
            "train_samples": len(train_texts),
            "validation_samples": len(val_texts),
            "test_samples": len(test_texts),
        },
    )


CONFIG_DATA_PROCESSING_DEFAULTS = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DistilBERT [CLS] fine-tuning baseline.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_distilbert_cls(
        model_name=args.model_name,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
