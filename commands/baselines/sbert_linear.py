import argparse
import os
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from commands.baselines.common import (
    data_paths,
    load_json_dataset,
    prepare_seed,
    reset_memory,
    resolve_baseline_output_dir,
    save_pickle,
    save_run_metadata,
    timed_energy,
    verify_model_available,
    write_evaluation_results,
    write_predictions,
)
from logger_config import logger


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    embeddings = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = model(**encoded)
        pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def predict_one(
    text: str,
    tokenizer,
    model,
    classifier: LogisticRegression,
    device: torch.device,
    max_length: int,
) -> int:
    emb = encode_texts([text], tokenizer, model, device, batch_size=1, max_length=max_length)
    return int(classifier.predict(emb)[0])


def run_sbert_linear(
    model_name: Optional[str] = None,
    batch_size: int = 64,
    max_length: int = 128,
    classifier_max_iter: int = 2000,
    classifier_c: float = 1.0,
) -> None:
    seed = prepare_seed()
    baseline_name = "sbert_linear"
    model_name = model_name or os.getenv("SBERT_MODEL") or DEFAULT_MODEL
    experiment_name, _output_base, output_dir = resolve_baseline_output_dir(baseline_name)

    train_path, test_path = data_paths()
    train_texts, train_labels = load_json_dataset(train_path)
    test_texts, test_labels = load_json_dataset(test_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running {baseline_name}: model={model_name}, device={device}, output={output_dir}")

    reset_memory()
    verify_model_available(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    with timed_energy(output_dir, experiment_name, "sbert_linear_training") as train_timer:
        train_embeddings = encode_texts(
            train_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        classifier = LogisticRegression(
            C=classifier_c,
            max_iter=classifier_max_iter,
            random_state=seed,
            solver="lbfgs",
            n_jobs=1,
        )
        classifier.fit(train_embeddings, train_labels)

    preds = []
    latencies_ms = []
    with timed_energy(output_dir, experiment_name, "sbert_linear_inference") as infer_timer:
        for text in test_texts:
            started = time.perf_counter()
            preds.append(predict_one(text, tokenizer, model, classifier, device, max_length=max_length))
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

    save_pickle(os.path.join(output_dir, "classifier.pkl"), classifier)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
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
            "classifier": "LogisticRegression",
            "classifier_c": classifier_c,
            "classifier_max_iter": classifier_max_iter,
            "embedding_batch_size": batch_size,
            "max_length": max_length,
            "seed": seed,
        },
    )
    save_run_metadata(
        output_dir,
        {
            "baseline": baseline_name,
            "model": model_name,
            "seed": seed,
            "train_samples": len(train_texts),
            "test_samples": len(test_texts),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SBERT/MiniLM embeddings + linear classifier baseline.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--classifier-max-iter", type=int, default=2000)
    parser.add_argument("--classifier-c", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_sbert_linear(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        classifier_max_iter=args.classifier_max_iter,
        classifier_c=args.classifier_c,
    )
