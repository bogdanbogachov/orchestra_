import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import CONFIG
from logger_config import logger

def load_test_data() -> Tuple[List[str], List[int]]:
    paths_config = CONFIG["paths"]
    test_file = paths_config["data"]["test"]

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    labels = [int(item["label"]) for item in data]
    return texts, labels

def _load_predictions(predictions_path: str) -> Dict[str, Any]:
    with open(predictions_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, latencies_ms: Optional[List[float]] = None) -> Dict[str, Any]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    out: Dict[str, Any] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if latencies_ms is not None and len(latencies_ms) > 0:
        lat = np.asarray(latencies_ms, dtype=np.float64)
        out["avg_latency_ms"] = float(lat.mean())
        out["std_latency_ms"] = float(lat.std())
    return out

def run_evaluation(head: Optional[str] = None):
    logger.info("Starting evaluation pipeline...")
    
    paths_config = CONFIG["paths"]
    experiment_name = CONFIG.get("experiment", "orchestra")
    experiments_dir = paths_config["experiments"]
    experiment_dir = os.path.join(experiments_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    _, labels = load_test_data()
    y_true = np.asarray(labels, dtype=np.int64)
    logger.info(f"Loaded {len(y_true)} test labels")

    default_pred_path = os.path.join(experiment_dir, "default_head", "test_predictions.json")
    custom_pred_path = os.path.join(experiment_dir, "custom_head", "test_predictions.json")

    eval_cfg = CONFIG.get("evaluation", {})
    chosen_head = (head or eval_cfg.get("head") or "default_head").strip()

    if chosen_head not in {"default_head", "custom_head"}:
        raise ValueError(f"Invalid eval head: {chosen_head}. Use 'default_head' or 'custom_head'.")

    common = {
        "experiment": experiment_name,
        "test_samples": int(len(y_true)),
        "inputs": {
            "test_data": paths_config["data"]["test"],
            "default_predictions": default_pred_path,
            "custom_predictions": custom_pred_path,
        },
    }

    # Evaluate ONE head at a time.
    pred_path = default_pred_path if chosen_head == "default_head" else custom_pred_path
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing predictions for {chosen_head}: {pred_path}. Run inference first.")

    payload = _load_predictions(pred_path)
    preds = np.asarray([int(x["pred"]) for x in payload["predictions"]], dtype=np.int64)
    if len(preds) != len(y_true):
        raise ValueError(
            f"Prediction/label length mismatch for {chosen_head}: y_true={len(y_true)}, preds={len(preds)}"
        )

    lat = [float(x.get("latency_ms", 0.0)) for x in payload["predictions"]]
    head_metrics = _compute_metrics(y_true, preds, lat)

    head_results = {
        **common,
        "head": chosen_head,
        **head_metrics,
    }

    out_file = os.path.join(experiment_dir, chosen_head, "evaluation_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(head_results, f, indent=2)
    logger.info(f"✓ Saved {chosen_head} evaluation to {out_file}")
