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
    
    # Extract global_exp_num and restructure path
    import re
    match = re.match(r'^(.+)_(\d+)_(\d+)$', experiment_name)
    if match:
        base_name = match.group(1)
        global_exp_num = match.group(2)
        per_config_exp_num = match.group(3)
        base_with_per_config = f"{base_name}_{per_config_exp_num}"
        experiment_dir = os.path.join(experiments_dir, global_exp_num, base_with_per_config)
    else:
        # Fallback for non-standard experiment names
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
    # Exclude first prediction from latency statistics (warmup/cold start)
    lat_for_stats = lat[1:] if len(lat) > 1 else lat
    if len(lat) > 1:
        logger.info(f"  Excluding first prediction from latency stats (warmup: {lat[0]:.2f} ms, using {len(lat_for_stats)} samples)")
    head_metrics = _compute_metrics(y_true, preds, lat_for_stats)
    
    # Extract FLOPs and memory metrics from predictions payload (inference metrics)
    metrics = payload.get("metrics", {})
    inference_flops_per_sample = metrics.get("flops_per_sample", 0)
    inference_total_flops = metrics.get("total_flops", 0)
    inference_peak_memory_mb = metrics.get("peak_memory_mb", 0.0)
    inference_memory_info = metrics.get("memory_info", {})
    inference_energy_consumption = metrics.get("energy_consumption", {})
    inference_carbon_footprint = metrics.get("carbon_footprint", {})
    
    # Try to load training metrics if available
    training_metrics_path = os.path.join(experiment_dir, chosen_head, "training_metrics.json")
    training_metrics = {}
    if os.path.exists(training_metrics_path):
        try:
            with open(training_metrics_path, "r", encoding="utf-8") as f:
                training_metrics = json.load(f)
            logger.info(f"✓ Loaded training metrics from {training_metrics_path}")
        except Exception as e:
            logger.warning(f"Could not load training metrics: {e}")

    head_results = {
        **common,
        "head": chosen_head,
        **head_metrics,
        "inference_metrics": {
            "flops_per_sample": int(inference_flops_per_sample),
            "total_flops": int(inference_total_flops),
            "peak_memory_mb": float(inference_peak_memory_mb),
            "memory_info": inference_memory_info,
        },
    }

    # Add inference energy and carbon metrics if available
    if inference_energy_consumption:
        head_results["inference_metrics"]["energy_consumption"] = inference_energy_consumption
    if inference_carbon_footprint:
        head_results["inference_metrics"]["carbon_footprint"] = inference_carbon_footprint
    
    # Add training metrics if available
    if training_metrics:
        head_results["training_metrics"] = {
            "flops_per_sample_forward": training_metrics.get("flops_per_sample_forward", 0),
            "flops_per_sample_backward": training_metrics.get("flops_per_sample_backward", 0),
            "flops_per_sample_total": training_metrics.get("flops_per_sample_total", 0),
            "total_flops": training_metrics.get("total_flops", 0),
            "total_samples_processed": training_metrics.get("total_samples_processed", 0),
            "peak_memory_mb": training_metrics.get("peak_memory_mb", 0.0),
            "memory_info": training_metrics.get("memory_info", {}),
            "training_steps": training_metrics.get("training_steps", 0),
            "training_epochs": training_metrics.get("training_epochs", 0),
            "calculation_method": training_metrics.get("calculation_method", "standard_industry_approach"),
            "backward_multiplier": training_metrics.get("backward_multiplier", 2.0),
        }
        
        # Add training energy and carbon metrics if available (Green AI metrics)
        if "energy_consumption" in training_metrics:
            head_results["training_metrics"]["energy_consumption"] = training_metrics["energy_consumption"]
        if "carbon_footprint" in training_metrics:
            head_results["training_metrics"]["carbon_footprint"] = training_metrics["carbon_footprint"]

    out_file = os.path.join(experiment_dir, chosen_head, "evaluation_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(head_results, f, indent=2)
    logger.info(f"✓ Saved {chosen_head} evaluation to {out_file}")
