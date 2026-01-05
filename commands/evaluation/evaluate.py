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

def extract_global_experiment_number(experiment_name: str) -> Optional[int]:
    """
    Extract global experiment number from experiment name.
    
    Format: base_name_global_exp_num_per_config_exp_num
    Example: "35_l_default_8_1" -> 8
    """
    import re
    # Pattern: base_name_global_exp_num_per_config_exp_num
    # Extract the second-to-last number (global experiment number)
    match = re.match(r'^(.+)_(\d+)_(\d+)$', experiment_name)
    if match:
        return int(match.group(2))  # global_exp_num is the second-to-last number
    return None


def run_evaluation(head: Optional[str] = None):
    logger.info("Starting evaluation pipeline...")
    
    paths_config = CONFIG["paths"]
    experiment_name = CONFIG.get("experiment", "orchestra")
    experiments_dir = paths_config["experiments"]
    
    # Extract global experiment number and use nested directory structure
    global_exp_num = extract_global_experiment_number(experiment_name)
    if global_exp_num is not None:
        # New structure: experiments/global_exp_num/experiment_name
        experiment_dir = os.path.join(experiments_dir, str(global_exp_num), experiment_name)
        logger.info(f"Using nested directory structure: experiments/{global_exp_num}/{experiment_name}")
    else:
        # Fallback to old structure if global number can't be extracted
        experiment_dir = os.path.join(experiments_dir, experiment_name)
        logger.warning(f"Could not extract global experiment number from '{experiment_name}', using flat structure")
    
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

    # Evaluate best model and milestone model if available
    head_dir = os.path.join(experiment_dir, chosen_head)
    best_pred_path = os.path.join(head_dir, "test_predictions_best.json")
    
    # Find milestone prediction file dynamically (e.g., test_predictions_95percent.json)
    import glob
    milestone_pattern = os.path.join(head_dir, "test_predictions_*percent.json")
    milestone_files = glob.glob(milestone_pattern)
    milestone_pred_path = milestone_files[0] if milestone_files else None
    
    # Fallback to default prediction file if best doesn't exist
    default_pred_path_for_head = default_pred_path if chosen_head == "default_head" else custom_pred_path
    if not os.path.exists(best_pred_path):
        best_pred_path = default_pred_path_for_head
    
    if not os.path.exists(best_pred_path):
        raise FileNotFoundError(f"Missing predictions for {chosen_head}: {best_pred_path}. Run inference first.")

    # Evaluate best model
    logger.info(f"Evaluating best model predictions from {best_pred_path}")
    payload_best = _load_predictions(best_pred_path)
    preds_best = np.asarray([int(x["pred"]) for x in payload_best["predictions"]], dtype=np.int64)
    if len(preds_best) != len(y_true):
        raise ValueError(
            f"Prediction/label length mismatch for {chosen_head} (best): y_true={len(y_true)}, preds={len(preds_best)}"
        )

    lat_best = [float(x.get("latency_ms", 0.0)) for x in payload_best["predictions"]]
    lat_for_stats_best = lat_best[1:] if len(lat_best) > 1 else lat_best
    if len(lat_best) > 1:
        logger.info(f"  Excluding first prediction from latency stats (warmup: {lat_best[0]:.2f} ms, using {len(lat_for_stats_best)} samples)")
    head_metrics_best = _compute_metrics(y_true, preds_best, lat_for_stats_best)
    
    # Evaluate milestone model if available
    head_metrics_milestone = None
    payload_milestone = None
    if milestone_pred_path and os.path.exists(milestone_pred_path):
        # Extract threshold from filename (e.g., "test_predictions_95percent.json" -> 95)
        import re
        match = re.search(r'(\d+)percent', os.path.basename(milestone_pred_path))
        threshold_percent = match.group(1) if match else "?"
        logger.info(f"Evaluating {threshold_percent}% milestone model predictions from {milestone_pred_path}")
        payload_milestone = _load_predictions(milestone_pred_path)
        preds_milestone = np.asarray([int(x["pred"]) for x in payload_milestone["predictions"]], dtype=np.int64)
        if len(preds_milestone) != len(y_true):
            logger.warning(f"Prediction/label length mismatch for {chosen_head} ({threshold_percent}%): y_true={len(y_true)}, preds={len(preds_milestone)}")
        else:
            lat_milestone = [float(x.get("latency_ms", 0.0)) for x in payload_milestone["predictions"]]
            lat_for_stats_milestone = lat_milestone[1:] if len(lat_milestone) > 1 else lat_milestone
            if len(lat_milestone) > 1:
                logger.info(f"  Excluding first prediction from latency stats (warmup: {lat_milestone[0]:.2f} ms, using {len(lat_for_stats_milestone)} samples)")
            head_metrics_milestone = _compute_metrics(y_true, preds_milestone, lat_for_stats_milestone)
    
    # Use best model metrics as primary
    head_metrics = head_metrics_best
    payload = payload_best
    
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
        "model_type": "best",
        **head_metrics,
        "inference_metrics": {
            "flops_per_sample": int(inference_flops_per_sample),
            "total_flops": int(inference_total_flops),
            "peak_memory_mb": float(inference_peak_memory_mb),
            "memory_info": inference_memory_info,
        },
    }
    
    # Add 95% milestone model results if available
    if head_metrics_milestone is not None and payload_milestone is not None:
        metrics_milestone = payload_milestone.get("metrics", {})
        head_results["milestone_95_percent"] = {
            "model_type": "95_percent_milestone",
            **head_metrics_milestone,
            "inference_metrics": {
                "flops_per_sample": int(metrics_milestone.get("flops_per_sample", 0)),
                "total_flops": int(metrics_milestone.get("total_flops", 0)),
                "peak_memory_mb": float(metrics_milestone.get("peak_memory_mb", 0.0)),
                "memory_info": metrics_milestone.get("memory_info", {}),
            },
        }
        
        # Add inference energy and carbon metrics for milestone if available
        inference_energy_milestone = metrics_milestone.get("energy_consumption", {})
        inference_carbon_milestone = metrics_milestone.get("carbon_footprint", {})
        if inference_energy_milestone:
            head_results["milestone_95_percent"]["inference_metrics"]["energy_consumption"] = inference_energy_milestone
        if inference_carbon_milestone:
            head_results["milestone_95_percent"]["inference_metrics"]["carbon_footprint"] = inference_carbon_milestone
        
        # Try to load milestone training metrics if available (find dynamically)
        milestone_metrics_pattern = os.path.join(experiment_dir, chosen_head, "milestone_*_percent_metrics.json")
        milestone_metrics_files = glob.glob(milestone_metrics_pattern)
        milestone_metrics_path = milestone_metrics_files[0] if milestone_metrics_files else None
        if os.path.exists(milestone_metrics_path):
            try:
                with open(milestone_metrics_path, "r", encoding="utf-8") as f:
                    milestone_training_metrics = json.load(f)
                head_results["milestone_95_percent"]["training_metrics_at_milestone"] = milestone_training_metrics
                logger.info(f"✓ Loaded 95% milestone training metrics from {milestone_metrics_path}")
            except Exception as e:
                logger.warning(f"Could not load milestone training metrics: {e}")
    
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
