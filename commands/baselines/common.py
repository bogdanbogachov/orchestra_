import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from commands.training.paths import extract_global_experiment_number
from commands.training.seed_utils import set_seed
from commands.utils.metrics import EnergyTracker, get_memory_usage, reset_memory_tracking
from config import CONFIG
from logger_config import logger


BASELINE_HEADS = {"sbert_linear", "distilbert_cls"}


def load_json_dataset(path: str) -> Tuple[List[str], List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [str(item["text"]) for item in data], [int(item["label"]) for item in data]


def resolve_seed() -> Optional[int]:
    seed = os.getenv("SEED")
    if seed is not None:
        try:
            return int(seed)
        except ValueError:
            logger.warning(f"Invalid SEED environment variable: {seed}. Falling back to config.")
    return CONFIG.get("training", {}).get("seed")


def prepare_seed() -> Optional[int]:
    seed = resolve_seed()
    if seed is not None:
        set_seed(seed)
        logger.info(f"Set random seed to {seed}")
    else:
        logger.info("Using non-deterministic baseline run (no fixed seed)")
    return seed


def resolve_baseline_output_dir(baseline_name: str) -> Tuple[str, str, str]:
    experiment_name = os.getenv("EXP") or CONFIG.get("experiment", "orchestra")
    experiments_dir = CONFIG.get("paths", {}).get("experiments", "experiments")

    import re

    match = re.match(r"^(.+)_(\d+)_(\d+)$", experiment_name)
    global_exp_num = int(match.group(2)) if match else extract_global_experiment_number(experiment_name)
    if global_exp_num is not None:
        if match:
            run_dir_name = f"{match.group(1)}_{match.group(3)}"
        else:
            run_dir_name = experiment_name
        output_base = os.path.join(experiments_dir, str(global_exp_num), run_dir_name)
    else:
        output_base = os.path.join(experiments_dir, experiment_name)

    output_dir = os.path.join(output_base, baseline_name)
    os.makedirs(output_dir, exist_ok=True)
    return experiment_name, output_base, output_dir


def compute_classification_metrics(
    labels: Sequence[int],
    preds: Sequence[int],
    latencies_ms: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(preds, dtype=np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    if latencies_ms:
        lat = np.asarray(latencies_ms, dtype=np.float64)
        out["avg_latency_ms"] = float(lat.mean())
        out["std_latency_ms"] = float(lat.std())
    return out


def write_predictions(path: str, texts: Sequence[str], labels: Sequence[int], preds: Sequence[int], latencies_ms):
    payload = {
        "predictions": [
            {
                "text": text,
                "label": int(label),
                "pred": int(pred),
                "latency_ms": float(latency),
            }
            for text, label, pred, latency in zip(texts, labels, preds, latencies_ms)
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_evaluation_results(
    output_dir: str,
    experiment_name: str,
    baseline_name: str,
    model_name: str,
    train_path: str,
    test_path: str,
    test_labels: Sequence[int],
    test_preds: Sequence[int],
    latencies_ms: Sequence[float],
    train_samples: int,
    train_seconds: float,
    inference_seconds: float,
    training_energy: Dict[str, Any],
    inference_energy: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = compute_classification_metrics(test_labels, test_preds, latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms)

    result = {
        "experiment": experiment_name,
        "test_samples": int(len(test_labels)),
        "inputs": {
            "train_data": train_path,
            "test_data": test_path,
        },
        "head": baseline_name,
        "model_type": "baseline",
        "baseline": {
            "name": baseline_name,
            "model": model_name,
        },
        **metrics,
        "inference_metrics": {
            "total_flops": 0,
            "flops_per_sample": 0,
            "peak_memory_mb": _peak_memory_mb(),
            "memory_info": get_memory_usage(_device_for_memory()),
            "duration_seconds": float(inference_seconds),
            "energy_consumption": inference_energy,
            "carbon_footprint": {
                "emissions_gco2eq": float(inference_energy.get("emissions_gco2eq", 0.0)),
                "emissions_rate_gco2eq_per_hour": float(inference_energy.get("emissions_rate_gco2eq_per_hour", 0.0)),
            },
        },
        "training_metrics": {
            "total_samples_processed": int(train_samples),
            "training_steps": 0,
            "training_epochs": 0,
            "duration_seconds": float(train_seconds),
            "peak_memory_mb": _peak_memory_mb(),
            "memory_info": get_memory_usage(_device_for_memory()),
            "energy_consumption": training_energy,
            "carbon_footprint": {
                "emissions_gco2eq": float(training_energy.get("emissions_gco2eq", 0.0)),
                "emissions_rate_gco2eq_per_hour": float(training_energy.get("emissions_rate_gco2eq_per_hour", 0.0)),
            },
        },
    }
    if extra:
        result["baseline"].update(extra)

    out_file = os.path.join(output_dir, "evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved baseline evaluation to {out_file}")
    return result


def save_pickle(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def timed_energy(output_dir: str, experiment_name: str, task_name: str):
    return _TimedEnergy(output_dir, experiment_name, task_name)


class _TimedEnergy:
    def __init__(self, output_dir: str, experiment_name: str, task_name: str):
        self.tracker = EnergyTracker(output_dir, experiment_name, task_name)
        self.started = 0.0
        self.seconds = 0.0
        self.energy: Dict[str, Any] = {}

    def __enter__(self):
        self.started = time.perf_counter()
        self.tracker.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.seconds = time.perf_counter() - self.started
        self.energy = self.tracker.stop()
        return False


def reset_memory() -> None:
    reset_memory_tracking(_device_for_memory())


def _device_for_memory() -> Optional[torch.device]:
    return torch.device("cuda") if torch.cuda.is_available() else None


def _peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    return float(get_memory_usage(None).get("cpu_rss_mb", 0.0))


def data_paths() -> Tuple[str, str]:
    env_train = os.getenv("TRAIN_PATH")
    env_test = os.getenv("TEST_PATH")
    if env_train and env_test:
        return env_train, env_test

    paths = CONFIG["paths"]["data"]
    return paths["train"], paths["test"]


def num_labels_from_data(*label_lists: Sequence[int]) -> int:
    labels = set()
    for label_list in label_lists:
        labels.update(int(x) for x in label_list)
    return max(labels) + 1 if labels else int(CONFIG.get("model", {}).get("num_labels", 0))


def save_run_metadata(output_dir: str, metadata: Dict[str, Any]) -> None:
    path = Path(output_dir) / "baseline_run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
