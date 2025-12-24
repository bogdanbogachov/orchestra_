import json
import time
import os
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CONFIG
from logger_config import logger


def _resolve_default_adapter_path(adapter_path: Optional[str]) -> str:
    if adapter_path is not None:
        return adapter_path
    paths_config = CONFIG["paths"]
    experiment_name = CONFIG.get("experiment", "orchestra")

    return os.path.join(paths_config["experiments"], experiment_name, "default_head")


def _load_default_model_and_tokenizer(adapter_path: Optional[str] = None):
    model_config = CONFIG["model"]
    model_path = CONFIG["paths"]["model"]
    adapter_path = _resolve_default_adapter_path(adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = model_config.get("pad_token", tokenizer.eos_token)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(model_config["torch_dtype"], torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=model_config["num_labels"],
        dtype=torch_dtype,
        device_map=model_config["device_map"],
    )

    # Set pad_token_id to match tokenizer so default head can find last non-padding token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapters from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("✓ Loaded finetuned score layer (classification head) from adapter")
    else:
        logger.info(f"Adapter path {adapter_path} not found, using base model with untrained score layer")

    return model, tokenizer, adapter_path


def _predict_default_single(
    model,
    tokenizer,
    input_text: str,
    labels=None,
) -> Dict[str, Any]:
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    return {"logits": logits, "probs": probs, "loss": outputs.loss}


def run_infer_default(
    input_text_or_json: Optional[Union[str, os.PathLike]] = None,
    labels=None,
    adapter_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Union[Dict[str, Any], Dict[str, Any]]:
    """
    If input is a string text -> returns single-item output dict.
    If input is a path to a JSON file (list of {"text","label"}) OR None -> runs dataset inference and saves predictions.
    """
    model, tokenizer, resolved_adapter_path = _load_default_model_and_tokenizer(adapter_path)

    # Dataset mode
    if input_text_or_json is None or (isinstance(input_text_or_json, (str, os.PathLike)) and os.path.exists(str(input_text_or_json))):
        paths_config = CONFIG["paths"]
        experiment_name = CONFIG.get("experiment", "orchestra")
        test_path = str(input_text_or_json) if input_text_or_json is not None else paths_config["data"]["test"]

        with open(test_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if output_path is None:
            output_path = os.path.join(os.path.dirname(resolved_adapter_path), "default_head", "test_predictions.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        results: List[Dict[str, Any]] = []
        logger.info(f"Running default-head inference on dataset: {test_path} ({len(data)} samples)")
        for i, item in enumerate(data):
            text = item["text"]
            true_label = item.get("label")
            start = time.time()
            out = _predict_default_single(model, tokenizer, text, labels=None)
            latency_ms = (time.time() - start) * 1000.0
            probs = out["probs"].squeeze(0).detach().cpu().tolist()
            pred = int(int(torch.tensor(probs).argmax().item()))
            results.append(
                {
                    "text": text,
                    "true_label": true_label,
                    "pred": pred,
                    "probs": probs,
                    "latency_ms": float(latency_ms),
                }
            )
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{len(data)} samples")

        payload = {
            "experiment": experiment_name,
            "head": "default_head",
            "adapter_path": resolved_adapter_path,
            "input_path": test_path,
            "num_samples": len(results),
            "predictions": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"✓ Saved default-head predictions to {output_path}")

        return payload

    # Single-text mode
    return _predict_default_single(model, tokenizer, str(input_text_or_json), labels=labels)
