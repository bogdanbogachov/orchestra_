import json
import time
import os
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from config import CONFIG
from logger_config import logger
from models.custom_llama_classification import LlamaClassificationHead


def _resolve_custom_adapter_path(adapter_path: Optional[str]) -> str:
    if adapter_path is not None:
        return adapter_path
    paths_config = CONFIG["paths"]
    experiment_name = CONFIG.get("experiment", "orchestra")

    return os.path.join(paths_config["experiments"], experiment_name, "custom_head")


def _load_custom_model_tokenizer_and_head(adapter_path: Optional[str] = None):
    model_config = CONFIG["model"]
    model_path = CONFIG["paths"]["model"]
    adapter_path = _resolve_custom_adapter_path(adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = model_config.get("pad_token", tokenizer.eos_token)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(model_config["torch_dtype"], torch.float32)

    base_model = AutoModel.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=model_config["device_map"],
    )

    # Set pad_token_id to match tokenizer for consistency
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    classifier = None
    if os.path.exists(adapter_path):
        logger.info(f"Loading LoRA adapters from {adapter_path}")
        base_model = PeftModel.from_pretrained(base_model, adapter_path)

        classifier_path = os.path.join(adapter_path, "classifier.pt")
        if os.path.exists(classifier_path):
            logger.info(f"Loading fine-tuned classifier from {classifier_path}")
            classifier_state = torch.load(classifier_path, map_location=base_model.device)
            pooling_strategy = model_config["pooling_strategy"]
            use_fft = model_config["use_fft"]
            logger.info(f"Custom head configuration - Pooling type: {pooling_strategy}, FFT used: {use_fft}")
            classifier = LlamaClassificationHead(
                config=base_model.config,
                num_labels=model_config["num_labels"],
                pooling_strategy=pooling_strategy,
                use_fft=use_fft,
            ).to(base_model.device)
            classifier.load_state_dict(classifier_state)
        else:
            logger.info("Classifier not found, using randomly initialized classifier")
            pooling_strategy = model_config["pooling_strategy"]
            use_fft = model_config["use_fft"]
            logger.info(f"Custom head configuration - Pooling type: {pooling_strategy}, FFT used: {use_fft}")

    return base_model, tokenizer, classifier, adapter_path


def _predict_custom_single(
    base_model,
    tokenizer,
    classifier,
    input_text: str,
    labels=None,
) -> Dict[str, Any]:
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = base_model(**inputs)
        hidden_states = outputs.last_hidden_state

    return classifier(
        hidden_states=hidden_states,
        attention_mask=inputs.get("attention_mask"),
        # attention_mask=None,
        labels=labels,
    )


def run_infer_custom(
    input_text_or_json: Optional[Union[str, os.PathLike]] = None,
    labels=None,
    adapter_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Union[Dict[str, Any], Dict[str, Any]]:
    """
    If input is a string text -> returns single-item output dict.
    If input is a path to a JSON file (list of {"text","label"}) OR None -> runs dataset inference and saves predictions.
    """
    base_model, tokenizer, classifier, resolved_adapter_path = _load_custom_model_tokenizer_and_head(adapter_path)

    # Dataset mode
    if input_text_or_json is None or (isinstance(input_text_or_json, (str, os.PathLike)) and os.path.exists(str(input_text_or_json))):
        paths_config = CONFIG["paths"]
        experiment_name = CONFIG.get("experiment", "orchestra")
        test_path = str(input_text_or_json) if input_text_or_json is not None else paths_config["data"]["test"]

        with open(test_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if output_path is None:
            output_path = os.path.join(os.path.dirname(resolved_adapter_path), "custom_head", "test_predictions.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        results: List[Dict[str, Any]] = []
        logger.info(f"Running custom-head inference on dataset: {test_path} ({len(data)} samples)")
        for i, item in enumerate(data):
            text = item["text"]
            true_label = item.get("label")
            start = time.time()
            out = _predict_custom_single(base_model, tokenizer, classifier, text, labels=None)
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
            "head": "custom_head",
            "adapter_path": resolved_adapter_path,
            "input_path": test_path,
            "num_samples": len(results),
            "predictions": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"✓ Saved custom-head predictions to {output_path}")
        return payload

    # Single-text mode
    return _predict_custom_single(base_model, tokenizer, classifier, str(input_text_or_json), labels=labels)
