import json
import time
import os
from typing import Any, Dict, List, Optional, Union

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import CONFIG
from logger_config import logger
from commands.utils.metrics import get_memory_usage, reset_memory_tracking, calculate_flops_for_transformer, EnergyTracker


def _resolve_default_adapter_path(adapter_path: Optional[str]) -> str:
    if adapter_path is not None:
        return adapter_path
    paths_config = CONFIG["paths"]
    experiment_name = CONFIG.get("experiment", "orchestra")
    
    # Extract global_exp_num and restructure path
    import re
    match = re.match(r'^(.+)_(\d+)_(\d+)$', experiment_name)
    if match:
        base_name = match.group(1)
        global_exp_num = match.group(2)
        per_config_exp_num = match.group(3)
        base_with_per_config = f"{base_name}_{per_config_exp_num}"
        return os.path.join(paths_config["experiments"], global_exp_num, base_with_per_config, "default_head")
    else:
        # Fallback for non-standard experiment names
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
    max_length: int = 512,
) -> Dict[str, Any]:
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
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

    # Get max_length from training config to match training tokenization
    training_config = CONFIG.get("training", {})
    max_length = training_config.get("max_length", 512)

    # Dataset mode
    if input_text_or_json is None or (isinstance(input_text_or_json, (str, os.PathLike)) and os.path.exists(str(input_text_or_json))):
        paths_config = CONFIG["paths"]
        experiment_name = CONFIG.get("experiment", "orchestra")
        test_path = str(input_text_or_json) if input_text_or_json is not None else paths_config["data"]["test"]

        with open(test_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if output_path is None:
            output_path = os.path.join(resolved_adapter_path, "test_predictions.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        results: List[Dict[str, Any]] = []
        logger.info(f"Running default-head inference on dataset: {test_path} ({len(data)} samples)")
        
        # Initialize energy tracker for Green AI metrics
        output_dir = os.path.dirname(output_path)
        energy_tracker = None
        try:
            energy_tracker = EnergyTracker(
                output_dir=output_dir,
                experiment_name=experiment_name,
                task_name="inference_default_head"
            )
            energy_tracker.start()
            logger.info("✓ Started energy tracking for inference")
        except Exception as e:
            logger.warning(f"Could not initialize energy tracker: {e}")
        
        # Reset memory tracking and calculate FLOPs on first sample
        device = next(model.parameters()).device
        reset_memory_tracking(device)
        flops_per_sample = 0
        peak_memory_mb = 0.0
        
        for i, item in enumerate(data):
            text = item["text"]
            true_label = item.get("label")
            start = time.time()
            out = _predict_default_single(model, tokenizer, text, labels=None, max_length=max_length)
            latency_ms = (time.time() - start) * 1000.0
            probs = out["probs"].squeeze(0).detach().cpu().tolist()
            pred = int(int(torch.tensor(probs).argmax().item()))
            
            # Calculate FLOPs on first sample using standard industry approach (thop)
            if i == 0:
                try:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    flops_per_sample = calculate_flops_for_transformer(
                        model, inputs["input_ids"], inputs.get("attention_mask")
                    )
                    logger.info(f"  Calculated forward FLOPs per sample: {flops_per_sample:,} (standard approach)")
                except Exception as e:
                    logger.warning(f"  Could not calculate FLOPs: {e}")
                    flops_per_sample = 0
            
            # Track peak memory usage
            memory_info = get_memory_usage(device)
            if device.type == 'cuda' and torch.cuda.is_available():
                current_peak = memory_info.get('gpu_max_allocated_mb', 0.0)
            else:
                current_peak = memory_info.get('cpu_rss_mb', 0.0)
            peak_memory_mb = max(peak_memory_mb, current_peak)
            
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

        # Get final memory stats
        final_memory_info = get_memory_usage(device)
        
        # Stop energy tracking and get metrics
        energy_metrics = {}
        if energy_tracker is not None:
            try:
                energy_metrics = energy_tracker.stop()
                logger.info("✓ Stopped energy tracking")
            except Exception as e:
                logger.warning(f"Error stopping energy tracker: {e}")
        
        # Calculate total FLOPs for entire inference run
        total_flops = int(flops_per_sample * len(results)) if flops_per_sample > 0 else 0
        if total_flops > 0:
            logger.info(f"  Total inference FLOPs: {total_flops:,} (for {len(results)} samples)")
        
        payload = {
            "experiment": experiment_name,
            "head": "default_head",
            "adapter_path": resolved_adapter_path,
            "input_path": test_path,
            "num_samples": len(results),
            "predictions": results,
            "metrics": {
                "flops_per_sample": int(flops_per_sample),
                "total_flops": int(total_flops),
                "peak_memory_mb": float(peak_memory_mb),
                "memory_info": {k: float(v) for k, v in final_memory_info.items()},
            },
        }
        
        # Add energy and carbon metrics (Green AI metrics)
        if energy_metrics:
            payload["metrics"]["energy_consumption"] = {
                "energy_consumed_kwh": energy_metrics.get("energy_consumed_kwh", 0.0),
                "cpu_energy_kwh": energy_metrics.get("cpu_energy_kwh", 0.0),
                "gpu_energy_kwh": energy_metrics.get("gpu_energy_kwh", 0.0),
                "ram_energy_kwh": energy_metrics.get("ram_energy_kwh", 0.0),
                "duration_seconds": energy_metrics.get("duration_seconds", 0.0),
            }
            payload["metrics"]["carbon_footprint"] = {
                "emissions_gco2eq": energy_metrics.get("emissions_gco2eq", 0.0),
                "emissions_rate_gco2eq_per_hour": energy_metrics.get("emissions_rate_gco2eq_per_hour", 0.0),
                "country_name": energy_metrics.get("country_name", "unknown"),
                "region": energy_metrics.get("region", "unknown"),
            }
            
            energy_kwh = energy_metrics.get("energy_consumed_kwh", 0.0)
            emissions = energy_metrics.get("emissions_gco2eq", 0.0)
            if energy_kwh > 0:
                logger.info(f"  Energy consumed: {energy_kwh:.4f} kWh")
                logger.info(f"  Carbon footprint: {emissions:.4f} gCO₂eq")
                if len(results) > 0:
                    energy_per_sample = energy_kwh / len(results)
                    logger.info(f"  Energy per sample: {energy_per_sample:.6f} kWh")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"✓ Saved default-head predictions to {output_path}")

        return payload

    # Single-text mode
    training_config = CONFIG.get("training", {})
    max_length = training_config.get("max_length", 512)
    return _predict_default_single(model, tokenizer, str(input_text_or_json), labels=labels, max_length=max_length)
