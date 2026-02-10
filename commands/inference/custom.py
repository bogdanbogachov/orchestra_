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
from commands.utils.metrics import get_memory_usage, reset_memory_tracking, calculate_flops_for_transformer, EnergyTracker


def _resolve_custom_adapter_path(adapter_path: Optional[str]) -> str:
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
        return os.path.join(paths_config["experiments"], global_exp_num, base_with_per_config, "custom_head")
    else:
        # Fallback for non-standard experiment names
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
            
            # Check for missing/unexpected keys when loading
            result = classifier.load_state_dict(classifier_state, strict=False)
            if result.missing_keys:
                logger.warning(f"⚠️ Missing keys when loading classifier: {result.missing_keys}")
                logger.warning("⚠️ Some classifier weights were NOT loaded - this may cause poor performance!")
            if result.unexpected_keys:
                logger.warning(f"⚠️ Unexpected keys in classifier checkpoint: {result.unexpected_keys}")
            
            if not result.missing_keys and not result.unexpected_keys:
                logger.info("✓ Successfully loaded all classifier weights")
            else:
                logger.error("❌ Classifier state dict has mismatched keys - accuracy may be severely degraded!")
        else:
            logger.info("Classifier not found, using randomly initialized classifier")


    return base_model, tokenizer, classifier, adapter_path


def _predict_custom_single(
    base_model,
    tokenizer,
    classifier,
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
    inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = base_model(**inputs)
        hidden_states = outputs.last_hidden_state

    return classifier(
        hidden_states=hidden_states,
        attention_mask=inputs.get("attention_mask"),
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
        logger.info(f"Running custom-head inference on dataset: {test_path} ({len(data)} samples)")
        
        # Initialize energy tracker for Green AI metrics
        output_dir = os.path.dirname(output_path)
        energy_tracker = None
        try:
            energy_tracker = EnergyTracker(
                output_dir=output_dir,
                experiment_name=experiment_name,
                task_name="inference_custom_head"
            )
            energy_tracker.start()
            logger.info("✓ Started energy tracking for inference")
        except Exception as e:
            logger.warning(f"Could not initialize energy tracker: {e}")
        
        # Reset memory tracking and calculate FLOPs on first sample
        device = next(base_model.parameters()).device
        reset_memory_tracking(device)
        flops_per_sample = 0
        peak_memory_mb = 0.0
        
        for i, item in enumerate(data):
            text = item["text"]
            true_label = item.get("label")
            start = time.time()
            out = _predict_custom_single(base_model, tokenizer, classifier, text, labels=None, max_length=max_length)
            latency_ms = (time.time() - start) * 1000.0
            probs = out["probs"].squeeze(0).detach().cpu().tolist()
            pred = int(int(torch.tensor(probs).argmax().item()))
            
            # Calculate FLOPs on first sample
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
                    
                    # Calculate FLOPs for base model
                    base_flops = calculate_flops_for_transformer(
                        base_model, inputs["input_ids"], inputs.get("attention_mask")
                    )
                    logger.info(f"  Base model FLOPs per sample: {base_flops:,}")
                    
                    # Calculate FLOPs for classifier (approximate by running forward pass)
                    # We'll use thop to profile the combined forward pass
                    classifier_flops = 0
                    try:
                        from thop import profile
                        with torch.no_grad():
                            base_outputs = base_model(**inputs)
                            hidden_states = base_outputs.last_hidden_state
                            # Profile classifier forward pass
                            classifier_flops, _ = profile(
                                classifier,
                                inputs=(hidden_states, inputs.get("attention_mask")),
                                verbose=False
                            )
                        classifier_flops = int(classifier_flops)
                        classifier_flops_calculated = True
                        logger.info(f"  Classifier FLOPs per sample: {classifier_flops:,}")
                    except Exception as e:
                        # Fallback: just use base model FLOPs
                        logger.warning(f"  Could not calculate classifier FLOPs: {e}")
                        logger.warning(f"  Falling back to base model FLOPs only")
                        classifier_flops_calculated = False
                    
                    flops_per_sample = int(base_flops + classifier_flops)
                    
                    if classifier_flops_calculated:
                        logger.info(f"  Total forward FLOPs per sample: {flops_per_sample:,} (base: {base_flops:,} + classifier: {classifier_flops:,})")
                    else:
                        logger.info(f"  Total forward FLOPs per sample: {flops_per_sample:,} (base model only, classifier FLOPs unavailable)")
                    logger.info(f"  (Using standard industry approach)")
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
            "head": "custom_head",
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
        logger.info(f"✓ Saved custom-head predictions to {output_path}")
        return payload

    # Single-text mode
    training_config = CONFIG.get("training", {})
    max_length = training_config.get("max_length", 512)
    return _predict_custom_single(base_model, tokenizer, classifier, str(input_text_or_json), labels=labels, max_length=max_length)
