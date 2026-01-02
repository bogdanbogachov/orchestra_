import torch
from typing import Dict, Any, Optional, Tuple
import psutil
import os
import contextlib
import logging

logger = logging.getLogger(__name__)

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    logger.warning("thop not available - FLOPs calculation will be disabled")


def get_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    memory_info = {}
    
    if device is not None and device.type == 'cuda':
        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated(device) / (1024 ** 2)
            memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved(device) / (1024 ** 2)
            memory_info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            memory_info['gpu_max_reserved_mb'] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        # CPU memory
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        memory_info['cpu_rss_mb'] = mem_info.rss / (1024 ** 2)
        memory_info['cpu_vms_mb'] = mem_info.vms / (1024 ** 2)
    
    return memory_info


def reset_memory_tracking(device: Optional[torch.device] = None) -> None:
    if device is not None and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _clear_thop_hooks_and_attributes(model: torch.nn.Module) -> None:
    def _recursive_clean(m: torch.nn.Module):
        # Remove thop buffers/attrs
        for attr in ("total_ops", "total_params"):
            try:
                if hasattr(m, "_buffers") and attr in m._buffers:
                    m._buffers.pop(attr, None)
            except Exception:
                pass
            try:
                if hasattr(m, "__dict__") and attr in m.__dict__:
                    del m.__dict__[attr]
            except Exception:
                pass
            try:
                if hasattr(m, attr):
                    delattr(m, attr)
            except Exception:
                pass

        # Remove hooks
        try:
            if hasattr(m, "_forward_hooks"):
                m._forward_hooks.clear()
            if hasattr(m, "_forward_pre_hooks"):
                m._forward_pre_hooks.clear()
            if hasattr(m, "_backward_hooks"):
                m._backward_hooks.clear()
        except Exception:
            pass

        for child in m.children():
            _recursive_clean(child)

    try:
        _recursive_clean(model)
    except Exception as e:
        logger.warning(f"Error during thop cleanup: {e}")


class TransformerModelWrapper(torch.nn.Module):
    """Wrapper to handle keyword arguments for thop profiling."""

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, input_ids, attention_mask=None):
        kwargs = {'input_ids': input_ids}
        if attention_mask is not None:
            kwargs['attention_mask'] = attention_mask
        return self.model(**kwargs)


def estimate_transformer_forward_flops_per_sample(model: torch.nn.Module, seq_len: int) -> int:
    # Works for LLaMA-like decoder-only blocks (good approximation)
    cfg = getattr(model, "config", None)
    if cfg is None and hasattr(model, "base_model"):
        cfg = getattr(model.base_model, "config", None)
    if cfg is None:
        return 0

    L = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    d = int(getattr(cfg, "hidden_size", 0) or 0)
    f = int(getattr(cfg, "intermediate_size", 0) or 0)
    if L <= 0 or d <= 0 or f <= 0 or seq_len <= 0:
        return 0

    s = int(seq_len)

    # Per-layer forward FLOPs (multiply-add counted as 2 FLOPs):
    # - Attention projections (Q,K,V,out): 8 * s * d^2
    # - Attention matmuls (QK^T and AV): 4 * s^2 * d
    # - Gated MLP (LLaMA): 6 * s * d * f
    flops_per_layer = (8 * s * d * d) + (4 * s * s * d) + (6 * s * d * f)
    flops = L * flops_per_layer

    # Add classification head (rough): 2 * d * num_labels
    num_labels = int(getattr(cfg, "num_labels", 0) or 0)
    if num_labels > 0:
        flops += 2 * d * num_labels

    return int(flops)


def calculate_flops_for_transformer(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> int:
    # Return forward FLOPs PER SAMPLE
    try:
        seq_len = int(input_ids.shape[1])
        return estimate_transformer_forward_flops_per_sample(model, seq_len)
    except Exception as e:
        logger.warning(f"FLOPs estimate failed: {e}")
        return 0


def calculate_training_flops(
    forward_flops_per_sample: int,
    num_samples: int,
    backward_multiplier: float = 2.0,
) -> int:
    if forward_flops_per_sample <= 0 or num_samples <= 0:
        return 0

    total_flops_per_sample = forward_flops_per_sample * (1 + backward_multiplier)
    return int(total_flops_per_sample * num_samples)


class EnergyTracker:
    def __init__(self, output_dir: str, experiment_name: str, task_name: str = "training"):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.task_name = task_name
        try:
            from commands.utils.gpu_energy import NvidiaSMIEnergyTracker
            self.tracker = NvidiaSMIEnergyTracker(output_dir, experiment_name, task_name)
        except Exception as e:
            self.tracker = None
            logger.warning(f"Could not initialize nvidia-smi energy tracker: {e}")

    def start(self) -> None:
        if self.tracker is not None:
            try:
                self.tracker.start()
            except Exception as e:
                logger.warning(f"Failed to start energy tracker: {e}")

    def stop(self) -> Dict[str, Any]:
        if self.tracker is not None:
            try:
                return self.tracker.stop()
            except Exception as e:
                logger.warning(f"Failed to stop energy tracker: {e}")
                return self._get_empty_metrics()
        return self._get_empty_metrics()

    def _get_empty_metrics(self) -> Dict[str, Any]:
        return {
            "energy_consumed_kwh": 0.0,
            "emissions_gco2eq": 0.0,
            "emissions_rate_gco2eq_per_hour": 0.0,
            "cpu_energy_kwh": 0.0,
            "gpu_energy_kwh": 0.0,
            "ram_energy_kwh": 0.0,
            "duration_seconds": 0.0,
            "country_name": "unknown",
            "region": "unknown",
        }

    @contextlib.contextmanager
    def track(self):
        self.start()
        try:
            yield
        finally:
            self.stop()
