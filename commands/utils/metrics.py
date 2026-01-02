"""
Utility functions for tracking FLOPs, memory usage, energy consumption, and carbon footprint.

Uses standard industry tools:
- thop: For FLOPs calculation
- nvidia-smi: For GPU energy consumption tracking (works with MIG devices)
"""
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
    """
    Get current memory usage in MB.
    
    Args:
        device: PyTorch device. If None, returns CPU memory.
        
    Returns:
        Dictionary with memory usage metrics in MB.
    """
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
    """
    Reset memory tracking counters.
    
    Args:
        device: PyTorch device. If None, resets CPU tracking.
    """
    if device is not None and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _clear_thop_hooks_and_attributes(model: torch.nn.Module) -> None:
    """
    Remove all thop hooks and attributes from a model.
    
    This is critical for PEFT/LoRA models to prevent AttributeError
    during inference after profiling.
    
    Args:
        model: PyTorch model to clean
    """
    def _recursive_clean(module):
        """Recursively clean module and all submodules."""
        try:
            # Remove thop attributes (try both hasattr and direct __dict__ check)
            for attr in ['total_ops', 'total_params']:
                if hasattr(module, attr):
                    try:
                        delattr(module, attr)
                    except (AttributeError, KeyError, TypeError):
                        pass
                # Also check __dict__ directly (for wrapped modules)
                if hasattr(module, '__dict__') and attr in module.__dict__:
                    try:
                        del module.__dict__[attr]
                    except (AttributeError, KeyError, TypeError):
                        pass

            # Remove all hooks (safely)
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()
            if hasattr(module, '_forward_pre_hooks'):
                module._forward_pre_hooks.clear()
            if hasattr(module, '_backward_hooks'):
                module._backward_hooks.clear()
        except Exception:
            # If cleaning fails for a module, continue with others
            pass

        # Recursively clean children
        for child in module.children():
            _recursive_clean(child)

    # Clean the model and all submodules
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


def calculate_flops_for_transformer(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> int:
    """
    Calculate FLOPs for a transformer model with specific inputs.

    This uses the standard industry approach (thop library).
    Calculates forward pass FLOPs only.

    IMPORTANT: This function cleans up all thop artifacts after profiling
    to prevent AttributeError with PEFT/LoRA models during subsequent inference.

    Args:
        model: PyTorch model
        input_ids: Input token IDs tensor
        attention_mask: Optional attention mask tensor

    Returns:
        Number of FLOPs (forward pass only) as integer
    """
    if not THOP_AVAILABLE:
        logger.warning("thop not available, returning 0 FLOPs")
        return 0

    was_training = model.training
    model.eval()

    try:
        # Clear any existing thop artifacts before profiling
        _clear_thop_hooks_and_attributes(model)

        with torch.no_grad():
            wrapped = TransformerModelWrapper(model)
            flops, params = profile(
                wrapped,
                inputs=(input_ids, attention_mask) if attention_mask is not None else (input_ids,),
                verbose=False
            )

        return int(flops)

    except Exception as e:
        logger.error(f"Failed to calculate FLOPs: {e}", exc_info=True)
        return 0
    finally:
        # CRITICAL: Clean up thop artifacts to prevent inference errors
        _clear_thop_hooks_and_attributes(model)

        # Restore training mode
        if was_training:
            model.train()


def calculate_training_flops(
    forward_flops_per_sample: int,
    num_samples: int,
    backward_multiplier: float = 2.0,
) -> int:
    """
    Calculate total training FLOPs using the standard industry approach.

    Standard approach:
    - Forward pass FLOPs: measured directly
    - Backward pass FLOPs: approximated as backward_multiplier × forward pass
    - Total FLOPs per sample = Forward + Backward
    - Total training FLOPs = Total FLOPs per sample × num_samples

    Args:
        forward_flops_per_sample: Forward pass FLOPs for one sample
        num_samples: Total number of training samples processed
        backward_multiplier: Backward pass cost relative to forward
                            - Full training: 2.0 (standard assumption)
                            - LoRA/PEFT: 0.5-1.0 (fewer params to update)

    Returns:
        Total training FLOPs (forward + backward) as integer
    """
    if forward_flops_per_sample <= 0 or num_samples <= 0:
        return 0

    total_flops_per_sample = forward_flops_per_sample * (1 + backward_multiplier)
    return int(total_flops_per_sample * num_samples)


class EnergyTracker:
    """
    Track energy consumption and CO₂ emissions using nvidia-smi.

    This uses nvidia-smi to query GPU energy counters, which works with MIG devices
    by reading the parent GPU's energy consumption.
    """

    def __init__(self, output_dir: str, experiment_name: str, task_name: str = "training"):
        """
        Initialize energy tracker.

        Args:
            output_dir: Directory to save energy data (for compatibility, not actively used)
            experiment_name: Name of the experiment
            task_name: Name of the task (e.g., "training", "inference")
        """
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
        """Start tracking energy consumption."""
        if self.tracker is not None:
            try:
                self.tracker.start()
            except Exception as e:
                logger.warning(f"Failed to start energy tracker: {e}")

    def stop(self) -> Dict[str, Any]:
        """
        Stop tracking and return energy and carbon metrics.

        Returns:
            Dictionary with energy consumption (kWh) and CO₂ emissions (gCO₂eq)
        """
        if self.tracker is not None:
            try:
                return self.tracker.stop()
            except Exception as e:
                logger.warning(f"Failed to stop energy tracker: {e}")
                return self._get_empty_metrics()
        return self._get_empty_metrics()

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics when tracking is unavailable."""
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
        """
        Context manager for energy tracking.

        Usage:
            with energy_tracker.track():
                # Your code here
        """
        self.start()
        try:
            yield
        finally:
            self.stop()
