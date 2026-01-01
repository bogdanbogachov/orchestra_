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


def reset_memory_tracking(device: Optional[torch.device] = None):
    """
    Reset memory tracking counters.
    
    Args:
        device: PyTorch device. If None, resets CPU tracking.
    """
    if device is not None and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def calculate_flops(model: torch.nn.Module, input_shape: Tuple[int, ...], device: Optional[torch.device] = None) -> int:
    """
    Calculate FLOPs (Floating Point Operations) for a model forward pass.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, seq_len, ...)
        device: Device to run calculation on
        
    Returns:
        Number of FLOPs as integer
    """
    try:
        from thop import profile, clever_format
    except ImportError:
        raise ImportError(
            "thop is required for FLOPs calculation. Install it with: pip install thop"
        )
    
    # Create dummy input
    if device is None:
        device = next(model.parameters()).device
    
    # For transformer models, we need input_ids and attention_mask
    # We'll create a dummy input based on the model type
    try:
        # Try to create appropriate dummy input
        batch_size, seq_len = input_shape[0], input_shape[1]
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        dummy_attention_mask = torch.ones((batch_size, seq_len), device=device)
        
        # Check if model expects these inputs
        dummy_inputs = {'input_ids': dummy_input_ids, 'attention_mask': dummy_attention_mask}
        
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_inputs,), verbose=False)
        
        return int(flops)
    except Exception as e:
        # Fallback: try with just input_ids
        try:
            batch_size, seq_len = input_shape[0], input_shape[1]
            dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            
            with torch.no_grad():
                flops, params = profile(model, inputs=(dummy_input_ids,), verbose=False)
            
            return int(flops)
        except Exception:
            # If all else fails, return 0 and log warning
            import warnings
            warnings.warn(f"Could not calculate FLOPs: {e}")
            return 0


def calculate_flops_for_transformer(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> int:
    """
    Calculate FLOPs for a transformer model with specific inputs.
    
    This uses the standard industry approach (thop library).
    Calculates forward pass FLOPs only.
    
    Args:
        model: PyTorch model
        input_ids: Input token IDs tensor
        attention_mask: Optional attention mask tensor
        
    Returns:
        Number of FLOPs (forward pass only) as integer
    """
    try:
        from thop import profile
    except ImportError:
        raise ImportError(
            "thop is required for FLOPs calculation. Install it with: pip install thop"
        )
    
    # Clear any existing thop attributes from previous profiling
    def clear_thop_attributes(model):
        for module in model.modules():
            if hasattr(module, 'total_ops'):
                delattr(module, 'total_ops')
            if hasattr(module, 'total_params'):
                delattr(module, 'total_params')
    
    # Ensure model is in eval mode for profiling
    was_training = model.training
    model.eval()
    
    try:
        # Clear any previous profiling attributes
        clear_thop_attributes(model)
        
        with torch.no_grad():
            # thop.profile expects the model directly, not a wrapper function
            # Try passing inputs as tuple first (most common case)
            if attention_mask is not None:
                flops, params = profile(
                    model,
                    inputs=(input_ids, attention_mask),
                    verbose=False
                )
            else:
                flops, params = profile(
                    model,
                    inputs=(input_ids,),
                    verbose=False
                )
        return int(flops)
    except Exception as e:
        # Fallback: try with dictionary inputs using a Module wrapper
        try:
            clear_thop_attributes(model)
            with torch.no_grad():
                inputs_dict = {'input_ids': input_ids}
                if attention_mask is not None:
                    inputs_dict['attention_mask'] = attention_mask
                # Create a wrapper class that thop can profile (must be a Module, not a function)
                class ModelWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    def forward(self, inputs_dict):
                        return self.model(**inputs_dict)
                
                wrapped_model = ModelWrapper(model)
                flops, params = profile(
                    wrapped_model,
                    inputs=(inputs_dict,),
                    verbose=False
                )
            return int(flops)
        except Exception as e2:
            import warnings
            warnings.warn(f"Could not calculate FLOPs: {e}, {e2}")
            return 0
    finally:
        # Restore training mode
        if was_training:
            model.train()


def calculate_training_flops(
    forward_flops_per_sample: int,
    num_samples: int,
) -> int:
    """
    Calculate total training FLOPs using the standard industry approach.
    
    Standard approach:
    - Forward pass FLOPs: measured directly
    - Backward pass FLOPs: 2x forward pass (standard assumption in papers)
    - Total FLOPs per sample = Forward + Backward = 3x forward pass
    - Total training FLOPs = Total FLOPs per sample × num_samples
    
    This follows the widely accepted convention where backward pass
    requires approximately 2x the FLOPs of forward pass due to gradient
    computation (as used in papers like "EfficientNet", "Vision Transformer", etc.)
    
    Args:
        forward_flops_per_sample: Forward pass FLOPs for one sample
        num_samples: Total number of training samples processed
        
    Returns:
        Total training FLOPs (forward + backward) as integer
    """
    if forward_flops_per_sample <= 0 or num_samples <= 0:
        return 0
    
    # Standard approach: backward pass = 2x forward pass
    # Total per sample = forward + backward = 3x forward
    total_flops_per_sample = forward_flops_per_sample * 3
    total_training_flops = total_flops_per_sample * num_samples
    
    return int(total_training_flops)


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
            import warnings
            warnings.warn(f"Could not initialize nvidia-smi energy tracker: {e}")
    
    def start(self):
        """Start tracking energy consumption."""
        if self.tracker is not None:
            try:
                self.tracker.start()
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to start energy tracker: {e}")
    
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
                import warnings
                warnings.warn(f"Failed to stop energy tracker: {e}")
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
