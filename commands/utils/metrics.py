"""
Utility functions for tracking FLOPs, memory usage, energy consumption, and carbon footprint.

Uses standard industry tools:
- thop: For FLOPs calculation
- CodeCarbon: For energy consumption and CO₂ emissions tracking (Green AI standard)
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
    
    # Create a wrapper function for thop to handle dictionary inputs
    def model_wrapper(input_ids, attention_mask=None):
        if attention_mask is not None:
            return model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            return model(input_ids=input_ids)
    
    try:
        with torch.no_grad():
            if attention_mask is not None:
                flops, params = profile(model_wrapper, inputs=(input_ids, attention_mask), verbose=False)
            else:
                flops, params = profile(model_wrapper, inputs=(input_ids,), verbose=False)
        return int(flops)
    except Exception as e:
        # Fallback: try direct model call with tuple inputs
        try:
            with torch.no_grad():
                if attention_mask is not None:
                    # Try passing as separate arguments
                    flops, params = profile(model, inputs=(input_ids, attention_mask), verbose=False)
                else:
                    flops, params = profile(model, inputs=(input_ids,), verbose=False)
            return int(flops)
        except Exception as e2:
            # Final fallback: try with dictionary
            try:
                with torch.no_grad():
                    inputs_dict = {'input_ids': input_ids}
                    if attention_mask is not None:
                        inputs_dict['attention_mask'] = attention_mask
                    flops, params = profile(model, inputs=(inputs_dict,), verbose=False)
                return int(flops)
            except Exception:
                import warnings
                warnings.warn(f"Could not calculate FLOPs: {e}, {e2}")
                return 0


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
    Wrapper for CodeCarbon EmissionsTracker to track energy consumption and CO₂ emissions.
    
    This follows the Green AI standard approach used in papers for measuring
    environmental impact of ML experiments.
    """
    
    def __init__(self, output_dir: str, experiment_name: str, task_name: str = "training"):
        """
        Initialize energy tracker.
        
        Args:
            output_dir: Directory to save emissions data
            experiment_name: Name of the experiment
            task_name: Name of the task (e.g., "training", "inference")
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.task_name = task_name
        self.tracker = None
        self._initialize_tracker()
    
    def _initialize_tracker(self):
        """Initialize CodeCarbon tracker."""
        try:
            from codecarbon import EmissionsTracker
            
            # Create output directory for emissions data
            os.makedirs(self.output_dir, exist_ok=True)
            emissions_file = os.path.join(self.output_dir, f"{self.task_name}_emissions.csv")
            
            self.tracker = EmissionsTracker(
                output_dir=self.output_dir,
                output_file=emissions_file,
                project_name=f"{self.experiment_name}_{self.task_name}",
                log_level="error",  # Reduce verbosity
            )
        except ImportError:
            self.tracker = None
            import warnings
            warnings.warn(
                "CodeCarbon not installed. Energy and carbon tracking will be disabled. "
                "Install with: pip install codecarbon"
            )
    
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
                self.tracker.stop()
                
                # Try to get emissions data from tracker
                # CodeCarbon stores data in different ways depending on version
                try:
                    # Try accessing the emissions data directly
                    if hasattr(self.tracker, '_emissions_data') and self.tracker._emissions_data:
                        emissions_data = self.tracker._emissions_data
                    elif hasattr(self.tracker, 'final_emissions_data') and self.tracker.final_emissions_data:
                        emissions_data = self.tracker.final_emissions_data
                    else:
                        # Try to read from CSV file
                        emissions_file = os.path.join(self.output_dir, f"{self.task_name}_emissions.csv")
                        if os.path.exists(emissions_file):
                            import pandas as pd
                            df = pd.read_csv(emissions_file)
                            if len(df) > 0:
                                last_row = df.iloc[-1]
                                emissions_data = {
                                    "energy_consumed": float(last_row.get("energy_consumed", 0.0)),
                                    "emissions": float(last_row.get("emissions", 0.0)),
                                    "emissions_rate": float(last_row.get("emissions_rate", 0.0)),
                                    "cpu_energy": float(last_row.get("cpu_energy", 0.0)),
                                    "gpu_energy": float(last_row.get("gpu_energy", 0.0)),
                                    "ram_energy": float(last_row.get("ram_energy", 0.0)),
                                    "duration": float(last_row.get("duration", 0.0)),
                                    "country_name": str(last_row.get("country_name", "unknown")),
                                    "region": str(last_row.get("region", "unknown")),
                                }
                            else:
                                emissions_data = None
                        else:
                            emissions_data = None
                    
                    if emissions_data:
                        return {
                            "energy_consumed_kwh": float(emissions_data.get("energy_consumed", 0.0)),
                            "emissions_gco2eq": float(emissions_data.get("emissions", 0.0)),
                            "emissions_rate_gco2eq_per_hour": float(emissions_data.get("emissions_rate", 0.0)),
                            "cpu_energy_kwh": float(emissions_data.get("cpu_energy", 0.0)),
                            "gpu_energy_kwh": float(emissions_data.get("gpu_energy", 0.0)),
                            "ram_energy_kwh": float(emissions_data.get("ram_energy", 0.0)),
                            "duration_seconds": float(emissions_data.get("duration", 0.0)),
                            "country_name": emissions_data.get("country_name", "unknown"),
                            "region": emissions_data.get("region", "unknown"),
                        }
                except Exception as e:
                    import warnings
                    warnings.warn(f"Could not extract emissions data: {e}")
                
                return self._get_empty_metrics()
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
