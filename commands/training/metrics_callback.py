"""
Custom callback for tracking FLOPs and memory usage during training.

Uses the standard industry approach for FLOPs calculation:
- Forward pass FLOPs: Measured directly using thop library
- Backward pass FLOPs: 2x forward pass (standard assumption in papers)
- Total training FLOPs = (Forward + Backward) × num_samples = 3 × Forward × num_samples
"""
import json
import os
import torch
from transformers import TrainerCallback
from logger_config import logger
from commands.utils.metrics import (
    get_memory_usage,
    reset_memory_tracking,
    calculate_flops_for_transformer,
    calculate_training_flops,
    EnergyTracker,
)


class TrainingMetricsCallback(TrainerCallback):
    """
    Callback to track FLOPs and peak memory usage during training.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the callback.
        
        Args:
            output_dir: Directory to save training metrics JSON file
        """
        self.output_dir = output_dir
        self.metrics_file = os.path.join(output_dir, "training_metrics.json")
        self.flops_per_sample = 0
        self.total_flops = 0
        self.peak_memory_mb = 0.0
        self.flops_calculated = False
        self.device = None
        self.total_samples_processed = 0
        self.energy_tracker = None
        self._samples_counted_via_substep = False  # Track if we're using substep counting
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of training."""
        # Get device from model
        if model is not None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                # Model has no parameters (unlikely but handle it)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Reset memory tracking at the start of training
            reset_memory_tracking(self.device)
            
            # Initialize energy tracker for Green AI metrics
            try:
                experiment_name = os.path.basename(os.path.dirname(self.output_dir))
                self.energy_tracker = EnergyTracker(
                    output_dir=self.output_dir,
                    experiment_name=experiment_name,
                    task_name="training"
                )
                self.energy_tracker.start()
                logger.info("✓ Started tracking training metrics (FLOPs, memory, energy, CO₂)")
            except Exception as e:
                logger.warning(f"Could not initialize energy tracker: {e}")
                self.energy_tracker = None
                logger.info("✓ Started tracking training metrics (FLOPs and memory)")
    
    def on_step_begin(self, args, state, control, model=None, inputs=None, **kwargs):
        """Called at the beginning of each training step."""
        if model is None or inputs is None:
            return
        
        # Calculate FLOPs on the first step
        if not self.flops_calculated and state.global_step == 0:
            try:
                # Extract input_ids and attention_mask from inputs
                input_ids = inputs.get('input_ids')
                attention_mask = inputs.get('attention_mask')
                
                if input_ids is not None:
                    # For custom head models, we need to handle differently
                    # Check if model has base_model attribute (custom head)
                    if hasattr(model, 'base_model'):
                        # Calculate FLOPs for base model
                        base_flops = calculate_flops_for_transformer(
                            model.base_model, input_ids, attention_mask
                        )
                        
                        # Try to calculate classifier FLOPs
                        try:
                            from thop import profile
                            with torch.no_grad():
                                base_outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
                                hidden_states = base_outputs.last_hidden_state
                                # Profile classifier forward pass
                                classifier_flops, _ = profile(
                                    model.classifier,
                                    inputs=(hidden_states, attention_mask),
                                    verbose=False
                                )
                            self.flops_per_sample = int(base_flops + classifier_flops)
                        except Exception:
                            # Fallback: just use base model FLOPs
                            self.flops_per_sample = int(base_flops)
                    else:
                        # Default head model
                        self.flops_per_sample = calculate_flops_for_transformer(
                            model, input_ids, attention_mask
                        )
                    
                    logger.info(f"  Calculated forward FLOPs per sample: {self.flops_per_sample:,}")
                    logger.info(f"  (Using standard industry approach: backward = 2x forward)")
                    self.flops_calculated = True
            except Exception as e:
                logger.warning(f"  Could not calculate training FLOPs: {e}")
                self.flops_per_sample = 0
                self.flops_calculated = True  # Don't try again
    
    def on_substep_end(self, args, state, control, model=None, inputs=None, **kwargs):
        """Called at the end of each gradient accumulation substep."""
        # Track samples processed in each substep using actual batch size (most accurate method)
        if inputs is not None:
            if isinstance(inputs, dict):
                # Try to get batch size from input_ids or any tensor
                if 'input_ids' in inputs and inputs['input_ids'] is not None:
                    actual_batch_size = inputs['input_ids'].shape[0]
                    self.total_samples_processed += actual_batch_size
                    self._samples_counted_via_substep = True
                    return
                else:
                    # Try to find any tensor to get batch size
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor) and value.dim() > 0:
                            actual_batch_size = value.shape[0]
                            self.total_samples_processed += actual_batch_size
                            self._samples_counted_via_substep = True
                            return
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step (after all gradient accumulation)."""
        if self.device is not None:
            # Track peak memory usage
            memory_info = get_memory_usage(self.device)
            if self.device.type == 'cuda' and torch.cuda.is_available():
                current_peak = memory_info.get('gpu_max_allocated_mb', 0.0)
            else:
                current_peak = memory_info.get('cpu_rss_mb', 0.0)
            self.peak_memory_mb = max(self.peak_memory_mb, current_peak)
            
            # Note: Sample counting is primarily done in on_substep_end for accuracy
            # This ensures we count actual batch sizes, not theoretical ones
            # Fallback: if on_substep_end wasn't called or inputs weren't available,
            # try to get from inputs here, or use args as last resort
            if not self._samples_counted_via_substep:
                # on_substep_end wasn't used, try to count here
                samples_this_step = 0
                if 'inputs' in kwargs and kwargs['inputs'] is not None:
                    inputs = kwargs['inputs']
                    if isinstance(inputs, dict):
                        if 'input_ids' in inputs and inputs['input_ids'] is not None:
                            actual_batch_size = inputs['input_ids'].shape[0]
                            grad_accum = args.gradient_accumulation_steps if args is not None and hasattr(args, 'gradient_accumulation_steps') else 1
                            samples_this_step = actual_batch_size * grad_accum
                        else:
                            # Try to find any tensor to get batch size
                            for key, value in inputs.items():
                                if isinstance(value, torch.Tensor) and value.dim() > 0:
                                    actual_batch_size = value.shape[0]
                                    grad_accum = args.gradient_accumulation_steps if args is not None and hasattr(args, 'gradient_accumulation_steps') else 1
                                    samples_this_step = actual_batch_size * grad_accum
                                    break
                
                # Final fallback: use args if inputs not available
                if samples_this_step == 0 and args is not None:
                    batch_size = args.per_device_train_batch_size
                    grad_accum = args.gradient_accumulation_steps
                    num_gpus = args.n_gpu if hasattr(args, 'n_gpu') else 1
                    samples_this_step = batch_size * grad_accum * num_gpus
                
                if samples_this_step > 0:
                    self.total_samples_processed += samples_this_step
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of training."""
        # Get final memory stats
        final_memory_info = get_memory_usage(self.device) if self.device is not None else {}
        
        # Stop energy tracking and get metrics
        energy_metrics = {}
        if self.energy_tracker is not None:
            try:
                energy_metrics = self.energy_tracker.stop()
                logger.info("✓ Stopped energy tracking")
            except Exception as e:
                logger.warning(f"Error stopping energy tracker: {e}")
        
        # Calculate total FLOPs using standard industry approach
        # Standard approach: backward pass = 2x forward pass
        # Total = forward + backward = 3x forward per sample
        self.total_flops = calculate_training_flops(
            self.flops_per_sample,
            self.total_samples_processed
        )
        
        # Calculate backward FLOPs per sample (for reporting)
        backward_flops_per_sample = self.flops_per_sample * 2 if self.flops_per_sample > 0 else 0
        
        # Prepare metrics dictionary
        training_metrics = {
            "flops_per_sample_forward": int(self.flops_per_sample),
            "flops_per_sample_backward": int(backward_flops_per_sample),
            "flops_per_sample_total": int(self.flops_per_sample * 3) if self.flops_per_sample > 0 else 0,
            "total_flops": int(self.total_flops),
            "total_samples_processed": int(self.total_samples_processed),
            "peak_memory_mb": float(self.peak_memory_mb),
            "memory_info": {k: float(v) for k, v in final_memory_info.items()},
            "training_steps": state.global_step if state else 0,
            "training_epochs": state.epoch if state else 0,
            "calculation_method": "standard_industry_approach",
            "backward_multiplier": 2.0,  # Standard multiplier
        }
        
        # Add energy and carbon metrics (Green AI metrics)
        if energy_metrics:
            training_metrics["energy_consumption"] = {
                "energy_consumed_kwh": energy_metrics.get("energy_consumed_kwh", 0.0),
                "cpu_energy_kwh": energy_metrics.get("cpu_energy_kwh", 0.0),
                "gpu_energy_kwh": energy_metrics.get("gpu_energy_kwh", 0.0),
                "ram_energy_kwh": energy_metrics.get("ram_energy_kwh", 0.0),
                "duration_seconds": energy_metrics.get("duration_seconds", 0.0),
            }
            training_metrics["carbon_footprint"] = {
                "emissions_gco2eq": energy_metrics.get("emissions_gco2eq", 0.0),
                "emissions_rate_gco2eq_per_hour": energy_metrics.get("emissions_rate_gco2eq_per_hour", 0.0),
                "country_name": energy_metrics.get("country_name", "unknown"),
                "region": energy_metrics.get("region", "unknown"),
            }
        
        # Save metrics to file
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(training_metrics, f, indent=2)
        
        logger.info(f"✓ Saved training metrics to {self.metrics_file}")
        logger.info(f"  Forward FLOPs per sample: {self.flops_per_sample:,}")
        logger.info(f"  Backward FLOPs per sample: {backward_flops_per_sample:,} (2x forward, standard approach)")
        logger.info(f"  Total FLOPs per sample: {self.flops_per_sample * 3:,} (forward + backward)")
        logger.info(f"  Total training FLOPs: {self.total_flops:,} (for {self.total_samples_processed:,} samples)")
        logger.info(f"  Peak memory during training: {self.peak_memory_mb:.2f} MB")
        
        if energy_metrics:
            energy_kwh = energy_metrics.get("energy_consumed_kwh", 0.0)
            emissions = energy_metrics.get("emissions_gco2eq", 0.0)
            logger.info(f"  Energy consumed: {energy_kwh:.4f} kWh")
            logger.info(f"  Carbon footprint: {emissions:.4f} gCO₂eq")
            if self.total_samples_processed > 0:
                energy_per_sample = energy_kwh / self.total_samples_processed
                logger.info(f"  Energy per sample: {energy_per_sample:.6f} kWh")

