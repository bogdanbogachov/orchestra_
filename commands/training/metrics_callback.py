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
    def __init__(self, output_dir: str):
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

    def on_train_batch_begin(self, args, state, control, model=None, inputs=None, **kwargs):
        if self.flops_calculated or model is None or inputs is None:
            return

        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None
        if input_ids is None:
            return

        try:
            # IMPORTANT: unwrap accelerate/DDP wrappers
            real_model = model
            if hasattr(model, "module"):
                real_model = model.module

            # If it's PEFT, base_model exists; but for FLOPs just profile the whole forward
            self.flops_per_sample = int(calculate_flops_for_transformer(real_model, input_ids, attention_mask))
            logger.info(f"  Calculated forward FLOPs per sample: {self.flops_per_sample:,}")
        except Exception as e:
            logger.warning(f"  Could not calculate training FLOPs: {e}")
            self.flops_per_sample = 0
        finally:
            self.flops_calculated = True

    def on_substep_end(self, args, state, control, model=None, inputs=None, **kwargs):
        if inputs is None or not isinstance(inputs, dict):
            return

        # --- FLOPs: compute once, as soon as we have real tensors ---
        if (not self.flops_calculated) and (model is not None):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")

            if input_ids is not None:
                try:
                    # If it's a custom-head model, profile base_model (stable)
                    target = model.base_model if hasattr(model, "base_model") else model
                    self.flops_per_sample = int(
                        calculate_flops_for_transformer(target, input_ids, attention_mask)
                    )
                    logger.info(f"  Calculated forward FLOPs per sample: {self.flops_per_sample:,}")
                except Exception as e:
                    logger.warning(f"  Could not calculate training FLOPs: {e}")
                    self.flops_per_sample = 0
                finally:
                    self.flops_calculated = True  # don't try again

        # --- Sample counting (your existing logic) ---
        if "input_ids" in inputs and inputs["input_ids"] is not None:
            actual_batch_size = inputs["input_ids"].shape[0]
            self.total_samples_processed += actual_batch_size
            self._samples_counted_via_substep = True
            return

        for _, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                actual_batch_size = value.shape[0]
                self.total_samples_processed += actual_batch_size
                self._samples_counted_via_substep = True
                return

    def on_step_end(self, args, state, control, model=None, **kwargs):
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
