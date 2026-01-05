import json
import os
import time
from transformers import TrainerCallback
from logger_config import logger
from commands.utils.metrics import (
    get_memory_usage,
    calculate_training_flops,
)


class AccuracyMilestoneCallback(TrainerCallback):
    """
    Callback to track when eval_accuracy reaches a specified milestone threshold.
    - Snapshots training metrics when threshold is reached
    - Tracks which checkpoint first reached the threshold
    - Modifies checkpoint folder names to include eval_accuracy
    """
    
    def __init__(self, output_dir: str, accuracy_threshold: float = 0.95, metrics_callback=None):
        self.output_dir = output_dir
        self.accuracy_threshold = accuracy_threshold
        self.metrics_callback = metrics_callback  # Reference to TrainingMetricsCallback
        self.first_milestone_checkpoint = None
        self.first_milestone_step = None
        self.first_milestone_accuracy = None
        # Create dynamic file name based on threshold (e.g., milestone_95_percent_metrics.json for 0.95)
        threshold_percent = int(accuracy_threshold * 100)
        self.milestone_metrics_file = os.path.join(output_dir, f"milestone_{threshold_percent}_percent_metrics.json")
        self.checkpoint_info_file = os.path.join(output_dir, "checkpoint_info.json")
        threshold_key = f"first_{threshold_percent}_percent"
        self.checkpoint_info = {threshold_key: None, "all_checkpoints": []}
        
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Called after each evaluation step."""
        if logs is None:
            return
            
        eval_accuracy = logs.get("eval_accuracy")
        if eval_accuracy is None:
            return
            
        current_step = state.global_step
        
        # Record checkpoint info with accuracy
        checkpoint_info = {
            "step": current_step,
            "epoch": state.epoch if hasattr(state, 'epoch') else None,
            "eval_accuracy": float(eval_accuracy),
            "eval_loss": float(logs.get("eval_loss", 0.0)),
        }
        self.checkpoint_info["all_checkpoints"].append(checkpoint_info)
        
        # Check if we've reached the threshold for the first time
        if (eval_accuracy >= self.accuracy_threshold and 
            self.first_milestone_checkpoint is None):
            
            self.first_milestone_checkpoint = current_step
            self.first_milestone_step = current_step
            self.first_milestone_accuracy = float(eval_accuracy)
            
            threshold_percent = self.accuracy_threshold * 100
            logger.info(f"🎯 MILESTONE: Reached {eval_accuracy*100:.2f}% accuracy (threshold: {threshold_percent:.1f}%) at step {current_step}!")
            
            # Snapshot training metrics at this point
            self._snapshot_training_metrics(state)
            
            # Record in checkpoint info
            threshold_key = f"first_{int(threshold_percent)}_percent"
            self.checkpoint_info[threshold_key] = {
                "step": current_step,
                "epoch": state.epoch if hasattr(state, 'epoch') else None,
                "eval_accuracy": float(eval_accuracy),
                "eval_loss": float(logs.get("eval_loss", 0.0)),
            }
            
            # Save checkpoint info
            self._save_checkpoint_info()
    
    def _snapshot_training_metrics(self, state):
        """Snapshot all training metrics when accuracy threshold is reached."""
        if self.metrics_callback is None:
            logger.warning("Metrics callback not available, cannot snapshot training metrics")
            return
        
        try:
            # Get current training metrics from the metrics callback
            device = self.metrics_callback.device
            final_memory_info = get_memory_usage(device) if device is not None else {}
            
            # Calculate current FLOPs up to this point
            current_total_flops = calculate_training_flops(
                self.metrics_callback.flops_per_sample,
                self.metrics_callback.total_samples_processed
            )
            
            backward_flops_per_sample = self.metrics_callback.flops_per_sample * 2 if self.metrics_callback.flops_per_sample > 0 else 0
            
            # Prepare snapshot metrics
            threshold_percent = int(self.accuracy_threshold * 100)
            snapshot_metrics = {
                "milestone": f"{threshold_percent}_percent_accuracy",
                "accuracy_threshold": float(self.accuracy_threshold),
                "step": state.global_step,
                "epoch": state.epoch if hasattr(state, 'epoch') else None,
                "flops_per_sample_forward": int(self.metrics_callback.flops_per_sample),
                "flops_per_sample_backward": int(backward_flops_per_sample),
                "flops_per_sample_total": int(self.metrics_callback.flops_per_sample * 3) if self.metrics_callback.flops_per_sample > 0 else 0,
                "total_flops_at_milestone": int(current_total_flops),
                "total_samples_processed_at_milestone": int(self.metrics_callback.total_samples_processed),
                "peak_memory_mb": float(self.metrics_callback.peak_memory_mb),
                "memory_info": {k: float(v) for k, v in final_memory_info.items()},
                "training_steps": state.global_step if state else 0,
                "training_epochs": state.epoch if state else 0,
                "calculation_method": "standard_industry_approach",
                "backward_multiplier": 2.0,
            }
            
            # Get energy metrics if available (snapshot current state)
            # Note: We can't get current metrics without stopping, so we'll estimate
            # based on elapsed time. Full metrics will be in final training_metrics.json
            if self.metrics_callback.energy_tracker is not None and self.metrics_callback.energy_tracker.tracker is not None:
                try:
                    tracker = self.metrics_callback.energy_tracker.tracker
                    if hasattr(tracker, 'start_time') and tracker.start_time is not None:
                        duration_seconds = time.time() - tracker.start_time
                        # Estimate energy based on current state (approximate)
                        # Full accurate metrics will be in final training_metrics.json
                        snapshot_metrics["energy_consumption"] = {
                            "duration_seconds": duration_seconds,
                            "note": "Estimated at milestone - see final training_metrics.json for complete metrics"
                        }
                except Exception as e:
                    logger.warning(f"Could not estimate energy metrics snapshot: {e}")
            
            # Save snapshot
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.milestone_metrics_file, "w", encoding="utf-8") as f:
                json.dump(snapshot_metrics, f, indent=2)
            
            threshold_percent = int(self.accuracy_threshold * 100)
            logger.info(f"✓ Snapshot training metrics at {threshold_percent}% milestone to {self.milestone_metrics_file}")
            
        except Exception as e:
            logger.error(f"Error snapshotting training metrics: {e}")
    
    def _save_checkpoint_info(self):
        """Save checkpoint information including which one first reached 95%."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.checkpoint_info_file, "w", encoding="utf-8") as f:
                json.dump(self.checkpoint_info, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save checkpoint info: {e}")
    
    def on_train_end(self, args, state, control, model=None, **kwargs):
        """Save final checkpoint info at end of training."""
        self._save_checkpoint_info()
        
        threshold_percent = int(self.accuracy_threshold * 100)
        if self.first_milestone_checkpoint is not None:
            logger.info(f"✓ First {threshold_percent}% accuracy reached at step {self.first_milestone_checkpoint} "
                       f"(accuracy: {self.first_milestone_accuracy*100:.2f}%)")
        else:
            logger.info(f"⚠ {threshold_percent}% accuracy threshold not reached during training")
    
    def get_first_95_percent_checkpoint(self):
        """Get the checkpoint step that first reached the accuracy threshold."""
        return self.first_milestone_checkpoint
    
    def get_checkpoint_info(self):
        """Get all checkpoint information."""
        return self.checkpoint_info

