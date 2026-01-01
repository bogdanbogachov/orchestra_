"""
GPU energy tracking using nvidia-smi.
Works with MIG devices by querying the parent GPU energy counter.
"""
import subprocess
import time
import os
from typing import Dict, Any, Optional


def get_gpu_energy_nvidia_smi() -> Dict[str, Any]:
    """
    Get GPU energy consumption using nvidia-smi.
    Works with MIG devices by querying the parent GPU.
    
    Returns:
        Dictionary with energy metrics
    """
    try:
        # Query energy consumption counter (cumulative since last reset)
        # This works even with MIG devices
        result = subprocess.run(
            ['nvidia-smi', 
             '--query-gpu=index,uuid,power.draw,power.limit,energy.consumed',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_idx = int(parts[0])
                    gpu_uuid = parts[1]
                    power_draw = float(parts[2])  # Current power draw in Watts
                    power_limit = float(parts[3])  # Power limit in Watts
                    energy_consumed_mj = float(parts[4])  # Cumulative energy in mJ (millijoules)
                    
                    # Convert mJ to kWh: 1 kWh = 3,600,000,000 mJ
                    energy_kwh = energy_consumed_mj / 3_600_000_000.0
                    
                    return {
                        'gpu_index': gpu_idx,
                        'gpu_uuid': gpu_uuid,
                        'power_draw_watts': power_draw,
                        'power_limit_watts': power_limit,
                        'energy_consumed_mj': energy_consumed_mj,
                        'energy_consumed_kwh': energy_kwh,
                        'success': True,
                    }
    except Exception as e:
        pass
    
    return {'success': False}


class NvidiaSMIEnergyTracker:
    """
    Track GPU energy consumption using nvidia-smi.
    Works with MIG devices by reading parent GPU energy counter.
    """
    
    def __init__(self, output_dir: str, experiment_name: str, task_name: str = "training"):
        """
        Initialize nvidia-smi energy tracker.
        
        Args:
            output_dir: Directory to save energy data (for compatibility, not used)
            experiment_name: Name of the experiment (for logging)
            task_name: Name of the task (e.g., "training", "inference")
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.task_name = task_name
        self.initial_energy_mj = None
        self.final_energy_mj = None
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Record initial energy reading."""
        self.start_time = time.time()
        energy_data = get_gpu_energy_nvidia_smi()
        if energy_data.get('success'):
            self.initial_energy_mj = energy_data.get('energy_consumed_mj', 0.0)
        else:
            self.initial_energy_mj = None
    
    def stop(self) -> Dict[str, Any]:
        """
        Record final energy reading and calculate consumption.
        
        Returns:
            Dictionary with energy metrics in kWh
        """
        self.end_time = time.time()
        energy_data = get_gpu_energy_nvidia_smi()
        
        if energy_data.get('success') and self.initial_energy_mj is not None:
            self.final_energy_mj = energy_data.get('energy_consumed_mj', 0.0)
            
            # Calculate energy consumed during tracking period
            energy_delta_mj = self.final_energy_mj - self.initial_energy_mj
            
            # Convert mJ to kWh
            energy_kwh = energy_delta_mj / 3_600_000_000.0
            
            duration_seconds = self.end_time - self.start_time if self.start_time else 0.0
            
            # Estimate CPU/RAM energy (rough estimate: 10% of total for CPU-heavy ML jobs)
            # For GPU jobs, most energy is GPU, so this is conservative
            cpu_energy_kwh = energy_kwh * 0.1  # Estimate
            ram_energy_kwh = energy_kwh * 0.05  # Estimate
            
            # Calculate carbon footprint (Canada average: ~150 gCO₂eq/kWh)
            # Adjust based on your province if known
            grid_intensity = 150.0  # gCO₂eq/kWh (Canada average)
            total_energy_estimate = energy_kwh * 1.15  # Add 15% for CPU/RAM
            emissions = total_energy_estimate * grid_intensity
            emissions_per_hour = emissions / (duration_seconds / 3600.0) if duration_seconds > 0 else 0.0
            
            return {
                'energy_consumed_kwh': total_energy_estimate,
                'gpu_energy_kwh': energy_kwh,
                'cpu_energy_kwh': cpu_energy_kwh,
                'ram_energy_kwh': ram_energy_kwh,
                'duration_seconds': duration_seconds,
                'emissions_gco2eq': emissions,
                'emissions_rate_gco2eq_per_hour': emissions_per_hour,
                'country_name': 'Canada',
                'region': 'unknown',
                'source': 'nvidia_smi',
                'initial_energy_mj': self.initial_energy_mj,
                'final_energy_mj': self.final_energy_mj,
            }
        
        # Return empty metrics if tracking failed
        return {
            'energy_consumed_kwh': 0.0,
            'gpu_energy_kwh': 0.0,
            'cpu_energy_kwh': 0.0,
            'ram_energy_kwh': 0.0,
            'duration_seconds': 0.0,
            'emissions_gco2eq': 0.0,
            'emissions_rate_gco2eq_per_hour': 0.0,
            'country_name': 'unknown',
            'region': 'unknown',
            'source': 'nvidia_smi_failed',
        }

