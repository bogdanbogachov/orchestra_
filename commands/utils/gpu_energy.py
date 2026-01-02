import subprocess
import time
import logging
from typing import Dict, Any
import threading

logger = logging.getLogger(__name__)


def get_gpu_energy_nvidia_smi() -> Dict[str, Any]:
    def _run_query(fields: str):
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )

    try:
        # 1) Try energy counter (often unsupported -> your return code 2)
        result = _run_query("index,uuid,power.draw,power.limit,energy.consumed")

        # 2) Fallback: power only (always available on most systems)
        energy_supported = True
        if result.returncode != 0:
            energy_supported = False
            result = _run_query("index,uuid,power.draw,power.limit")

        if result.returncode != 0:
            logger.warning(f"nvidia-smi failed rc={result.returncode} stderr={result.stderr} stdout={result.stdout}")
            return {"success": False, "error": f"nvidia-smi failed: {result.stderr or result.stdout}"}

        if not result.stdout.strip():
            logger.warning("nvidia-smi returned empty output")
            return {"success": False, "error": "empty output"}

        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]

        gpu_idx = int(parts[0])
        gpu_uuid = parts[1]

        def _to_float(x: str):
            x = x.strip()
            if x in {"[N/A]", "N/A", ""}:
                return None
            try:
                return float(x)
            except ValueError:
                return None

        power_draw = _to_float(parts[2])
        power_limit = _to_float(parts[3])

        energy_mj = None
        if energy_supported and len(parts) >= 5:
            try:
                energy_mj = float(parts[4])
            except ValueError:
                energy_mj = None

        return {
            "gpu_index": gpu_idx,
            "gpu_uuid": gpu_uuid,
            "power_draw_watts": power_draw,  # may be None
            "power_limit_watts": power_limit,  # may be None
            "energy_consumed_mj": energy_mj,  # may be None
            "energy_counter_supported": energy_mj is not None,
            "success": True,
        }

    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi command timed out")
        return {"success": False, "error": "timeout"}
    except FileNotFoundError:
        logger.warning("nvidia-smi command not found")
        return {"success": False, "error": "nvidia-smi not found"}
    except Exception as e:
        logger.warning(f"Error calling nvidia-smi: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


class NvidiaSMIEnergyTracker:
    def __init__(self, output_dir: str, experiment_name: str, task_name: str = "training"):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.task_name = task_name
        self.initial_energy_mj = None
        self.final_energy_mj = None
        self.start_time = None
        self.end_time = None
        self._use_energy_counter = False
        self._stop_evt = None
        self._thread = None
        self._energy_joules = 0.0
        self._last_t = None

    def start(self):
        self.start_time = time.time()

        energy_data = get_gpu_energy_nvidia_smi()
        if energy_data.get("success") and energy_data.get("energy_counter_supported"):
            self._use_energy_counter = True
            self.initial_energy_mj = energy_data.get("energy_consumed_mj", 0.0)
            logger.info(f"Started energy tracking (counter): initial={self.initial_energy_mj:.2f} mJ")
            return

        # Fallback: integrate power.draw over time
        self._use_energy_counter = False
        self.initial_energy_mj = None
        self._energy_joules = 0.0
        self._last_t = time.time()
        self._stop_evt = threading.Event()

        def _sampler():
            while not self._stop_evt.is_set():
                now = time.time()
                dt = now - (self._last_t or now)
                self._last_t = now

                d = get_gpu_energy_nvidia_smi()
                if d.get("success"):
                    p = d.get("power_draw_watts")
                    if isinstance(p, (int, float)) and dt > 0:
                        self._energy_joules += float(p) * float(dt)

                time.sleep(0.5)

        self._thread = threading.Thread(target=_sampler, daemon=True)
        self._thread.start()
        logger.info("Started energy tracking (integrating power.draw)")
    
    def stop(self) -> Dict[str, Any]:
        self.end_time = time.time()
        duration_seconds = self.end_time - self.start_time if self.start_time else 0.0

        if self._use_energy_counter:
            energy_data = get_gpu_energy_nvidia_smi()
            if energy_data.get("success") and self.initial_energy_mj is not None:
                self.final_energy_mj = energy_data.get("energy_consumed_mj", 0.0)
                energy_delta_mj = self.final_energy_mj - self.initial_energy_mj
                energy_kwh = energy_delta_mj / 3_600_000_000.0
            else:
                energy_kwh = 0.0
        else:
            if self._stop_evt is not None:
                self._stop_evt.set()
            if self._thread is not None:
                self._thread.join(timeout=2)
            energy_kwh = self._energy_joules / 3_600_000.0  # J -> Wh (3600), then /1000 -> kWh

        cpu_energy_kwh = energy_kwh * 0.1
        ram_energy_kwh = energy_kwh * 0.05
        grid_intensity = 150.0
        total_energy_estimate = energy_kwh * 1.15
        emissions = total_energy_estimate * grid_intensity
        emissions_per_hour = emissions / (duration_seconds / 3600.0) if duration_seconds > 0 else 0.0

        return {
            "energy_consumed_kwh": total_energy_estimate,
            "gpu_energy_kwh": energy_kwh,
            "cpu_energy_kwh": cpu_energy_kwh,
            "ram_energy_kwh": ram_energy_kwh,
            "duration_seconds": duration_seconds,
            "emissions_gco2eq": emissions,
            "emissions_rate_gco2eq_per_hour": emissions_per_hour,
            "country_name": "Canada",
            "region": "unknown",
            "source": "nvidia_smi",
            "initial_energy_mj": self.initial_energy_mj,
            "final_energy_mj": self.final_energy_mj,
        }
