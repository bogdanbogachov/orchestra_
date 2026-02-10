import os
import re
from typing import Optional, Tuple

from logger_config import logger


def extract_global_experiment_number(experiment_name: str) -> Optional[int]:
    match = re.match(r"^(.+)_(\d+)_(\d+)$", experiment_name)
    if match:
        return int(match.group(2))

    parts = experiment_name.split("_")
    if len(parts) >= 3:
        try:
            _ = int(parts[-1])
            second_last = int(parts[-2])
            return second_last
        except (ValueError, IndexError):
            pass

    return None


def resolve_output_dirs(
    experiments_dir: str,
    experiment_name: str,
    use_custom_head: bool,
) -> Tuple[str, str, str, Optional[int]]:
    global_exp_num = extract_global_experiment_number(experiment_name)
    if global_exp_num is not None:
        output_base = os.path.join(experiments_dir, str(global_exp_num), experiment_name)
        logger.info(f"Using nested directory structure: experiments/{global_exp_num}/{experiment_name}")
    else:
        output_base = os.path.join(experiments_dir, experiment_name)
        logger.warning(f"Could not extract global experiment number from '{experiment_name}', using flat structure")
        logger.warning("Expected format: base_name_global_exp_num_per_config_exp_num (e.g., '35_l_default_8_1')")

    head_type = "custom_head" if use_custom_head else "default_head"
    output_dir = os.path.join(output_base, head_type)
    os.makedirs(output_dir, exist_ok=True)

    # Safety check when global number exists
    if global_exp_num is not None:
        expected_global_exp_path = os.path.join(experiments_dir, str(global_exp_num))
        if not output_dir.startswith(expected_global_exp_path):
            raise ValueError(
                f"Output directory {output_dir} does not include global experiment number {global_exp_num}. "
                f"Expected path to start with {expected_global_exp_path}"
            )

    return output_base, output_dir, head_type, global_exp_num


def verify_output_dir(
    experiments_dir: str,
    experiment_name: str,
    head_type: str,
    output_dir: str,
    global_exp_num: Optional[int],
):
    if global_exp_num is None:
        return

    actual_path = os.path.abspath(output_dir)
    expected_components = [experiments_dir, str(global_exp_num), experiment_name, head_type]
    expected_path = os.path.abspath(os.path.join(*expected_components))
    if actual_path != expected_path:
        logger.error(f"Path mismatch! Actual: {actual_path}, Expected: {expected_path}")
        raise ValueError("Output directory path verification failed. Check directory structure.")
