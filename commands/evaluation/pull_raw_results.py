"""
Pull evaluation_results.json files out of the experiments tree.

For each indicated global experiment number, walks experiments/<num>/ and
copies every evaluation_results.json it finds into raw_results_pull/, mirroring
the original path (raw_results_pull/<num>/<run_name>/<head>/evaluation_results.json).
"""

import os
import shutil
from typing import List, Optional

from config import CONFIG
from logger_config import logger

RESULTS_FILENAME = "evaluation_results.json"


def _resolve_experiment_nums(experiments_dir: str, experiment_nums: Optional[List[int]]) -> List[int]:
    if experiment_nums is not None:
        return experiment_nums

    if not os.path.exists(experiments_dir):
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")

    subdirs = [d for d in os.listdir(experiments_dir)
               if os.path.isdir(os.path.join(experiments_dir, d)) and d.isdigit()]
    if not subdirs:
        raise ValueError(f"No global experiment number directories found in {experiments_dir}")

    resolved = sorted(int(d) for d in subdirs)
    logger.info(f"No experiment_nums specified, pulling all found global experiments: {resolved}")
    return resolved


def _find_evaluation_results(experiment_dir: str) -> List[str]:
    """Recursively find every evaluation_results.json under experiment_dir."""
    found = []
    for root, _dirs, files in os.walk(experiment_dir):
        if RESULTS_FILENAME in files:
            found.append(os.path.join(root, RESULTS_FILENAME))
    return sorted(found)


def run_pull_raw_results(experiment_nums: Optional[List[int]] = None,
                          output_dir: str = "raw_results_pull") -> int:
    """
    Copy evaluation_results.json files for the indicated experiments into output_dir,
    preserving the original directory structure rooted at the experiments directory.

    Args:
        experiment_nums: Global experiment numbers to pull (the top-level folder names
                          under the experiments directory, e.g. [9, 12, 19]). If None,
                          every global experiment number found is pulled.
        output_dir: Root folder the results are copied into (default: "raw_results_pull").

    Returns:
        Total number of evaluation_results.json files copied.
    """
    paths_config = CONFIG.get("paths", {})
    experiments_dir = paths_config.get("experiments", "experiments")

    resolved_nums = _resolve_experiment_nums(experiments_dir, experiment_nums)

    total_copied = 0
    for exp_num in resolved_nums:
        experiment_dir = os.path.join(experiments_dir, str(exp_num))
        if not os.path.exists(experiment_dir):
            logger.warning(f"Experiment directory not found, skipping: {experiment_dir}")
            continue

        result_files = _find_evaluation_results(experiment_dir)
        if not result_files:
            logger.warning(f"No {RESULTS_FILENAME} files found under {experiment_dir}")
            continue

        for src_path in result_files:
            # Preserve the path relative to the experiments dir, e.g.
            # experiments/9/banking77_default_10/default_head/evaluation_results.json
            # -> raw_results_pull/9/banking77_default_10/default_head/evaluation_results.json
            rel_path = os.path.relpath(src_path, experiments_dir)
            dest_path = os.path.join(output_dir, rel_path)

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)
            total_copied += 1

        logger.info(f"Pulled {len(result_files)} {RESULTS_FILENAME} file(s) from global experiment {exp_num}")

    logger.info(f"Done. Copied {total_copied} {RESULTS_FILENAME} file(s) into {os.path.abspath(output_dir)}")
    return total_copied
