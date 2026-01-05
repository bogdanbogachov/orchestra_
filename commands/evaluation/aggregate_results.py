"""
Aggregate evaluation results across multiple experiment runs.

This module aggregates metrics from evaluation_results.json files across
all experiment runs, groups them by experiment type, and generates
publication-quality tables and charts following ACL/NeurIPS/EMNLP standards.

Note: The first run of each experiment type is excluded from aggregation
as it may be a cold start that takes significantly longer to complete.
"""

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from config import CONFIG
from logger_config import logger

# Set publication-quality matplotlib settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 13
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1

# Set seaborn style for publication-quality plots
sns.set_style("whitegrid")
sns.set_palette("husl")


def parse_experiment_configs(config_path: str = "experiment_configs.sh") -> List[Tuple[str, str]]:
    """
    Parse experiment_configs.sh to extract base experiment names and their order.
    
    Returns:
        List of tuples (base_name, eval_head) in the order they appear in the config.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config file not found: {config_path}")
    
    base_names = []
    seen = set()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all add_experiments calls
    pattern = r'add_experiments\s+"([^"]+)"\s+"([^"]+)"'
    matches = re.findall(pattern, content)
    
    for base_name, eval_head in matches:
        # Only add unique base names in order
        if base_name not in seen:
            base_names.append((base_name, eval_head))
            seen.add(base_name)
    
    logger.info(f"Parsed {len(base_names)} experiment types from {config_path}")
    return base_names


def extract_base_name(experiment_name: str) -> Optional[str]:
    """
    Extract base experiment name from full experiment name.
    
    Example: "35_l_default_6_1" -> "35_l_default"
    Pattern: base_name_global_exp_num_per_config_exp_num
    """
    # Pattern: base_name_global_exp_num_per_config_exp_num
    # We want to extract everything before the last two numbers (separated by underscores)
    # Match: any characters, then _number_number at the end
    match = re.match(r'^(.+)_(\d+)_(\d+)$', experiment_name)
    if match:
        return match.group(1)
    return None


def extract_global_experiment_number(experiment_name: str) -> Optional[int]:
    """
    Extract global experiment number from full experiment name.
    
    Example: "35_l_default_8_1" -> 8
    Pattern: base_name_global_exp_num_per_config_exp_num
    """
    match = re.match(r'^(.+)_(\d+)_(\d+)$', experiment_name)
    if match:
        return int(match.group(2))  # global_exp_num is the second-to-last number
    return None


def extract_run_number(experiment_name: str) -> Optional[int]:
    """
    Extract per-config experiment number (run number) from full experiment name.
    
    Example: "35_l_default_6_1" -> 1
    Pattern: base_name_global_exp_num_per_config_exp_num
    """
    match = re.match(r'^(.+)_(\d+)_(\d+)$', experiment_name)
    if match:
        return int(match.group(3))
    return None


def load_evaluation_results(experiment_dir: str, head: str, model_type: str = "best") -> Optional[Dict[str, Any]]:
    """
    Load evaluation results from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        head: "default_head" or "custom_head"
        model_type: "best" for best model, or "milestone" for milestone model
    
    Returns:
        Evaluation results dictionary or None if file doesn't exist
    """
    if model_type == "best":
        results_path = os.path.join(experiment_dir, head, "evaluation_results.json")
    elif model_type == "milestone":
        # Find milestone results - look for milestone_*_percent in the results
        head_dir = os.path.join(experiment_dir, head)
        import glob
        milestone_pattern = os.path.join(head_dir, "evaluation_results.json")
        # We'll check if the results file contains milestone data
        results_path = milestone_pattern
    else:
        results_path = os.path.join(experiment_dir, head, "evaluation_results.json")
    
    if not os.path.exists(results_path):
        return None
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
            # If loading milestone, extract milestone data if available
            if model_type == "milestone" and "milestone_95_percent" in results:
                # Return milestone results as primary
                milestone_data = results["milestone_95_percent"].copy()
                # Keep common fields
                milestone_data["experiment"] = results.get("experiment")
                milestone_data["head"] = results.get("head")
                milestone_data["test_samples"] = results.get("test_samples")
                return milestone_data
            elif model_type == "milestone":
                # Try to find any milestone data (could be different threshold)
                for key in results.keys():
                    if key.startswith("milestone_") and key.endswith("_percent"):
                        milestone_data = results[key].copy()
                        milestone_data["experiment"] = results.get("experiment")
                        milestone_data["head"] = results.get("head")
                        milestone_data["test_samples"] = results.get("test_samples")
                        return milestone_data
                return None
            
            return results
    except Exception as e:
        logger.warning(f"Failed to load {results_path}: {e}")
        return None


def extract_flat_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract all numeric metrics from evaluation results into a flat dictionary.
    
    Handles nested structures like inference_metrics and training_metrics.
    """
    metrics = {}
    
    # Top-level metrics
    for key in ['accuracy', 'precision', 'recall', 'f1', 'avg_latency_ms', 'std_latency_ms']:
        if key in results:
            metrics[key] = float(results[key])
    
    # Inference metrics
    if 'inference_metrics' in results:
        inf_metrics = results['inference_metrics']
        for key, value in inf_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f'inference_{key}'] = float(value)
            elif isinstance(value, dict):
                # Handle nested dicts like energy_consumption
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        metrics[f'inference_{key}_{sub_key}'] = float(sub_value)
    
    # Training metrics
    if 'training_metrics' in results:
        train_metrics = results['training_metrics']
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                metrics[f'training_{key}'] = float(value)
            elif isinstance(value, dict):
                # Handle nested dicts
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        metrics[f'training_{key}_{sub_key}'] = float(sub_value)
    
    return metrics


def aggregate_metrics(experiments_dir: str, base_names: List[Tuple[str, str]], 
                     include_milestone: bool = True,
                     global_exp_num: Optional[int] = None) -> Dict[Tuple[str, int], Dict[str, List[float]]]:
    """
    Aggregate metrics across all runs of each experiment type, separated by global experiment number.
    Excludes the first run (cold start) from each experiment type.
    Can aggregate both best models and milestone models.
    
    Args:
        experiments_dir: Path to experiments directory
        base_names: List of (base_name, eval_head) tuples
        include_milestone: If True, also aggregate milestone model results
        global_exp_num: If specified, only aggregate experiments from this global experiment number.
                       If None, aggregate all global experiment numbers separately.
    
    Returns:
        Dictionary mapping (base_name, global_exp_num) -> metric_name -> list of values
    """
    aggregated = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(experiments_dir):
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
    
    # First pass: collect all runs with their run numbers, grouped by base_name
    # Handle nested structure: experiments/global_exp_num/experiment_name
    runs_by_base = defaultdict(list)  # base_name -> [(run_number, item, eval_head, exp_path, global_exp_num), ...]
    
    # Check if we have nested structure (experiments/global_exp_num/experiment_name)
    has_nested_structure = False
    for item in os.listdir(experiments_dir):
        item_path = os.path.join(experiments_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            has_nested_structure = True
            break
    
    if has_nested_structure:
        # Nested structure: experiments/global_exp_num/experiment_name
        logger.info("Detected nested directory structure: experiments/global_exp_num/experiment_name")
        for global_exp_num_dir in os.listdir(experiments_dir):
            global_exp_num_path = os.path.join(experiments_dir, global_exp_num_dir)
            if not os.path.isdir(global_exp_num_path) or not global_exp_num_dir.isdigit():
                continue
            
            # Filter by global_exp_num if specified
            current_global_exp_num = int(global_exp_num_dir)
            if global_exp_num is not None and current_global_exp_num != global_exp_num:
                continue
            
            for item in os.listdir(global_exp_num_path):
                exp_path = os.path.join(global_exp_num_path, item)
                if not os.path.isdir(exp_path):
                    continue
                
                base_name = extract_base_name(item)
                if base_name is None:
                    continue
                
                run_number = extract_run_number(item)
                if run_number is None:
                    continue
                
                # Find matching base name and eval head
                matching_config = None
                for bname, eval_head in base_names:
                    if bname == base_name:
                        matching_config = (bname, eval_head)
                        break
                
                if matching_config is None:
                    continue
                
                bname, eval_head = matching_config
                current_global_exp_num = int(global_exp_num_dir)
                # Group by both base_name and global_exp_num
                key = (bname, current_global_exp_num)
                runs_by_base[key].append((run_number, item, eval_head, exp_path, current_global_exp_num))
    else:
        # Flat structure: experiments/experiment_name (backward compatibility)
        logger.info("Using flat directory structure: experiments/experiment_name")
        for item in os.listdir(experiments_dir):
            exp_path = os.path.join(experiments_dir, item)
            if not os.path.isdir(exp_path):
                continue
            
            base_name = extract_base_name(item)
            if base_name is None:
                continue
            
            run_number = extract_run_number(item)
            if run_number is None:
                continue
            
            # Find matching base name and eval head
            matching_config = None
            for bname, eval_head in base_names:
                if bname == base_name:
                    matching_config = (bname, eval_head)
                    break
            
            if matching_config is None:
                continue
            
            bname, eval_head = matching_config
            # For flat structure, use None as global_exp_num
            key = (bname, None)
            runs_by_base[key].append((run_number, item, eval_head, exp_path, None))
    
    # Second pass: sort by run number, exclude first run, then aggregate
    for key, runs in runs_by_base.items():
        bname, global_exp_num_key = key
        # Sort by run number
        runs.sort(key=lambda x: x[0])
        
        # Exclude the first run (cold start)
        if len(runs) > 1:
            runs_to_process = runs[1:]  # Skip first run
            global_exp_str = f" (global exp {global_exp_num_key})" if global_exp_num_key is not None else ""
            logger.info(f"Excluding first run (cold start) for {bname}{global_exp_str}. Processing {len(runs_to_process)} runs out of {len(runs)} total.")
        else:
            runs_to_process = runs
            global_exp_str = f" (global exp {global_exp_num_key})" if global_exp_num_key is not None else ""
            logger.warning(f"Only {len(runs)} run found for {bname}{global_exp_str}, cannot exclude cold start.")
        
        # Process remaining runs - aggregate best models
        for run_data in runs_to_process:
            if len(run_data) == 5:
                run_number, item, eval_head, exp_path, global_exp_num = run_data
            else:
                run_number, item, eval_head, exp_path = run_data[:4]
            
            # Load evaluation results for best model
            results = load_evaluation_results(exp_path, eval_head, model_type="best")
            if results is None:
                logger.debug(f"No evaluation results found for {item}/{eval_head} (best model)")
            else:
                # Extract metrics
                metrics = extract_flat_metrics(results)
                
                # Add to aggregation - use key (bname, global_exp_num) instead of just bname
                # Use "best_" prefix when milestone tracking is enabled for clarity
                # Otherwise use unprefixed names for backward compatibility
                for metric_name, value in metrics.items():
                    if include_milestone:
                        aggregated[key][f"best_{metric_name}"].append(value)
                    else:
                        aggregated[key][metric_name].append(value)
                
                logger.debug(f"Loaded best model metrics from {item}/{eval_head} (run {run_number})")
            
            # Also load milestone model results if available
            if include_milestone:
                milestone_results = load_evaluation_results(exp_path, eval_head, model_type="milestone")
                if milestone_results is not None:
                    milestone_metrics = extract_flat_metrics(milestone_results)
                    
                    # Add to aggregation with "milestone_" prefix
                    for metric_name, value in milestone_metrics.items():
                        aggregated[key][f"milestone_{metric_name}"].append(value)
                    
                    logger.debug(f"Loaded milestone model metrics from {item}/{eval_head} (run {run_number})")
    
    # Log summary
    for key in aggregated:
        bname, global_exp_num_key = key
        # Count best model runs (check both prefixed and unprefixed)
        best_runs = len(aggregated[key].get('accuracy', aggregated[key].get('best_accuracy', [])))
        milestone_runs = len(aggregated[key].get('milestone_accuracy', []))
        global_exp_str = f" (global exp {global_exp_num_key})" if global_exp_num_key is not None else ""
        if milestone_runs > 0:
            logger.info(f"Aggregated {best_runs} best model runs and {milestone_runs} milestone model runs for {bname}{global_exp_str} (excluding cold start)")
        else:
            logger.info(f"Aggregated {best_runs} runs for {bname}{global_exp_str} (excluding cold start)")
    
    return aggregated


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean and standard deviation for a list of values."""
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'count': 0}
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)),  # Sample standard deviation
        'count': len(values)
    }


def create_aggregation_table(aggregated: Dict[Tuple[str, Optional[int]], Dict[str, List[float]]], 
                            base_names: List[Tuple[str, str]],
                            global_exp_num: Optional[int] = None) -> pd.DataFrame:
    """
    Create a pandas DataFrame with aggregated metrics.
    
    Each row is an experiment type, each column is a metric (mean ± std).
    Handles both best and milestone model metrics.
    """
    # Collect all metric names
    all_metrics = set()
    for bname in aggregated:
        all_metrics.update(aggregated[bname].keys())
    
    # Filter to key metrics for the main table
    # Include both best and milestone versions
    base_key_metrics = [
        'accuracy', 'precision', 'recall', 'f1',
        'avg_latency_ms', 'std_latency_ms',
        'inference_flops_per_sample', 'inference_peak_memory_mb',
    ]
    
    # Build key metrics list - include both best_ and milestone_ prefixed versions
    key_metrics = []
    for metric in base_key_metrics:
        if f"best_{metric}" in all_metrics:
            key_metrics.append(f"best_{metric}")
        if f"milestone_{metric}" in all_metrics:
            key_metrics.append(f"milestone_{metric}")
        # Also include unprefixed if exists (backward compatibility)
        if metric in all_metrics and f"best_{metric}" not in all_metrics:
            key_metrics.append(metric)
    
    # Add training metrics if available (both best and milestone)
    training_metrics = [m for m in all_metrics if 'training_' in m]
    training_key_metrics = [m for m in training_metrics if 'flops' in m or 'memory' in m]
    key_metrics.extend(training_key_metrics)
    
    # Filter to metrics that exist
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    # Build table data - group by base_name and global_exp_num
    table_data = []
    for bname, _ in base_names:
        # Find all entries for this base_name
        matching_keys = [k for k in aggregated.keys() if k[0] == bname]
        # Sort by global_exp_num for consistent ordering
        matching_keys.sort(key=lambda x: (x[1] if x[1] is not None else float('inf'), x[0]))
        
        for key in matching_keys:
            bname_key, global_exp_num_key = key
            exp_label = f"{bname_key}"
            if global_exp_num_key is not None:
                exp_label += f" (G{global_exp_num_key})"
            
            row = {'Experiment': exp_label}
            for metric in key_metrics:
                if metric in aggregated[key]:
                    stats = compute_statistics(aggregated[key][metric])
                    # Format as mean ± std
                    if stats['count'] > 0:
                        row[metric] = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
                    else:
                        row[metric] = "N/A"
                else:
                    row[metric] = "N/A"
            
            table_data.append(row)
    
    df = pd.DataFrame(table_data)
    return df


def create_charts(aggregated: Dict[Tuple[str, Optional[int]], Dict[str, List[float]]], 
                 base_names: List[Tuple[str, str]], 
                 output_dir: str,
                 global_exp_num: Optional[int] = None):
    """
    Create publication-quality charts for each metric.
    
    X-axis: experiment type (in order from config)
    Y-axis: metric value
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all metric names
    all_metrics = set()
    for bname in aggregated:
        all_metrics.update(aggregated[bname].keys())
    
    # Filter to key metrics for charts
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1',
        'avg_latency_ms', 'inference_flops_per_sample', 'inference_peak_memory_mb',
    ]
    
    # Add training metrics if available
    training_metrics = [m for m in all_metrics if m.startswith('training_')]
    key_metrics.extend([m for m in training_metrics if 'flops' in m or 'memory' in m])
    
    # Filter to metrics that exist
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    # Create charts for each metric
    for metric in key_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        experiment_names = []
        means = []
        stds = []
        
        for bname, _ in base_names:
            # Find all entries for this base_name
            matching_keys = [k for k in aggregated.keys() if k[0] == bname]
            # Sort by global_exp_num for consistent ordering
            matching_keys.sort(key=lambda x: (x[1] if x[1] is not None else float('inf'), x[0]))
            
            for key in matching_keys:
                if metric not in aggregated[key]:
                    continue
                
                stats = compute_statistics(aggregated[key][metric])
                if stats['count'] > 0:
                    bname_key, global_exp_num_key = key
                    exp_label = f"{bname_key}"
                    if global_exp_num_key is not None:
                        exp_label += f" (G{global_exp_num_key})"
                    experiment_names.append(exp_label)
                    means.append(stats['mean'])
                    stds.append(stats['std'])
        
        if not experiment_names:
            plt.close(fig)
            continue
        
        # Adjust figure size based on number of experiments
        num_experiments = len(experiment_names)
        fig_width = max(12, num_experiments * 1.2)
        fig.set_size_inches(fig_width, 6)
        
        # Create bar plot with error bars
        x_pos = np.arange(len(experiment_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     edgecolor='black', linewidth=1.2)
        
        # Customize plot
        ax.set_xlabel('Experiment Type', fontweight='bold')
        ax.set_ylabel(_format_metric_name(metric), fontweight='bold')
        ax.set_title(f'{_format_metric_name(metric)} Across Experiments', 
                    fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        y_max = max(m + s for m, s in zip(means, stds)) if means else 0
        y_range = max(means) - min(means) if means else 1
        label_offset = max(y_range * 0.02, y_max * 0.01) if y_range > 0 else 0.01
        
        for i, (mean, std) in enumerate(zip(means, stds)):
            label_y = mean + std + label_offset
            ax.text(i, label_y, f'{mean:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        safe_metric_name = metric.replace('/', '_').replace(' ', '_')
        output_path = os.path.join(output_dir, f'{safe_metric_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved chart: {output_path}")
        plt.close(fig)


def _format_metric_name(metric: str) -> str:
    """Format metric name for display in charts and tables."""
    # Handle best_ and milestone_ prefixes
    is_best = metric.startswith('best_')
    is_milestone = metric.startswith('milestone_')
    
    if is_best:
        metric = metric[5:]  # Remove 'best_' prefix
        prefix = "Best "
    elif is_milestone:
        metric = metric[11:]  # Remove 'milestone_' prefix
        prefix = "Milestone "
    else:
        prefix = ""
    
    # Replace underscores with spaces and capitalize
    formatted = metric.replace('_', ' ').title()
    
    # Special formatting
    replacements = {
        'Ms': 'ms',
        'F1': 'F1',
        'Flops': 'FLOPs',
        'Mb': 'MB',
        'Kwh': 'kWh',
        'Gco2Eq': 'gCO₂eq',
    }
    
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return prefix + formatted


def save_latex_table(df: pd.DataFrame, output_path: str):
    """Save DataFrame as LaTeX table for publication."""
    # Format column names
    df_formatted = df.copy()
    df_formatted.columns = [_format_metric_name(col) for col in df_formatted.columns]
    
    # Replace ± with \pm for LaTeX
    df_formatted = df_formatted.replace('±', r'$\pm$', regex=False)
    
    # Generate LaTeX
    latex_str = df_formatted.to_latex(index=False, escape=False, longtable=False)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    logger.info(f"Saved LaTeX table: {output_path}")


def run_aggregate_results(experiment_configs_path: str = "experiment_configs.sh",
                         output_dir: str = "experiment_aggregations",
                         global_exp_num: Optional[int] = None):
    """
    Main function to aggregate evaluation results across experiments.
    
    Args:
        experiment_configs_path: Path to experiment_configs.sh
        output_dir: Directory to save aggregated results, tables, and charts
        global_exp_num: If specified, only aggregate experiments from this global experiment number.
                       If None, aggregate all global experiment numbers separately.
    """
    logger.info("=" * 100)
    logger.info("AGGREGATING EXPERIMENT RESULTS")
    if global_exp_num is not None:
        logger.info(f"Filtering to global experiment number: {global_exp_num}")
    logger.info("=" * 100)
    
    # Get experiments directory from config
    paths_config = CONFIG.get("paths", {})
    experiments_dir = paths_config.get("experiments", "experiments")
    
    # Parse experiment configs
    logger.info(f"Parsing experiment configs from {experiment_configs_path}")
    base_names = parse_experiment_configs(experiment_configs_path)
    
    if not base_names:
        raise ValueError("No experiment types found in config file")
    
    # Aggregate metrics (include both best and milestone models)
    logger.info(f"Scanning experiments directory: {experiments_dir}")
    aggregated = aggregate_metrics(experiments_dir, base_names, include_milestone=True, global_exp_num=global_exp_num)
    
    if not aggregated:
        raise ValueError("No evaluation results found. Run experiments first.")
    
    # Create output directory - include global_exp_num in path if specified
    if global_exp_num is not None:
        output_dir = os.path.join(output_dir, f"global_exp_{global_exp_num}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create aggregation table
    logger.info("Creating aggregation table...")
    df = create_aggregation_table(aggregated, base_names, global_exp_num=global_exp_num)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "aggregated_metrics.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV table: {csv_path}")
    
    # Save as LaTeX
    latex_path = os.path.join(output_dir, "aggregated_metrics.tex")
    save_latex_table(df, latex_path)
    
    # Print table to console
    logger.info("\n" + "=" * 100)
    logger.info("AGGREGATED METRICS TABLE")
    logger.info("=" * 100)
    print(df.to_string(index=False))
    
    # Create charts
    logger.info("\nCreating charts...")
    charts_dir = os.path.join(output_dir, "charts")
    create_charts(aggregated, base_names, charts_dir, global_exp_num=global_exp_num)
    
    # Save detailed JSON with all statistics
    detailed_stats = {}
    for key in aggregated:
        bname, global_exp_num_key = key
        key_str = f"{bname}"
        if global_exp_num_key is not None:
            key_str += f"_G{global_exp_num_key}"
        detailed_stats[key_str] = {}
        for metric in aggregated[key]:
            detailed_stats[key_str][metric] = compute_statistics(aggregated[key][metric])
    
    json_path = os.path.join(output_dir, "detailed_statistics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_stats, f, indent=2)
    logger.info(f"Saved detailed statistics: {json_path}")
    
    logger.info("\n" + "=" * 100)
    logger.info(f"✓ Aggregation complete! Results saved to {output_dir}")
    logger.info("=" * 100)

