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
    
    Example: "35_l_default_10" -> "35_l_default"
    Pattern: base_name_per_config_exp_num (after removing global_exp_num from path)
    """
    # Pattern: base_name_per_config_exp_num
    # We want to extract everything before the last number
    match = re.match(r'^(.+)_(\d+)$', experiment_name)
    if match:
        return match.group(1)
    return None


def extract_run_number(experiment_name: str) -> Optional[int]:
    """
    Extract per-config experiment number (run number) from full experiment name.
    
    Example: "35_l_default_10" -> 10
    Pattern: base_name_per_config_exp_num
    """
    match = re.match(r'^(.+)_(\d+)$', experiment_name)
    if match:
        return int(match.group(2))
    return None


def extract_global_exp_num(experiment_name: str) -> Optional[int]:
    """
    Extract global experiment number from full experiment name.
    
    Example: "35_l_default_9_10" -> 9
    Pattern: base_name_global_exp_num_per_config_exp_num
    """
    match = re.match(r'^(.+)_(\d+)_(\d+)$', experiment_name)
    if match:
        return int(match.group(2))
    return None


def load_evaluation_results(experiment_dir: str, head: str) -> Optional[Dict[str, Any]]:
    """
    Load evaluation results from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        head: "default_head" or "custom_head"
    
    Returns:
        Evaluation results dictionary or None if file doesn't exist
    """
    results_path = os.path.join(experiment_dir, head, "evaluation_results.json")
    if not os.path.exists(results_path):
        return None
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
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


def aggregate_metrics(experiments_dir: str, base_names: List[Tuple[str, str]], global_exp_num: Optional[int] = None) -> Dict[str, Dict[str, List[float]]]:
    """
    Aggregate metrics across all runs of each experiment type.
    Excludes the first run (cold start) from each experiment type.
    
    Args:
        experiments_dir: Path to experiments directory
        base_names: List of (base_name, eval_head) tuples
        global_exp_num: Optional global experiment number to filter by. If None, uses first found.
    
    Returns:
        Dictionary mapping base_name -> metric_name -> list of values
    """
    aggregated = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(experiments_dir):
        raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
    
    # Determine which global_exp_num to use
    if global_exp_num is None:
        # Find the first available global_exp_num directory
        subdirs = [d for d in os.listdir(experiments_dir) 
                   if os.path.isdir(os.path.join(experiments_dir, d)) and d.isdigit()]
        if not subdirs:
            raise ValueError(f"No global experiment number directories found in {experiments_dir}")
        global_exp_num = int(sorted(subdirs)[0])
        logger.info(f"No global_exp_num specified, using first available: {global_exp_num}")
    
    # Use the global_exp_num subdirectory
    global_exp_dir = os.path.join(experiments_dir, str(global_exp_num))
    if not os.path.exists(global_exp_dir):
        raise FileNotFoundError(f"Global experiment directory not found: {global_exp_dir}")
    
    logger.info(f"Aggregating results from global experiment number: {global_exp_num}")
    
    # First pass: collect all runs with their run numbers, grouped by base_name
    runs_by_base = defaultdict(list)  # base_name -> [(run_number, item, eval_head, exp_path), ...]
    
    for item in os.listdir(global_exp_dir):
        exp_path = os.path.join(global_exp_dir, item)
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
        runs_by_base[bname].append((run_number, item, eval_head, exp_path))
    
    # Second pass: sort by run number, exclude first run, then aggregate
    for bname, runs in runs_by_base.items():
        # Sort by run number
        runs.sort(key=lambda x: x[0])
        
        # Exclude the first run (cold start)
        if len(runs) > 1:
            runs_to_process = runs[1:]  # Skip first run
            logger.info(f"Excluding first run (cold start) for {bname}. Processing {len(runs_to_process)} runs out of {len(runs)} total.")
        else:
            runs_to_process = runs
            logger.warning(f"Only {len(runs)} run found for {bname}, cannot exclude cold start.")
        
        # Process remaining runs
        for run_number, item, eval_head, exp_path in runs_to_process:
            # Load evaluation results
            results = load_evaluation_results(exp_path, eval_head)
            if results is None:
                logger.debug(f"No evaluation results found for {item}/{eval_head}")
                continue
            
            # Extract metrics
            metrics = extract_flat_metrics(results)
            
            # Add to aggregation
            for metric_name, value in metrics.items():
                aggregated[bname][metric_name].append(value)
            
            logger.debug(f"Loaded metrics from {item}/{eval_head} (run {run_number})")
    
    # Log summary
    for bname in aggregated:
        total_runs = len(aggregated[bname].get('accuracy', []))
        logger.info(f"Aggregated {total_runs} runs for {bname} (excluding cold start)")
    
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


def create_aggregation_table(aggregated: Dict[str, Dict[str, List[float]]], 
                            base_names: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with aggregated metrics.
    
    Each row is an experiment type, each column is a metric (mean ± std).
    """
    # Collect all metric names
    all_metrics = set()
    for bname in aggregated:
        all_metrics.update(aggregated[bname].keys())
    
    # Filter to key metrics for the main table
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1',
        'avg_latency_ms', 'std_latency_ms',
        'inference_flops_per_sample', 'inference_peak_memory_mb',
    ]
    
    # Add training metrics if available
    training_metrics = [m for m in all_metrics if m.startswith('training_')]
    key_metrics.extend([m for m in training_metrics if 'flops' in m or 'memory' in m])
    
    # Filter to metrics that exist
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    # Build table data
    table_data = []
    for bname, _ in base_names:
        if bname not in aggregated:
            continue
        
        row = {'Experiment': bname}
        for metric in key_metrics:
            if metric in aggregated[bname]:
                stats = compute_statistics(aggregated[bname][metric])
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


def create_charts(aggregated: Dict[str, Dict[str, List[float]]], 
                 base_names: List[Tuple[str, str]], 
                 output_dir: str):
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
            if bname not in aggregated or metric not in aggregated[bname]:
                continue
            
            stats = compute_statistics(aggregated[bname][metric])
            if stats['count'] > 0:
                experiment_names.append(bname)
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
    
    return formatted


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
        global_exp_num: Optional global experiment number to aggregate. If None, uses first available.
    """
    logger.info("=" * 100)
    logger.info("AGGREGATING EXPERIMENT RESULTS")
    logger.info("=" * 100)
    
    # Get experiments directory from config
    paths_config = CONFIG.get("paths", {})
    experiments_dir = paths_config.get("experiments", "experiments")
    
    # Parse experiment configs
    logger.info(f"Parsing experiment configs from {experiment_configs_path}")
    base_names = parse_experiment_configs(experiment_configs_path)
    
    if not base_names:
        raise ValueError("No experiment types found in config file")
    
    # Aggregate metrics
    logger.info(f"Scanning experiments directory: {experiments_dir}")
    aggregated = aggregate_metrics(experiments_dir, base_names, global_exp_num=global_exp_num)
    
    if not aggregated:
        raise ValueError("No evaluation results found. Run experiments first.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create aggregation table
    logger.info("Creating aggregation table...")
    df = create_aggregation_table(aggregated, base_names)
    
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
    create_charts(aggregated, base_names, charts_dir)
    
    # Save detailed JSON with all statistics
    detailed_stats = {}
    for bname in aggregated:
        detailed_stats[bname] = {}
        for metric in aggregated[bname]:
            detailed_stats[bname][metric] = compute_statistics(aggregated[bname][metric])
    
    json_path = os.path.join(output_dir, "detailed_statistics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_stats, f, indent=2)
    logger.info(f"Saved detailed statistics: {json_path}")
    
    logger.info("\n" + "=" * 100)
    logger.info(f"✓ Aggregation complete! Results saved to {output_dir}")
    logger.info("=" * 100)

