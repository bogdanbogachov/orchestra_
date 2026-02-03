"""
Aggregate evaluation results across multiple experiment runs.

This module aggregates metrics from evaluation_results.json files across
all experiment runs, groups them by experiment type, and generates
publication-quality tables and charts following ACL/NeurIPS/EMNLP standards.
"""

import gc
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to reduce memory usage
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
rcParams['figure.dpi'] = 200
rcParams['savefig.dpi'] = 200
rcParams['savefig.bbox'] = 'standard'  # Don't use 'tight' to avoid memory issues
rcParams['savefig.pad_inches'] = 0.1

# Set seaborn style for publication-quality plots
sns.set_style("whitegrid")
sns.set_palette("husl")


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


def aggregate_metrics(experiments_dir: str, global_exp_num: Optional[int] = None) -> Tuple[Dict[str, Dict[str, List[float]]], int]:
    """
    Aggregate metrics across all runs of each experiment type.
    
    Args:
        experiments_dir: Path to experiments directory
        global_exp_num: Optional global experiment number to filter by. If None, uses first found.
    
    Returns:
        Tuple of (aggregated metrics dictionary, global_exp_num used)
        Aggregated dict structure: {experiment_name: {metric_name: [values...]}}
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
    
    # Collect all experiment directories and their results
    experiments_found = []
    
    for item in os.listdir(global_exp_dir):
        exp_path = os.path.join(global_exp_dir, item)
        if not os.path.isdir(exp_path):
            continue
        
        # Try both heads
        for head in ["default_head", "custom_head"]:
            results = load_evaluation_results(exp_path, head)
            if results is None:
                continue
            
            # Extract metrics
            metrics = extract_flat_metrics(results)
            
            if not metrics:
                continue
            
            # Use experiment name as key (item is the experiment directory name)
            # Group by base name if we can extract it, otherwise use full name
            base_name = extract_base_name(item)
            if base_name:
                # Group runs of the same base experiment together
                experiment_key = base_name
            else:
                # Use full experiment name if we can't extract base name
                experiment_key = item
            
            # Add to aggregation
            for metric_name, value in metrics.items():
                aggregated[experiment_key][metric_name].append(value)
            
            experiments_found.append((item, head))
            logger.debug(f"Loaded metrics from {item}/{head}")
    
    if not experiments_found:
        logger.warning(f"No evaluation results found in {global_exp_dir}")
        return aggregated, global_exp_num
    
    # Log summary
    for exp_name in aggregated:
        total_runs = len(aggregated[exp_name].get('accuracy', []))
        logger.info(f"Aggregated {total_runs} runs for {exp_name}")
    
    return aggregated, global_exp_num


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


def get_experiment_sort_key(exp_name: str) -> int:
    """
    Get sort key for experiment name to order them in a specific sequence.
    
    Order: default, custom last, custom max, custom mean, custom attention,
           custom fft last, custom fft max, custom fft mean, custom fft attention
    
    Returns a sort key (lower number = earlier in order).
    """
    exp_lower = exp_name.lower()
    
    # Check for default (not custom)
    if 'default' in exp_lower and 'custom' not in exp_lower:
        return 1
    
    # Check for FFT experiments (they should come after non-FFT)
    has_fft = 'fft' in exp_lower
    
    # Base value: 2 for non-FFT custom, 6 for FFT custom
    base = 6 if has_fft else 2
    
    # Add offset based on pooling strategy
    if 'last' in exp_lower:
        return base + 0  # 2 or 6
    elif 'max' in exp_lower:
        return base + 1  # 3 or 7
    elif 'mean' in exp_lower:
        return base + 2  # 4 or 8
    elif 'attention' in exp_lower:
        return base + 3  # 5 or 9
    
    # Default: put unknown patterns at the end
    return 100


def create_aggregation_table(aggregated: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with aggregated metrics.
    
    Each row is an experiment type, each column is a metric (mean ± std).
    """
    # Collect all metric names
    all_metrics = set()
    for exp_name in aggregated:
        all_metrics.update(aggregated[exp_name].keys())
    
    # Filter to key metrics for the main table
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1',
        'avg_latency_ms', 'std_latency_ms',
        'inference_flops_per_sample', 'inference_peak_memory_mb',
        'inference_energy_consumption_energy_consumed_kwh',  # Total inference energy
        'inference_carbon_footprint_emissions_gco2eq',  # Total inference carbon footprint
    ]
    
    # Add training metrics if available
    training_metrics = [m for m in all_metrics if m.startswith('training_')]
    key_metrics.extend([m for m in training_metrics if 'flops' in m or 'memory' in m])
    
    # Add training energy and carbon metrics if available
    key_metrics.append('training_energy_consumption_energy_consumed_kwh')  # Total training energy
    key_metrics.append('training_carbon_footprint_emissions_gco2eq')  # Total training carbon footprint
    
    # Filter to metrics that exist
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    # Build table data - sort experiment names using custom order
    table_data = []
    sorted_exp_names = sorted(aggregated.keys(), key=get_experiment_sort_key)
    for exp_name in sorted_exp_names:
        row = {'Experiment': exp_name}
        for metric in key_metrics:
            if metric in aggregated[exp_name]:
                stats = compute_statistics(aggregated[exp_name][metric])
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


def create_charts(aggregated: Dict[str, Dict[str, List[float]]], output_dir: str):
    """
    Create publication-quality charts for each metric.
    
    X-axis: experiment type (sorted alphabetically)
    Y-axis: metric value
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all metric names
    all_metrics = set()
    for exp_name in aggregated:
        all_metrics.update(aggregated[exp_name].keys())
    
    # Filter to key metrics for charts
    key_metrics = [
        'accuracy', 'precision', 'recall', 'f1',
        'avg_latency_ms', 'inference_flops_per_sample', 'inference_peak_memory_mb',
    ]
    
    # Add training metrics if available - limit to essential ones to reduce memory
    training_metrics = [m for m in all_metrics if m.startswith('training_')]
    # Only include total_flops and peak_memory, skip per-sample and detailed memory
    essential_training = [
        m for m in training_metrics 
        if ('total_flops' in m or m == 'training_peak_memory_mb')
        and 'per_sample' not in m 
        and 'memory_info' not in m
    ]
    key_metrics.extend(essential_training)
    
    # Filter to metrics that exist
    key_metrics = [m for m in key_metrics if m in all_metrics]
    
    logger.info(f"Creating {len(key_metrics)} charts...")
    
    # Create charts for each metric
    for metric_idx, metric in enumerate(key_metrics):
        try:
            experiment_names = []
            means = []
            stds = []
            
            # Sort experiment names using custom order
            sorted_exp_names = sorted(aggregated.keys(), key=get_experiment_sort_key)
            for exp_name in sorted_exp_names:
                if metric not in aggregated[exp_name]:
                    continue
                
                stats = compute_statistics(aggregated[exp_name][metric])
                if stats['count'] > 0:
                    experiment_names.append(exp_name)
                    means.append(stats['mean'])
                    stds.append(stats['std'])
            
            if not experiment_names:
                continue
            
            # Calculate figure size more conservatively to reduce memory
            num_experiments = len(experiment_names)
            # More aggressive size reduction: max 12 inches, smaller multiplier
            fig_width = min(max(8, num_experiments * 0.8), 12)
            fig_height = 6  # Reduced from 7
            
            # Create figure with calculated size
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            # Create line plot with error bars
            x_pos = np.arange(len(experiment_names))
            ax.errorbar(x_pos, means, yerr=stds, fmt='-o', capsize=5, capthick=1.5,
                       markersize=6, linewidth=1.5, alpha=0.7, 
                       markerfacecolor='white', markeredgecolor='black', 
                       markeredgewidth=1, ecolor='black', elinewidth=1)
            
            # Customize plot
            ax.set_xlabel('Experiment Type', fontweight='bold')
            ax.set_ylabel(_format_metric_name(metric), fontweight='bold')
            ax.set_title(f'{_format_metric_name(metric)} Across Experiments', 
                        fontweight='bold', pad=15)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(experiment_names, rotation=45, ha='right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels above each point - format large numbers efficiently
            y_max = max(m + s for m, s in zip(means, stds)) if means else 0
            y_range = max(means) - min(means) if means else 1
            label_offset = max(y_range * 0.03, y_max * 0.02) if y_range > 0 else 0.01
            
            for i, (mean, std) in enumerate(zip(means, stds)):
                label_y = mean + std + label_offset
                # Format large numbers more efficiently to reduce text rendering memory
                if abs(mean) > 1e12:
                    label_text = f'{mean:.2e}'
                elif abs(mean) > 1e6:
                    label_text = f'{mean/1e6:.2f}M'
                else:
                    label_text = f'{mean:.4f}'
                ax.text(i, label_y, label_text, ha='center', va='bottom', fontsize=7)
            
            # Use subplots_adjust
            plt.subplots_adjust(bottom=0.25, top=0.90, left=0.12, right=0.95)
            
            # Save figure with reduced DPI to save memory
            safe_metric_name = metric.replace('/', '_').replace(' ', '_')
            output_path = os.path.join(output_dir, f'{safe_metric_name}.png')
            # Reduce DPI from 200 to 150 to significantly reduce memory usage
            # Don't use bbox_inches='tight' as it can cause memory issues
            plt.savefig(output_path, dpi=150)
            logger.info(f"Saved chart ({metric_idx+1}/{len(key_metrics)}): {output_path}")
            
            # Explicitly close and clean up
            plt.close(fig)
            del fig, ax
            gc.collect()  # Force garbage collection after each chart
            
        except MemoryError as e:
            logger.error(f"Memory error creating chart for {metric}: {e}. Skipping...")
            try:
                plt.close('all')
            except:
                pass
            gc.collect()
            continue
        except Exception as e:
            logger.error(f"Error creating chart for {metric}: {e}. Skipping...")
            try:
                plt.close('all')
            except:
                pass
            gc.collect()
            continue


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


def run_aggregate_results(output_dir: str = "aggregations",
                         global_exp_num: Optional[int] = None):
    """
    Main function to aggregate evaluation results across experiments.
    
    Args:
        output_dir: Base directory to save aggregated results (default: "aggregations")
        global_exp_num: Optional global experiment number to aggregate. 
                       If None, aggregates all global experiments separately.
    """
    logger.info("AGGREGATING EXPERIMENT RESULTS")
    logger.info("=" * 100)
    
    # Get experiments directory from config
    paths_config = CONFIG.get("paths", {})
    experiments_dir = paths_config.get("experiments", "experiments")
    
    # Determine which global experiment numbers to process
    if global_exp_num is not None:
        # Process single global experiment
        global_exp_nums = [global_exp_num]
    else:
        # Process all global experiments
        if not os.path.exists(experiments_dir):
            raise FileNotFoundError(f"Experiments directory not found: {experiments_dir}")
        
        subdirs = [d for d in os.listdir(experiments_dir) 
                   if os.path.isdir(os.path.join(experiments_dir, d)) and d.isdigit()]
        if not subdirs:
            raise ValueError(f"No global experiment number directories found in {experiments_dir}")
        
        global_exp_nums = sorted([int(d) for d in subdirs])
        logger.info(f"No global_exp_num specified, processing all global experiments: {global_exp_nums}")
    
    # Process each global experiment
    for exp_num in global_exp_nums:
        logger.info(f"\n{'=' * 100}")
        logger.info(f"Processing global experiment {exp_num}")
        logger.info(f"{'=' * 100}")
        
        # Aggregate metrics
        logger.info(f"Scanning experiments directory: {experiments_dir}")
        aggregated, used_exp_num = aggregate_metrics(experiments_dir, global_exp_num=exp_num)
        
        if not aggregated:
            logger.warning(f"No evaluation results found for global experiment {exp_num}. Skipping...")
            continue
        
        # Create output directory structure: aggregations/<global_exp_num>/
        exp_output_dir = os.path.join(output_dir, str(used_exp_num))
        os.makedirs(exp_output_dir, exist_ok=True)
        logger.info(f"Output directory: {os.path.abspath(exp_output_dir)}")
        
        # Create aggregation table
        logger.info("Creating aggregation table...")
        df = create_aggregation_table(aggregated)
        
        # Save as CSV
        csv_path = os.path.join(exp_output_dir, "aggregated_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV table: {csv_path}")
        
        # Save as LaTeX
        latex_path = os.path.join(exp_output_dir, "aggregated_metrics.tex")
        save_latex_table(df, latex_path)
        
        # Create charts
        logger.info("\nCreating charts...")
        charts_dir = os.path.join(exp_output_dir, "charts")
        create_charts(aggregated, charts_dir)
        
        # Save detailed JSON with all statistics
        detailed_stats = {}
        for exp_name in aggregated:
            detailed_stats[exp_name] = {}
            for metric in aggregated[exp_name]:
                detailed_stats[exp_name][metric] = compute_statistics(aggregated[exp_name][metric])
        
        json_path = os.path.join(exp_output_dir, "detailed_statistics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_stats, f, indent=2)
        logger.info(f"Saved detailed statistics: {json_path}")
        
        logger.info(f"\n✓ Aggregation complete for global experiment {used_exp_num}! Results saved to {exp_output_dir}")
    
    logger.info("\n" + "=" * 100)
    logger.info(f"✓ All aggregations complete! Results saved to {output_dir}")
    logger.info("=" * 100)
