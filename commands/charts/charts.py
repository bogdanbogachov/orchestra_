"""
Create bar charts comparing F1 scores across aggregations.

This module reads aggregated_metrics.csv files from selected aggregations,
extracts F1 scores for default, custom attention, custom fft attention, custom last, and custom fft last experiments,
and creates a bar chart with error bars.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

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
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1


def parse_mean_std(value_str: str) -> Tuple[float, float]:
    """
    Parse a string in the format "mean ± std" to extract mean and standard deviation.
    
    Args:
        value_str: String in format like "0.8882 ± 0.0083"
    
    Returns:
        Tuple of (mean, std)
    """
    # Match pattern: number ± number
    match = re.match(r'([\d.]+)\s*±\s*([\d.]+)', value_str.strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    # If no match, try to parse as just a number (mean only, std = 0)
    try:
        return float(value_str.strip()), 0.0
    except ValueError:
        logger.warning(f"Could not parse value: {value_str}, using (0.0, 0.0)")
        return 0.0, 0.0


def extract_f1_scores(aggregation_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Extract F1 scores from an aggregated_metrics.csv file.
    
    Args:
        aggregation_path: Path to the aggregation directory containing aggregated_metrics.csv
    
    Returns:
        Dictionary mapping experiment names to (mean, std) tuples for F1 scores.
        Only includes: default, custom_attention, custom_fft_attention, custom_last, custom_fft_last
    """
    csv_path = os.path.join(aggregation_path, "aggregated_metrics.csv")
    
    if not os.path.exists(csv_path):
        logger.warning(f"aggregated_metrics.csv not found in {aggregation_path}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV from {csv_path}: {e}")
        return {}
    
    f1_scores = {}
    
    # Define the experiment patterns we're looking for
    target_experiments = {
        'default': r'.*_default$',
        'custom_attention': r'.*_custom_attention$',
        'custom_fft_attention': r'.*_custom_fft_attention$',
        'custom_last': r'.*_custom_last$',
        'custom_fft_last': r'.*_custom_fft_last$'
    }
    
    # Iterate through rows and find matching experiments
    for _, row in df.iterrows():
        exp_name = str(row['Experiment']).strip()
        f1_value = str(row.get('f1', ''))
        
        if not f1_value or f1_value == 'nan':
            continue
        
        # Check if this experiment matches any of our targets
        for exp_type, pattern in target_experiments.items():
            if re.match(pattern, exp_name):
                mean, std = parse_mean_std(f1_value)
                f1_scores[exp_type] = (mean, std)
                logger.debug(f"Found {exp_type} in {aggregation_path}: F1 = {mean:.4f} ± {std:.4f}")
                break
    
    return f1_scores


def create_f1_bar_chart(aggregations_data: Dict[int, Dict[str, Tuple[float, float]]], 
                        output_dir: str = "charts",
                        aggregation_names: Optional[Dict[int, str]] = None,
                        aggregation_order: Optional[List[int]] = None):
    """
    Create a bar chart comparing F1 scores across aggregations.
    
    Args:
        aggregations_data: Dictionary mapping aggregation numbers to their F1 scores.
                          Format: {agg_num: {'default': (mean, std), 'custom_attention': (mean, std), 
                                             'custom_fft_attention': (mean, std), 'custom_last': (mean, std),
                                             'custom_fft_last': (mean, std)}}
        output_dir: Directory to save the chart
        aggregation_names: Optional dictionary mapping aggregation numbers to custom names.
                          If None, uses default "Aggregation {num}" format.
        aggregation_order: Optional list specifying the order of aggregations to display.
                          If None, uses sorted order.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define experiment types and their display names
    exp_types = ['default', 'custom_attention', 'custom_fft_attention', 'custom_last', 'custom_fft_last']
    exp_labels = ['Default', 'Custom Attention', 'Custom FFT Attention', 'Custom Last', 'Custom FFT Last']
    
    # Get aggregation numbers in the specified order, or sorted if no order provided
    if aggregation_order:
        # Filter to only include aggregations that have data, preserving order
        agg_nums = [agg_num for agg_num in aggregation_order if agg_num in aggregations_data]
    else:
        # Fallback to sorted order if no order specified
        agg_nums = sorted(aggregations_data.keys())
    
    if not agg_nums:
        logger.error("No aggregation data provided")
        return
    
    # Prepare data for plotting
    num_aggregations = len(agg_nums)
    num_exp_types = len(exp_types)
    
    # Calculate bar positions
    # Each aggregation has bars (one for each exp type) with no space between them
    # Space between different aggregations
    bar_width = 0.8  # Width of each bar
    group_width = num_exp_types * bar_width  # Total width of one aggregation group
    group_spacing = 1.0  # Space between aggregation groups
    
    # Calculate x positions for each bar
    x_positions = []
    group_centers = []
    
    for i, agg_num in enumerate(agg_nums):
        group_start = i * (group_width + group_spacing)
        group_center = group_start + group_width / 2
        group_centers.append(group_center)
        
        # Position bars within this group (no space between them)
        for j, exp_type in enumerate(exp_types):
            x_pos = group_start + j * bar_width + bar_width / 2
            x_positions.append((agg_num, exp_type, x_pos))
    
    # Extract means and stds
    means = []
    stds = []
    bar_labels = []
    
    for agg_num in agg_nums:
        for exp_type in exp_types:
            if exp_type in aggregations_data[agg_num]:
                mean, std = aggregations_data[agg_num][exp_type]
                means.append(mean)
                stds.append(std)
            else:
                means.append(0.0)
                stds.append(0.0)
            bar_labels.append(f"{exp_type}")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(max(12, num_aggregations * 2.5), 7))
    
    # Define colors for each aggregation (one color per aggregation)
    colors = plt.cm.tab10(np.linspace(0, 1, num_aggregations))
    color_map = {agg_num: colors[i] for i, agg_num in enumerate(agg_nums)}
    
    # Define hatch patterns for each experiment type
    hatches = ['', '///', '...', 'xxx', '+++']  # No hatch, diagonal lines, dots, crosshatch, plus
    
    # Plot bars
    bars = []
    
    for i, (agg_num, exp_type, x_pos) in enumerate(x_positions):
        color = color_map[agg_num]
        hatch_idx = exp_types.index(exp_type)
        hatch = hatches[hatch_idx]
        
        bar = ax.bar(x_pos, means[i], width=bar_width, yerr=stds[i],
                    color=color, hatch=hatch, edgecolor='black', linewidth=1,
                    error_kw={'elinewidth': 1.5, 'capsize': 5, 'capthick': 1.5})
        bars.append(bar)
    
    # Create legend entries
    legend_handles = []
    
    # Add experiment type entries (with hatch patterns)
    for exp_label, hatch_pattern in zip(exp_labels, hatches):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, 
            facecolor='white', edgecolor='black', hatch=hatch_pattern,
            label=exp_label))
    
    # Add aggregation entries (with colors)
    for agg_num, color in color_map.items():
        if aggregation_names and agg_num in aggregation_names:
            label = aggregation_names[agg_num]
        else:
            label = f'Aggregation {agg_num}'
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, 
            facecolor=color, edgecolor='black', label=label))
    
    # Set x-axis labels
    ax.set_xticks(group_centers)
    if aggregation_names:
        ax.set_xticklabels([aggregation_names.get(agg_num, f'Agg {agg_num}') for agg_num in agg_nums])
    else:
        ax.set_xticklabels([f'Agg {agg_num}' for agg_num in agg_nums])
    
    # Set labels
    ax.set_xlabel('Aggregation', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis to start from a reasonable minimum
    if means:
        y_min = max(0, min(means) - max(stds) - 0.05)
        y_max = max(means) + max(stds) + 0.05
        ax.set_ylim(y_min, y_max)
    
    # Add legend
    ax.legend(handles=legend_handles, loc='best', framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    output_path = os.path.join(output_dir, 'f1_scores_comparison.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved F1 scores chart: {output_path}")
    
    plt.close(fig)


def run_charts(aggregation_nums: List[int], aggregations_dir: str = "aggregations", 
               output_dir: str = "charts", aggregation_names: Optional[List[str]] = None):
    """
    Main function to create F1 score comparison charts.
    
    Args:
        aggregation_nums: List of aggregation numbers to include in the chart
        aggregations_dir: Base directory containing aggregation folders
        output_dir: Directory to save the charts
        aggregation_names: Optional list of custom names for aggregations (must match length of aggregation_nums)
    """
    logger.info("CREATING F1 SCORES COMPARISON CHART")
    logger.info("=" * 100)
    
    if not aggregation_nums:
        logger.error("No aggregation numbers provided")
        return
    
    # Validate aggregation names if provided
    if aggregation_names is not None:
        if len(aggregation_names) != len(aggregation_nums):
            logger.error(f"Number of aggregation names ({len(aggregation_names)}) must match number of aggregation numbers ({len(aggregation_nums)})")
            return
        # Create mapping from aggregation number to name
        name_map = {agg_num: name for agg_num, name in zip(aggregation_nums, aggregation_names)}
    else:
        name_map = None
    
    # Collect F1 scores from each aggregation, preserving order
    aggregations_data = {}
    valid_aggregation_order = []  # Track order of successfully loaded aggregations
    
    for agg_num in aggregation_nums:
        agg_path = os.path.join(aggregations_dir, str(agg_num))
        
        if not os.path.exists(agg_path):
            logger.warning(f"Aggregation directory not found: {agg_path}")
            continue
        
        f1_scores = extract_f1_scores(agg_path)
        
        if not f1_scores:
            logger.warning(f"No F1 scores found in aggregation {agg_num}, skipping...")
            continue
        
        # Check if we have at least one of the required experiment types
        required_types = ['default', 'custom_attention', 'custom_fft_attention', 'custom_last', 'custom_fft_last']
        if not any(exp_type in f1_scores for exp_type in required_types):
            logger.warning(f"None of the required experiment types found in aggregation {agg_num}, skipping...")
            continue
        
        aggregations_data[agg_num] = f1_scores
        valid_aggregation_order.append(agg_num)  # Add to order list
        logger.info(f"Loaded F1 scores from aggregation {agg_num}: {f1_scores}")
    
    if not aggregations_data:
        logger.error("No valid aggregation data found")
        return
    
    # Create the chart
    logger.info(f"\nCreating F1 scores comparison chart...")
    create_f1_bar_chart(aggregations_data, output_dir, aggregation_names=name_map, 
                       aggregation_order=valid_aggregation_order)
    
    logger.info(f"\n✓ Chart creation complete! Results saved to {output_dir}")
    logger.info("=" * 100)

