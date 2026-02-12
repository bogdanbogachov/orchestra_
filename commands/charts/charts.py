"""
Create bar charts comparing F1 scores across aggregations.

This module reads aggregated_metrics.csv files from selected aggregations,
extracts F1 scores for all experiment configurations,
and creates bar charts with error bars organized by dataset and noise condition.
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


def extract_all_f1_scores(aggregation_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Extract F1 scores from an aggregated_metrics.csv file for all experiments.
    
    Args:
        aggregation_path: Path to the aggregation directory containing aggregated_metrics.csv
    
    Returns:
        Dictionary mapping normalized experiment names to (mean, std) tuples for F1 scores.
        Normalizes experiment names by removing dataset prefix and normalizing FFT variants.
        Example: {'default': (0.88, 0.01), 'custom_last': (0.90, 0.01), ...}
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
    
    # Iterate through rows and extract all experiments
    for _, row in df.iterrows():
        exp_name = str(row['Experiment']).strip()
        f1_value = str(row.get('f1', ''))
        
        if not f1_value or f1_value == 'nan':
            continue
        
        # Extract the experiment type by removing the dataset prefix
        # Examples:
        # - banking77_clean_default -> default
        # - banking77_noise_custom_last -> custom_last
        # - clinc150_clean_fft40_custom_fft_attention -> fft_custom_fft_attention
        # - banking77_noise_fft50_custom_fft_last -> fft_custom_fft_last
        
        # Remove dataset prefix (banking77_clean_, banking77_noise_, clinc150_clean_, clinc150_noise_)
        exp_type = re.sub(r'^(banking77|clinc150)_(clean|noise)_', '', exp_name)
        
        # Normalize FFT variants (fft40, fft50 -> fft)
        exp_type = re.sub(r'fft\d+', 'fft', exp_type)
        
        mean, std = parse_mean_std(f1_value)
        f1_scores[exp_type] = (mean, std)
        logger.debug(f"Found {exp_type} in {aggregation_path}: F1 = {mean:.4f} ± {std:.4f}")
    
    return f1_scores


def create_f1_bar_chart_two_groups(group1_data: Dict[str, Tuple[float, float]],
                                   group1_label: str,
                                   group2_data: Dict[str, Tuple[float, float]],
                                   group2_label: str,
                                   output_path: str,
                                   figure_title: Optional[str] = None):
    """
    Create a bar chart with two groups, each containing all experiment configurations.
    
    Args:
        group1_data: Dictionary mapping experiment names to (mean, std) tuples for group 1
        group1_label: Label for group 1 (e.g., "Clean Banking")
        group2_data: Dictionary mapping experiment names to (mean, std) tuples for group 2
        group2_label: Label for group 2 (e.g., "Noisy Banking")
        output_path: Full path to save the chart
        figure_title: Optional title for the figure
    """
    # Get all unique experiment types from both groups
    all_exp_types = sorted(set(list(group1_data.keys()) + list(group2_data.keys())))
    
    if not all_exp_types:
        logger.error("No experiment data provided")
        return
    
    # Define the order we want for experiment types
    # This ensures consistent ordering across charts
    preferred_order = [
        'default',
        'custom_attention',
        'custom_last',
        'custom_max',
        'custom_mean',
        'fft_custom_fft_attention',
        'fft_custom_fft_last',
        'fft_custom_fft_max',
        'fft_custom_fft_mean'
    ]
    
    # Sort: first by preferred order, then alphabetically for any not in preferred order
    def sort_key(exp_type):
        if exp_type in preferred_order:
            return (0, preferred_order.index(exp_type))
        return (1, exp_type)
    
    all_exp_types = sorted(all_exp_types, key=sort_key)
    
    # Create display labels for experiment types
    def format_exp_label(exp_type: str) -> str:
        """Format experiment type for display."""
        # Replace underscores with spaces and capitalize
        label = exp_type.replace('_', ' ').title()
        # Handle special cases
        label = label.replace('Fft', 'FFT')
        label = label.replace('Custom', 'Custom')
        return label
    
    exp_labels = [format_exp_label(exp_type) for exp_type in all_exp_types]
    
    # Prepare data for plotting
    num_exp_types = len(all_exp_types)
    bar_width = 0.8  # Width of each bar
    group_spacing = 1.0  # Space between the two groups
    
    # Calculate x positions
    # Group 1: All bars grouped together (no space between bars within group)
    # Group 2: All bars grouped together (no space between bars within group)
    # Space between the two groups
    
    group1_width = num_exp_types * bar_width  # Total width of group 1
    group1_start = 0
    group2_start = group1_start + group1_width + group_spacing
    
    x_positions_group1 = []
    x_positions_group2 = []
    
    for i, exp_type in enumerate(all_exp_types):
        # Position bars within group 1 (no space between them)
        x1 = group1_start + i * bar_width + bar_width / 2
        x_positions_group1.append((exp_type, x1))
        
        # Position bars within group 2 (no space between them)
        x2 = group2_start + i * bar_width + bar_width / 2
        x_positions_group2.append((exp_type, x2))
    
    # Calculate group centers for x-axis labels
    group1_center = group1_start + group1_width / 2
    group2_center = group2_start + group1_width / 2
    
    # Extract means and stds
    means_group1 = []
    stds_group1 = []
    means_group2 = []
    stds_group2 = []
    
    for exp_type in all_exp_types:
        if exp_type in group1_data:
            mean, std = group1_data[exp_type]
            means_group1.append(mean)
            stds_group1.append(std)
        else:
            means_group1.append(0.0)
            stds_group1.append(0.0)
        
        if exp_type in group2_data:
            mean, std = group2_data[exp_type]
            means_group2.append(mean)
            stds_group2.append(std)
        else:
            means_group2.append(0.0)
            stds_group2.append(0.0)
    
    # Create the figure
    # Calculate width needed: 2 groups with bars + spacing
    total_width = group1_width + group_spacing + group1_width
    fig_width = max(14, total_width * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    
    # Define colors - use different colors for each group
    color_group1 = '#2E86AB'  # Blue
    color_group2 = '#A23B72'  # Purple
    
    # Define hatch patterns for different experiment types (9 unique patterns)
    # Using dense patterns for better visibility
    # The last pattern (fft_custom_fft_mean) will use filled squares effect
    hatches = [
        '',           # 1. No hatch (solid) - default
        '///',        # 2. Forward diagonal lines (dense) - custom_attention
        '\\\\\\',     # 3. Backward diagonal lines (dense) - custom_last
        '|||',        # 4. Vertical lines (dense) - custom_max
        '---',        # 5. Horizontal lines (dense) - custom_mean
        '+++',        # 6. Plus signs (dense) - fft_custom_fft_attention
        'xxx',        # 7. Crosshatch (dense) - fft_custom_fft_last
        '...',        # 8. Dots (dense) - fft_custom_fft_max
        'ooo'         # 9. Filled squares effect (dense circles) - fft_custom_fft_mean
    ]
    
    # Ensure we have exactly the right number of patterns
    if len(all_exp_types) <= len(hatches):
        # Use only the first N patterns where N = number of experiment types
        hatches = hatches[:len(all_exp_types)]
    else:
        # If somehow we have more than 9, extend with variations
        while len(hatches) < len(all_exp_types):
            hatches.append(hatches[len(hatches) % 9])
    
    # Plot bars for group 1
    bars1 = []
    for i, (exp_type, x_pos) in enumerate(x_positions_group1):
        hatch_idx = i % len(hatches)
        hatch = hatches[hatch_idx]
        bar = ax.bar(x_pos, means_group1[i], width=bar_width, yerr=stds_group1[i],
                    color=color_group1, hatch=hatch, edgecolor='black', linewidth=1,
                    alpha=0.8, error_kw={'elinewidth': 1.5, 'capsize': 5, 'capthick': 1.5})
        bars1.append(bar)
    
    # Plot bars for group 2
    bars2 = []
    for i, (exp_type, x_pos) in enumerate(x_positions_group2):
        hatch_idx = i % len(hatches)
        hatch = hatches[hatch_idx]
        bar = ax.bar(x_pos, means_group2[i], width=bar_width, yerr=stds_group2[i],
                    color=color_group2, hatch=hatch, edgecolor='black', linewidth=1,
                    alpha=0.8, error_kw={'elinewidth': 1.5, 'capsize': 5, 'capthick': 1.5})
        bars2.append(bar)
    
    # Create legend entries
    legend_handles = []
    
    # Add group labels
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, 
        facecolor=color_group1, edgecolor='black', alpha=0.8, label=group1_label))
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, 
        facecolor=color_group2, edgecolor='black', alpha=0.8, label=group2_label))
    
    # Add experiment type entries (with hatch patterns)
    for exp_label, hatch_pattern in zip(exp_labels, hatches[:len(exp_labels)]):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, 
            facecolor='white', edgecolor='black', hatch=hatch_pattern,
            label=exp_label))
    
    # Set x-axis labels (group labels at group centers)
    ax.set_xticks([group1_center, group2_center])
    ax.set_xticklabels([group1_label, group2_label])
    
    # Set labels
    ax.set_xlabel('Dataset Condition', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    
    # Don't set title (removed per user request)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis to start from a reasonable minimum
    all_means = means_group1 + means_group2
    all_stds = stds_group1 + stds_group2
    if all_means:
        y_min = max(0, min(all_means) - max(all_stds) - 0.05)
        y_max = max(all_means) + max(all_stds) + 0.05
        ax.set_ylim(y_min, y_max)
    
    # Add legend
    ax.legend(handles=legend_handles, loc='best', framealpha=0.9, ncol=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved F1 scores chart: {output_path}")
    
    plt.close(fig)


def run_charts(aggregation_nums: List[int], aggregations_dir: str = "aggregations", 
               output_dir: str = "charts", aggregation_names: Optional[List[str]] = None):
    """
    Main function to create F1 score comparison charts.
    
    Creates two separate figures:
    1. Banking dataset: Clean Banking vs Noisy Banking (all 9 configurations)
    2. CLINC dataset: Clean CLINC vs Noisy CLINC (all 9 configurations)
    
    Args:
        aggregation_nums: List of aggregation numbers to include in the charts
                         Expected: [62, 63, 66, 67] for banking_noise, banking_clean, clinc_clean, clinc_noise
        aggregations_dir: Base directory containing aggregation folders
        output_dir: Directory to save the charts
        aggregation_names: Optional list of custom names (not used in new format, kept for compatibility)
    """
    logger.info("CREATING F1 SCORES COMPARISON CHARTS")
    logger.info("=" * 100)
    
    if not aggregation_nums:
        logger.error("No aggregation numbers provided")
        return
    
    # Expected aggregation numbers:
    # 62: banking77_noise (noisy banking)
    # 63: banking77_clean (clean banking)
    # 66: clinc150_clean (clean clinc)
    # 67: clinc150_noise (noisy clinc)
    
    # Collect F1 scores from each aggregation
    aggregations_data = {}
    
    for agg_num in aggregation_nums:
        agg_path = os.path.join(aggregations_dir, str(agg_num))
        
        if not os.path.exists(agg_path):
            logger.warning(f"Aggregation directory not found: {agg_path}")
            continue
        
        f1_scores = extract_all_f1_scores(agg_path)
        
        if not f1_scores:
            logger.warning(f"No F1 scores found in aggregation {agg_num}, skipping...")
            continue
        
        aggregations_data[agg_num] = f1_scores
        logger.info(f"Loaded F1 scores from aggregation {agg_num}: {len(f1_scores)} experiments")
    
    if not aggregations_data:
        logger.error("No valid aggregation data found")
        return
    
    # Create Figure 1: BANKING77 dataset
    # Group 1: Clean BANKING77 (agg 63)
    # Group 2: Noisy BANKING77 (agg 62)
    if 63 in aggregations_data and 62 in aggregations_data:
        logger.info("\nCreating Figure 1: BANKING77 Dataset (Clean vs Noisy)...")
        output_path_fig1 = os.path.join(output_dir, 'f1_scores_banking.png')
        create_f1_bar_chart_two_groups(
            group1_data=aggregations_data[63],
            group1_label="Clean BANKING77",
            group2_data=aggregations_data[62],
            group2_label="Noisy BANKING77",
            output_path=output_path_fig1,
            figure_title=None
        )
        logger.info(f"✓ Figure 1 saved: {output_path_fig1}")
    else:
        logger.warning("Missing aggregation data for BANKING77 dataset (need 62 and 63)")
    
    # Create Figure 2: CLINC150 dataset
    # Group 1: Clean CLINC150 (agg 66)
    # Group 2: Noisy CLINC150 (agg 67)
    if 66 in aggregations_data and 67 in aggregations_data:
        logger.info("\nCreating Figure 2: CLINC150 Dataset (Clean vs Noisy)...")
        output_path_fig2 = os.path.join(output_dir, 'f1_scores_clinc.png')
        create_f1_bar_chart_two_groups(
            group1_data=aggregations_data[66],
            group1_label="Clean CLINC150",
            group2_data=aggregations_data[67],
            group2_label="Noisy CLINC150",
            output_path=output_path_fig2,
            figure_title=None
        )
        logger.info(f"✓ Figure 2 saved: {output_path_fig2}")
    else:
        logger.warning("Missing aggregation data for CLINC150 dataset (need 66 and 67)")
    
    logger.info(f"\n✓ Chart creation complete! Results saved to {output_dir}")
    logger.info("=" * 100)

