#!/usr/bin/env python3
"""
Run paired significance tests from raw seed-level experiment outputs.

Expected input layout:
  raw_results_pull/<aggregation_id>/<experiment_base>_<seed>/<head>/evaluation_results.json

Outputs:
  rebuttal_significance/significance_results.md
  rebuttal_significance/per_seed_metrics.csv
  rebuttal_significance/metric_summary.csv
  rebuttal_significance/primary_paired_tests.csv
  rebuttal_significance/all_pairwise_f1_paired_tests.csv
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "raw_results_pull"
OUT_DIR = ROOT / "rebuttal_significance"

METRICS = ["accuracy", "precision", "recall", "f1"]

DATASET_LABELS = {
    "62": "BANKING77 noisy",
    "63": "BANKING77 clean",
    "66": "CLINC150 clean",
    "67": "CLINC150 noisy",
    "71": "Production selected-5 noisy",
}

PRIMARY_COMPARISONS = [
    ("62", "banking77_noise_custom_attention", "banking77_noise_default", "attention vs default head"),
    ("62", "banking77_noise_custom_attention", "banking77_noise_custom_last", "attention vs custom last-token"),
    ("63", "banking77_clean_custom_attention", "banking77_clean_default", "attention vs default head"),
    ("63", "banking77_clean_custom_attention", "banking77_clean_custom_last", "attention vs custom last-token"),
    ("67", "clinc150_noise_custom_attention", "clinc150_noise_default", "attention vs default head"),
    ("67", "clinc150_noise_custom_attention", "clinc150_noise_custom_last", "attention vs custom last-token"),
    ("66", "clinc150_clean_custom_attention", "clinc150_clean_default", "attention vs default head"),
    ("66", "clinc150_clean_custom_attention", "clinc150_clean_custom_last", "attention vs custom last-token"),
    ("71", "additional_noise_custom_attention", "additional_noise_default", "attention vs default head"),
    ("71", "additional_noise_custom_attention", "additional_noise_custom_last", "attention vs custom last-token"),
]


def parse_experiment_dir(name: str) -> Tuple[str, int]:
    match = re.match(r"^(.+)_(\d+)$", name)
    if not match:
        raise ValueError(f"Cannot parse experiment directory name: {name}")
    return match.group(1), int(match.group(2))


def load_raw_results() -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"Missing raw results directory: {RAW_ROOT}")

    data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(RAW_ROOT.glob("*/*/*/evaluation_results.json")):
        aggregation = path.parents[2].name
        experiment_dir = path.parents[1].name
        base_name, seed = parse_experiment_dir(experiment_dir)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        data[aggregation][base_name][seed] = {
            metric: float(payload[metric])
            for metric in METRICS
            if metric in payload
        }

    return data


def ensure_complete(data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]]) -> List[str]:
    problems: List[str] = []
    for aggregation, experiments in sorted(data.items()):
        for exp_name, seed_map in sorted(experiments.items()):
            seeds = sorted(seed_map)
            if seeds != list(range(1, 11)):
                problems.append(f"{aggregation}/{exp_name}: expected seeds 1..10, found {seeds}")
            for seed, metrics in seed_map.items():
                missing = [m for m in METRICS if m not in metrics]
                if missing:
                    problems.append(f"{aggregation}/{exp_name}/seed {seed}: missing metrics {missing}")
    return problems


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def paired_test(
    data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    aggregation: str,
    treatment: str,
    comparator: str,
    metric: str,
    label: str,
) -> Dict[str, object]:
    treatment_seeds = set(data[aggregation][treatment])
    comparator_seeds = set(data[aggregation][comparator])
    paired_seeds = sorted(treatment_seeds & comparator_seeds)
    if len(paired_seeds) < 2:
        raise ValueError(f"Need at least two matched seeds for {aggregation}: {treatment} vs {comparator}")

    treatment_values = np.asarray([data[aggregation][treatment][s][metric] for s in paired_seeds], dtype=float)
    comparator_values = np.asarray([data[aggregation][comparator][s][metric] for s in paired_seeds], dtype=float)
    diff = treatment_values - comparator_values

    t_res = stats.ttest_rel(treatment_values, comparator_values)
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    se = sd_diff / math.sqrt(len(diff))
    df = len(diff) - 1
    tcrit = stats.t.ppf(0.975, df)
    ci_low = mean_diff - tcrit * se
    ci_high = mean_diff + tcrit * se
    dz = mean_diff / sd_diff if sd_diff else float("nan")

    # Wilcoxon is useful as a robustness check; zero_method handles rare equal pairs.
    try:
        w_res = stats.wilcoxon(treatment_values, comparator_values, zero_method="wilcox")
        wilcoxon_p = float(w_res.pvalue)
    except ValueError:
        wilcoxon_p = float("nan")

    treatment_mean, treatment_sd = mean_sd(treatment_values)
    comparator_mean, comparator_sd = mean_sd(comparator_values)

    return {
        "aggregation": aggregation,
        "dataset": DATASET_LABELS.get(aggregation, aggregation),
        "metric": metric,
        "comparison": label,
        "treatment": treatment,
        "comparator": comparator,
        "n": len(paired_seeds),
        "seeds": " ".join(str(s) for s in paired_seeds),
        "treatment_mean": treatment_mean,
        "treatment_sd": treatment_sd,
        "comparator_mean": comparator_mean,
        "comparator_sd": comparator_sd,
        "mean_diff": mean_diff,
        "diff_sd": sd_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t": float(t_res.statistic),
        "df": df,
        "p": float(t_res.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": dz,
    }


def holm_adjust(rows: List[Dict[str, object]], p_key: str = "p", out_key: str = "p_holm") -> None:
    indexed = sorted(enumerate(rows), key=lambda pair: float(pair[1][p_key]))
    m = len(indexed)
    previous = 0.0
    adjusted = [1.0] * len(rows)
    for rank, (idx, row) in enumerate(indexed, start=1):
        value = min(1.0, (m - rank + 1) * float(row[p_key]))
        value = max(previous, value)
        previous = value
        adjusted[idx] = value
    for row, value in zip(rows, adjusted):
        row[out_key] = value


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}"


def fmt_p(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "NA"
    value = float(value)
    if value < 0.001:
        return "<0.001"
    return f"{value:.4f}"


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_per_seed_rows(data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for aggregation, experiments in sorted(data.items()):
        for exp_name, seed_map in sorted(experiments.items()):
            for seed, metrics in sorted(seed_map.items()):
                row = {
                    "aggregation": aggregation,
                    "dataset": DATASET_LABELS.get(aggregation, aggregation),
                    "experiment": exp_name,
                    "seed": seed,
                }
                row.update(metrics)
                rows.append(row)
    return rows


def build_summary_rows(data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for aggregation, experiments in sorted(data.items()):
        for exp_name, seed_map in sorted(experiments.items()):
            row = {
                "aggregation": aggregation,
                "dataset": DATASET_LABELS.get(aggregation, aggregation),
                "experiment": exp_name,
                "n": len(seed_map),
            }
            for metric in METRICS:
                values = [seed_map[s][metric] for s in sorted(seed_map)]
                mean, sd = mean_sd(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_sd"] = sd
            rows.append(row)
    return rows


def build_primary_tests(data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for aggregation, treatment, comparator, label in PRIMARY_COMPARISONS:
        for metric in METRICS:
            rows.append(paired_test(data, aggregation, treatment, comparator, metric, label))

    for metric in METRICS:
        metric_rows = [r for r in rows if r["metric"] == metric]
        holm_adjust(metric_rows)

    return rows


def build_all_pairwise_f1(data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for aggregation, experiments in sorted(data.items()):
        for left, right in itertools.combinations(sorted(experiments), 2):
            label = f"{left} vs {right}"
            rows.append(paired_test(data, aggregation, left, right, "f1", label))

    for aggregation in sorted(data):
        agg_rows = [r for r in rows if r["aggregation"] == aggregation]
        holm_adjust(agg_rows)

    return rows


def find_primary(rows: List[Dict[str, object]], aggregation: str, comparison: str, metric: str = "f1") -> Dict[str, object]:
    for row in rows:
        if row["aggregation"] == aggregation and row["comparison"] == comparison and row["metric"] == metric:
            return row
    raise KeyError((aggregation, comparison, metric))


def write_markdown(
    data: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    summary_rows: List[Dict[str, object]],
    primary_rows: List[Dict[str, object]],
) -> None:
    coverage_rows = []
    for aggregation, experiments in sorted(data.items()):
        seed_counts = sorted({len(seed_map) for seed_map in experiments.values()})
        coverage_rows.append([
            aggregation,
            DATASET_LABELS.get(aggregation, aggregation),
            str(len(experiments)),
            ", ".join(str(x) for x in seed_counts),
        ])

    claim_rows = []
    for aggregation, treatment, comparator, label in PRIMARY_COMPARISONS:
        if label != "attention vs custom last-token":
            continue
        default_test = find_primary(primary_rows, aggregation, "attention vs default head")
        last_test = find_primary(primary_rows, aggregation, "attention vs custom last-token")
        claim_rows.append([
            DATASET_LABELS.get(aggregation, aggregation),
            f"{fmt_pct(float(last_test['treatment_mean']))} +/- {fmt_pct(float(last_test['treatment_sd']))}",
            f"{fmt_pct(float(default_test['comparator_mean']))} +/- {fmt_pct(float(default_test['comparator_sd']))}",
            f"{fmt_pct(float(last_test['comparator_mean']))} +/- {fmt_pct(float(last_test['comparator_sd']))}",
            f"{fmt_pct(float(default_test['mean_diff']))} pp",
            f"{fmt_pct(float(last_test['mean_diff']))} pp",
        ])

    primary_f1_rows = [r for r in primary_rows if r["metric"] == "f1"]
    primary_table_rows = [
        [
            str(r["dataset"]),
            str(r["comparison"]),
            f"{fmt_pct(float(r['mean_diff']))} pp",
            f"[{fmt_pct(float(r['ci_low']))}, {fmt_pct(float(r['ci_high']))}]",
            f"{float(r['t']):.3f} ({int(r['df'])})",
            fmt_p(float(r["p"])),
            fmt_p(float(r["p_holm"])),
            fmt_p(float(r["wilcoxon_p"])),
            f"{float(r['cohen_dz']):.2f}",
        ]
        for r in primary_f1_rows
    ]

    noisy_banking = find_primary(primary_rows, "62", "attention vs custom last-token")
    noisy_clinc = find_primary(primary_rows, "67", "attention vs custom last-token")
    production = find_primary(primary_rows, "71", "attention vs custom last-token")
    production_default = find_primary(primary_rows, "71", "attention vs default head")

    lines = [
        "# Significance Analysis From Raw Seed Results",
        "",
        "## Inputs",
        "",
        "Source directory: `raw_results_pull/`.",
        "",
        "The runner uses matched seed-level `evaluation_results.json` files and runs paired tests on seeds `1..10`. This is the correct test family for the reviewer comment because each condition was run under the same seed indices.",
        "",
        markdown_table(["Aggregation", "Dataset", "Configurations", "Seeds per configuration"], coverage_rows),
        "",
        "## Claim Check",
        "",
        markdown_table(
            [
                "Dataset",
                "Attention F1",
                "Default F1",
                "Custom last F1",
                "Gain vs default",
                "Gain vs custom last",
            ],
            claim_rows,
        ),
        "",
        "The reviewer's arithmetic is correct for the proxy noisy datasets: the `+2.6-2.8 F1` gain is against the default head, while the gain against the fairer custom last-token comparator is much smaller.",
        "",
        "## Primary Paired F1 Tests",
        "",
        markdown_table(
            [
                "Dataset",
                "Comparison",
                "Mean diff",
                "95% CI",
                "paired t(df)",
                "p",
                "Holm p",
                "Wilcoxon p",
                "Cohen dz",
            ],
            primary_table_rows,
        ),
        "",
        "Holm correction is applied within the primary F1 family shown above. The CSV output also contains paired tests for accuracy, precision, and recall.",
        "",
        "## Interpretation",
        "",
        f"- On noisy BANKING77, attention vs custom last-token is `+{fmt_pct(float(noisy_banking['mean_diff']))}` F1 points with paired `p={fmt_p(float(noisy_banking['p']))}` and Holm `p={fmt_p(float(noisy_banking['p_holm']))}`.",
        f"- On noisy CLINC150, attention vs custom last-token is `+{fmt_pct(float(noisy_clinc['mean_diff']))}` F1 points with paired `p={fmt_p(float(noisy_clinc['p']))}` and Holm `p={fmt_p(float(noisy_clinc['p_holm']))}`.",
        "- So the rebuttal should not defend `+2.6-2.8 F1` as a pooling-only result. It should explicitly revise the claim: the large gain is a custom-head/system gain over C0; the pooling-only gains over C3 are modest on proxy data.",
        f"- The production selected-5 result is much stronger: attention beats custom last-token by `{fmt_pct(float(production['mean_diff']))}` F1 points and default by `{fmt_pct(float(production_default['mean_diff']))}` F1 points.",
        "",
        "## Suggested Rebuttal Language",
        "",
        "> We thank the reviewer for pointing out that our headline comparison conflated the custom-head change with the pooling operator. We agree and will revise the abstract and results discussion to separate these effects. Using the fairer custom-head last-token baseline, attention pooling improves noisy BANKING77 and noisy CLINC150 by modest margins, and we now report paired tests over the ten matched seeds. The larger +2.6-2.8 F1 numbers should be described as gains over the default-head baseline rather than pooling-only gains. We will temper the headline claim accordingly and add the new industrial held-out evaluation, where the same attention-pooling configuration shows a substantially larger and statistically supported gain over both the default head and custom last-token baseline.",
        "",
        "## Output Files",
        "",
        "- `per_seed_metrics.csv`: raw seed-level metrics extracted from `raw_results_pull`.",
        "- `metric_summary.csv`: mean/std/count for each aggregation and configuration.",
        "- `primary_paired_tests.csv`: paired tests for attention vs default and attention vs custom last-token across accuracy, precision, recall, and F1.",
        "- `all_pairwise_f1_paired_tests.csv`: all pairwise F1 paired tests within each aggregation.",
        "",
    ]

    (OUT_DIR / "significance_results.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_raw_results()
    problems = ensure_complete(data)
    if problems:
        joined = "\n".join(f"- {p}" for p in problems)
        raise SystemExit(f"Raw result coverage is incomplete:\n{joined}")

    per_seed_rows = build_per_seed_rows(data)
    summary_rows = build_summary_rows(data)
    primary_rows = build_primary_tests(data)
    pairwise_rows = build_all_pairwise_f1(data)

    write_csv(
        OUT_DIR / "per_seed_metrics.csv",
        per_seed_rows,
        ["aggregation", "dataset", "experiment", "seed", *METRICS],
    )
    write_csv(
        OUT_DIR / "metric_summary.csv",
        summary_rows,
        [
            "aggregation",
            "dataset",
            "experiment",
            "n",
            *[f"{metric}_{stat}" for metric in METRICS for stat in ["mean", "sd"]],
        ],
    )
    test_fields = [
        "aggregation",
        "dataset",
        "metric",
        "comparison",
        "treatment",
        "comparator",
        "n",
        "seeds",
        "treatment_mean",
        "treatment_sd",
        "comparator_mean",
        "comparator_sd",
        "mean_diff",
        "diff_sd",
        "ci_low",
        "ci_high",
        "t",
        "df",
        "p",
        "p_holm",
        "wilcoxon_p",
        "cohen_dz",
    ]
    write_csv(OUT_DIR / "primary_paired_tests.csv", primary_rows, test_fields)
    write_csv(OUT_DIR / "all_pairwise_f1_paired_tests.csv", pairwise_rows, test_fields)
    write_markdown(data, summary_rows, primary_rows)

    print(OUT_DIR / "significance_results.md")


if __name__ == "__main__":
    main()
