#!/usr/bin/env python3
"""
Generate significance analysis for rebuttal response.

The available final 10-seed artifacts contain summary statistics
(mean/std/count) but not the raw per-seed paired values for all datasets.
Accordingly, this script reports Welch two-sample t-tests from summary stats
and explicitly marks paired tests as unavailable unless raw per-seed results
are present.
"""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "rebuttal_significance"
OUT_MD = OUT_DIR / "significance_results.md"


COMPARISONS = [
    {
        "dataset": "BANKING77 noisy",
        "aggregation": "62",
        "attention": "banking77_noise_custom_attention",
        "default": "banking77_noise_default",
        "custom_last": "banking77_noise_custom_last",
    },
    {
        "dataset": "BANKING77 clean",
        "aggregation": "63",
        "attention": "banking77_clean_custom_attention",
        "default": "banking77_clean_default",
        "custom_last": "banking77_clean_custom_last",
    },
    {
        "dataset": "CLINC150 noisy",
        "aggregation": "67",
        "attention": "clinc150_noise_custom_attention",
        "default": "clinc150_noise_default",
        "custom_last": "clinc150_noise_custom_last",
    },
    {
        "dataset": "CLINC150 clean",
        "aggregation": "66",
        "attention": "clinc150_clean_custom_attention",
        "default": "clinc150_clean_default",
        "custom_last": "clinc150_clean_custom_last",
    },
    {
        "dataset": "Production selected-5 noisy",
        "aggregation": "71",
        "attention": "additional_noise_custom_attention",
        "default": "additional_noise_default",
        "custom_last": "additional_noise_custom_last",
    },
]


def load_detail(aggregation: str) -> Dict:
    path = ROOT / "aggregations" / aggregation / "detailed_statistics.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric(detail: Dict, exp: str, metric_name: str = "f1") -> Dict[str, float]:
    return detail[exp][metric_name]


def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}"


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def welch_from_summary(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    mean_a, sd_a, n_a = a["mean"], a["std"], int(a["count"])
    mean_b, sd_b, n_b = b["mean"], b["std"], int(b["count"])
    result = stats.ttest_ind_from_stats(
        mean1=mean_a,
        std1=sd_a,
        nobs1=n_a,
        mean2=mean_b,
        std2=sd_b,
        nobs2=n_b,
        equal_var=False,
    )

    se = math.sqrt((sd_a ** 2) / n_a + (sd_b ** 2) / n_b)
    numerator = ((sd_a ** 2) / n_a + (sd_b ** 2) / n_b) ** 2
    denominator = ((sd_a ** 2 / n_a) ** 2) / (n_a - 1) + ((sd_b ** 2 / n_b) ** 2) / (n_b - 1)
    df = numerator / denominator
    diff = mean_a - mean_b
    tcrit = stats.t.ppf(0.975, df)
    pooled_sd = math.sqrt(((n_a - 1) * sd_a ** 2 + (n_b - 1) * sd_b ** 2) / (n_a + n_b - 2))
    cohen_d = diff / pooled_sd if pooled_sd else float("nan")

    return {
        "diff": diff,
        "t": float(result.statistic),
        "p": float(result.pvalue),
        "df": float(df),
        "ci_low": diff - tcrit * se,
        "ci_high": diff + tcrit * se,
        "cohen_d": cohen_d,
    }


def holm_adjust(rows: List[Dict[str, object]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda pair: pair[1]["p"])
    m = len(rows)
    prev = 0.0
    adjusted = [None] * m
    for rank, (idx, row) in enumerate(ordered, start=1):
        adj = min(1.0, (m - rank + 1) * float(row["p"]))
        adj = max(prev, adj)
        prev = adj
        adjusted[idx] = adj
    for row, adj in zip(rows, adjusted):
        row["p_holm"] = adj


def add_comparison(
    rows: List[Dict[str, object]],
    dataset: str,
    comparator_label: str,
    attention: Dict[str, float],
    comparator: Dict[str, float],
) -> None:
    test = welch_from_summary(attention, comparator)
    rows.append(
        {
            "dataset": dataset,
            "comparison": f"attention vs {comparator_label}",
            "attention_mean": attention["mean"],
            "attention_sd": attention["std"],
            "comparator_mean": comparator["mean"],
            "comparator_sd": comparator["std"],
            "n_attention": int(attention["count"]),
            "n_comparator": int(comparator["count"]),
            **test,
        }
    )


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    headers = list(headers)
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    rows: List[Dict[str, object]] = []
    claim_rows: List[Dict[str, object]] = []

    for item in COMPARISONS:
        detail = load_detail(item["aggregation"])
        att = metric(detail, item["attention"])
        default = metric(detail, item["default"])
        custom_last = metric(detail, item["custom_last"])

        claim_rows.append(
            {
                "dataset": item["dataset"],
                "attention": att,
                "default": default,
                "custom_last": custom_last,
                "gain_vs_default": att["mean"] - default["mean"],
                "gain_vs_custom_last": att["mean"] - custom_last["mean"],
            }
        )

        add_comparison(rows, item["dataset"], "default head", att, default)
        add_comparison(rows, item["dataset"], "custom last-token", att, custom_last)

    holm_adjust(rows)

    lines = [
        "# Significance Analysis for Reviewer Comment 3",
        "",
        "## Inputs",
        "",
        "Source files: `aggregations/{62,63,66,67,71}/detailed_statistics.json`.",
        "",
        "Important limitation: these aggregation files contain only `mean`, `std`, and `count`; they do not contain the ten raw seed-level values. Therefore the tests below are Welch two-sample t-tests from summary statistics. A paired t-test cannot be reconstructed from these summaries because it requires the per-seed paired differences.",
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
            [
                [
                    r["dataset"],
                    f"{fmt_pct(r['attention']['mean'])} +/- {fmt_pct(r['attention']['std'])}",
                    f"{fmt_pct(r['default']['mean'])} +/- {fmt_pct(r['default']['std'])}",
                    f"{fmt_pct(r['custom_last']['mean'])} +/- {fmt_pct(r['custom_last']['std'])}",
                    f"{fmt_pct(r['gain_vs_default'])} pp",
                    f"{fmt_pct(r['gain_vs_custom_last'])} pp",
                ]
                for r in claim_rows
            ],
        ),
        "",
        "The reviewer is correct about the arithmetic for the noisy proxy datasets: attention improves over the default head by 2.64 F1 points on noisy BANKING77 and 2.83 F1 points on noisy CLINC150, but over the fairer custom last-token comparator the gains are much smaller: 0.34 and 0.46 F1 points, respectively.",
        "",
        "On clean proxy data, attention is not consistently better than custom last-token pooling: it is 0.05 points lower on clean BANKING77 and 0.76 points lower on clean CLINC150. On the new production selected-5 evaluation, attention is materially stronger than both default and custom last-token pooling.",
        "",
        "## Welch Tests From Summary Statistics",
        "",
        markdown_table(
            [
                "Dataset",
                "Comparison",
                "Mean diff F1 pp",
                "95% CI pp",
                "t(df)",
                "p",
                "Holm p",
                "Cohen d",
            ],
            [
                [
                    str(r["dataset"]),
                    str(r["comparison"]),
                    fmt_pct(float(r["diff"])),
                    f"[{fmt_pct(float(r['ci_low']))}, {fmt_pct(float(r['ci_high']))}]",
                    f"{float(r['t']):.3f} ({float(r['df']):.1f})",
                    fmt_p(float(r["p"])),
                    fmt_p(float(r["p_holm"])),
                    f"{float(r['cohen_d']):.2f}",
                ]
                for r in rows
            ],
        ),
        "",
        "## Interpretation for Rebuttal",
        "",
        "- Do not defend the abstract-level `+2.6-2.8 F1` statement as a pooling-only gain. The reviewer is right that those numbers use the default head as comparator, while the paper itself says C0 differs by head/training setup.",
        "- The fair proxy-data statement should be narrowed: attention pooling yields small, positive noisy-data gains over custom last-token pooling, but the Welch tests from the available summaries do not make both proxy gains significant after Holm correction.",
        "- The strongest response is to say we will revise the abstract/headline claim to distinguish architecture/head effects from pooling effects, add statistical testing, and report the industrial/production evaluation separately.",
        "- The production selected-5 result is the useful new evidence: attention beats custom last-token by 8.82 F1 points and default by 11.43 F1 points in the available 10-seed summary, with Holm-adjusted p-values below 0.001 in these Welch tests.",
        "",
        "## Suggested Rebuttal Language",
        "",
        "> We thank the reviewer for pointing out that our headline comparison conflated the custom-head change with the pooling operator. We agree and will revise the abstract and results discussion to separate these effects. Using the fairer custom-head last-token baseline, attention pooling improves noisy BANKING77 by 0.34 F1 points and noisy CLINC150 by 0.46 F1 points; these are modest gains and should not be described as the main +2.6-2.8 point effect. We have now added significance testing across the ten seeds. In addition, our new held-out production evaluation shows that the same attention-pooling configuration outperforms the custom last-token baseline by 8.82 F1 points and the default head by 11.43 F1 points, which is the stronger empirical support for the deployment claim. We will update the paper to report these tests and temper the headline claim accordingly.",
        "",
        "## Paired-Test Note",
        "",
        "A paired t-test should be run on matched seed-level F1 values before final paper revision. The current repository copy does not include the raw seed-level folders for aggregations 62/63/66/67/71, only the aggregate summary files. If those folders are restored, the paired test should compare the ten seed-wise differences for attention vs custom last-token on each dataset.",
        "",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()
