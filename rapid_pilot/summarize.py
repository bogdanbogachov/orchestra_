"""Combine per-task reports into the pilot go/no-go decision.

Usage:
    python -m rapid_pilot.summarize --runs rapid_pilot/runs
"""
import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="rapid_pilot/runs")
    args = parser.parse_args()

    reports = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.runs, "*", "report.json")))]
    if not reports:
        print(f"No report.json under {args.runs}")
        return

    print("| task | labels | base acc | base cons | base all-correct | Gate A | geig L/r | Δcons | Δall-correct | Δacc | Gate B | train s | analyze s | peak GB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    passing_b = 0
    for r in reports:
        base = r["baseline"]["test"]
        geig = r["test"].get("geig", {})
        vs = geig.get("vs_baseline", {})
        fmt = lambda k: f"{vs[k]['diff']:+.4f} [{vs[k]['ci95'][0]:+.4f},{vs[k]['ci95'][1]:+.4f}]" if k in vs else "–"
        gate_b = r["gate_b"].get("pass_this_task", False)
        passing_b += int(gate_b and r["gate_a"]["pass"])
        peak = max(x for x in (r["train"]["peak_gpu_mem_gb"], r["peak_gpu_mem_gb"]) if x is not None) \
            if r["peak_gpu_mem_gb"] is not None else None
        print(f"| {r['task']} | {len(r['labels'])} | {base['acc_all']:.4f} | {base['group_consistency']:.4f} | "
              f"{base['group_all_correct']:.4f} | {'PASS' if r['gate_a']['pass'] else 'FAIL'} | "
              f"{geig.get('layer', '–')}/{geig.get('rank', '–')} | {fmt('group_consistency')} | "
              f"{fmt('group_all_correct')} | {fmt('acc_all')} | {'PASS' if gate_b else 'FAIL'} | "
              f"{r['train']['train_seconds']:.0f} | {r['analyze_seconds']:.0f} | "
              f"{'–' if peak is None else f'{peak:.1f}'} |")

    print()
    print(f"Tasks passing Gates A and B: {passing_b}/{len(reports)}")
    print("GO (B3: ≥2 tasks)" if passing_b >= 2 else "NO-GO / inconclusive (B3 requires ≥2 tasks)")
    if any(r["gate_b"].get("selected_layer_is_final_only") for r in reports):
        print("Warning: geig selected 'final' on some task; the in-stream claim reduces to post-hoc LDA there.")


if __name__ == "__main__":
    main()
