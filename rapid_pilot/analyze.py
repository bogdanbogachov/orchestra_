"""Training-free nuisance-subspace projection pilot (Gates A and B of rapid_pilot_plan_2026-10-28.md).

1. Load the trained classifier (LoRA merged into the backbone).
2. Capture pooled residual-stream states of fit-split duplicate groups at the candidate layers.
3. Estimate subspaces per layer:
     geig   - proposed: top generalized eigenvectors of Σ_w u = λ (Σ_b + εI) u
     sw     - Σ_w-only top eigenvectors (NAP / GRIT style)
     pca    - top principal components of all fit states
     random - random orthonormal basis
4. Insert h <- h - QQᵀ(h - μ) at the layer for every token but BOS; sweep (method, layer, rank) on val.
5. Select per method on val, evaluate once on test with paired group-level bootstrap vs baseline.

Usage:
    python -m rapid_pilot.analyze --task_dir rapid_pilot/data/site --run_dir rapid_pilot/runs/site
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.linalg
import torch
from peft import PeftModel

from rapid_pilot.common import (
    Classifier, PooledCapture, ResidualProjection, build_head, group_index, load_backbone, load_tokenizer,
    paired_bootstrap, precision_setup, read_jsonl, run_model, score, tokenize,
)

GATE_A_MIN_INCONSISTENT = 0.10
GATE_B_MIN_GAIN = 0.03
GATE_B_MAX_ACC_DROP = 0.005


def fit_statistics(states, groups):
    """Within-group, between-group and total covariances of pooled states (n, d)."""
    within = np.concatenate([states[g] - states[g].mean(0) for g in groups])
    s_w = within.T @ within / (len(within) - len(groups))
    means = np.stack([states[g].mean(0) for g in groups])
    centred = means - means.mean(0)
    s_b = centred.T @ centred / (len(groups) - 1)
    all_centred = states - states.mean(0)
    s_t = all_centred.T @ all_centred / (len(states) - 1)
    return s_w, s_b, s_t, states.mean(0)


def top_eigvecs(a, r_max, b=None):
    d = a.shape[0]
    _, vecs = scipy.linalg.eigh(a, b, subset_by_index=[d - r_max, d - 1])
    return vecs[:, ::-1]


def orthonormal(vectors):
    q, _ = np.linalg.qr(vectors)
    return q


def subspaces(s_w, s_b, s_t, r_max, ridge, seed):
    """Nested orthonormal bases (d, r_max) per method; the first r columns span the rank-r subspace."""
    d = s_w.shape[0]
    eps = ridge * np.trace(s_b) / d
    rng = np.random.default_rng(seed)
    return {
        "geig": orthonormal(top_eigvecs(s_w, r_max, s_b + eps * np.eye(d))),
        "sw": orthonormal(top_eigvecs(s_w, r_max)),
        "pca": orthonormal(top_eigvecs(s_t, r_max)),
        "random": orthonormal(rng.standard_normal((d, r_max))),
    }


def variance_ratio(basis, s_w, s_b):
    """tr(QᵀΣ_wQ) / tr(QᵀΣ_bQ): how nuisance-dominated the removed subspace is."""
    return float(np.trace(basis.T @ s_w @ basis) / max(np.trace(basis.T @ s_b @ basis), 1e-12))


def gain_ok(result, base):
    gain = max(result["group_consistency"] - base["group_consistency"],
               result["group_all_correct"] - base["group_all_correct"])
    return gain >= GATE_B_MIN_GAIN and result["acc_all"] - base["acc_all"] >= -GATE_B_MAX_ACC_DROP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model", default="downloaded_models/downloaded_3_2_1b")
    parser.add_argument("--layers", nargs="+", default=None, help="1-indexed layers and/or 'final' (default: L/4 L/2 3L/4 final)")
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 16, 64, 256])
    parser.add_argument("--methods", nargs="+", default=["geig", "sw", "pca", "random"])
    parser.add_argument("--ridge", type=float, default=0.1, help="ε = ridge·tr(Σ_b)/d")
    parser.add_argument("--val_groups_max", type=int, default=800)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype, autocast_dtype = precision_setup()
    train_log = json.load(open(os.path.join(args.run_dir, "train_log.json")))
    meta = json.load(open(os.path.join(args.task_dir, "meta.json")))

    tokenizer = load_tokenizer(args.model)
    backbone = load_backbone(args.model, weight_dtype, device)
    backbone = PeftModel.from_pretrained(backbone, os.path.join(args.run_dir, "adapter")).merge_and_unload()
    head = build_head(backbone.config, len(meta["label_names"]), train_log["pooling"]).to(device)
    head.load_state_dict(torch.load(os.path.join(args.run_dir, "head.pt"), map_location=device, weights_only=True))
    model = Classifier(backbone, head)
    n_layers = backbone.config.num_hidden_layers
    layers = args.layers or [str(n_layers // 4), str(n_layers // 2), str(3 * n_layers // 4), "final"]

    rng = np.random.default_rng(args.seed)

    def load_split(name, groups_max=None):
        records = read_jsonl(os.path.join(args.task_dir, f"{name}.jsonl"))
        if groups_max is not None:
            gids = sorted({r["group"] for r in records})
            keep = set(rng.choice(gids, size=min(groups_max, len(gids)), replace=False).tolist())
            records = [r for r in records if r["group"] in keep]
        return records

    def encode(records):
        return tokenize(records, tokenizer, args.max_length), np.array([r["label"] for r in records])

    fit = load_split("fit_groups")
    evals = {
        "val": (load_split("val_groups", args.val_groups_max), load_split("val_singles")),
        "test": (load_split("test_groups"), load_split("test_singles")),
    }
    encoded = {k: (encode(g), encode(s), group_index(g)) for k, (g, s) in evals.items()}
    pad_id = tokenizer.pad_token_id

    def evaluate(split, projection=None):
        (g_ids, g_labels), (s_ids, s_labels), groups = encoded[split]
        hook = None if projection is None else (projection[0], projection[1])
        g_preds, _ = run_model(model, backbone, g_ids, pad_id, device, args.batch_size, autocast_dtype, projection=hook)
        s_preds, _ = run_model(model, backbone, s_ids, pad_id, device, args.batch_size, autocast_dtype, projection=hook)
        return score(g_preds, g_labels, groups, s_preds, s_labels), {"group_preds": g_preds, "single_preds": s_preds}

    # Fit statistics at every candidate layer in one pass.
    fit_ids, _ = encode(fit)
    fit_groups = group_index(fit)
    captures = {layer: PooledCapture() for layer in layers}
    run_model(model, backbone, fit_ids, pad_id, device, args.batch_size, autocast_dtype, captures=captures)
    stats, bases, diagnostics = {}, {}, {}
    r_max = max(args.ranks)
    for layer in layers:
        states = captures[layer].result()
        s_w, s_b, s_t, mu = fit_statistics(states, fit_groups)
        stats[layer] = mu
        bases[layer] = subspaces(s_w, s_b, s_t, r_max, args.ridge, args.seed)
        diagnostics[layer] = {
            "tr_within_over_between": float(np.trace(s_w) / np.trace(s_b)),
            "removed_within_over_between": {
                m: {r: variance_ratio(bases[layer][m][:, :r], s_w, s_b) for r in args.ranks} for m in args.methods
            },
        }
        print(f"layer {layer}: tr(Sw)/tr(Sb)={diagnostics[layer]['tr_within_over_between']:.3f}", flush=True)
    print(f"fit capture done: {len(fit)} texts, {len(fit_groups)} groups, {time.time() - start:.0f}s", flush=True)

    def projection_for(method, layer, rank):
        return layer, ResidualProjection(bases[layer][method][:, :rank], stats[layer], device)

    # Baseline and validation sweep.
    base_val, _ = evaluate("val")
    print("baseline val", json.dumps(base_val), flush=True)
    grid = []
    for method in args.methods:
        for layer in layers:
            for rank in args.ranks:
                result, _ = evaluate("val", projection_for(method, layer, rank))
                grid.append({"method": method, "layer": layer, "rank": rank, **result})
                print(f"val {method} L{layer} r{rank}: cons={result['group_consistency']:.4f} "
                      f"allc={result['group_all_correct']:.4f} acc={result['acc_all']:.4f}", flush=True)

    def select(method):
        rows = [g for g in grid if g["method"] == method]
        eligible = [g for g in rows if g["acc_all"] - base_val["acc_all"] >= -GATE_B_MAX_ACC_DROP] or rows
        return max(eligible, key=lambda g: (g["group_all_correct"], g["acc_all"]))

    selected = {m: select(m) for m in args.methods}

    # Test: baseline and each method's selected configuration, once.
    base_test, base_test_preds = evaluate("test")
    (_, test_g_labels), (_, test_s_labels), test_groups = encoded["test"]
    test_results = {}
    for method, sel in selected.items():
        result, preds = evaluate("test", projection_for(method, sel["layer"], sel["rank"]))
        test_results[method] = {
            "layer": sel["layer"], "rank": sel["rank"], **result,
            "vs_baseline": paired_bootstrap(base_test_preds, preds, test_groups, test_g_labels, test_s_labels,
                                            args.bootstrap, args.seed),
        }

    # Gates.
    gate_a = {
        "val_inconsistent": 1 - base_val["group_consistency"],
        "test_inconsistent": 1 - base_test["group_consistency"],
    }
    gate_a["pass"] = gate_a["val_inconsistent"] >= GATE_A_MIN_INCONSISTENT
    gate_b = {}
    if "geig" in selected:
        g = selected["geig"]
        matched = {m: next(x for x in grid if x["method"] == m and x["layer"] == g["layer"] and x["rank"] == g["rank"])
                   for m in ("sw", "random") if m in args.methods}
        gate_b = {
            "b1_gain_without_accuracy_loss": gain_ok(g, base_val),
            "b2_beats_matched": {m: g["group_all_correct"] > x["group_all_correct"] for m, x in matched.items()},
            "selected_layer_is_final_only": g["layer"] == "final",
        }
        gate_b["pass_this_task"] = gate_b["b1_gain_without_accuracy_loss"] and all(gate_b["b2_beats_matched"].values())

    report = {
        "task": meta["task"],
        "labels": meta["label_names"],
        "layers": layers,
        "args": vars(args),
        "train": {k: train_log[k] for k in ("best_val_acc", "train_seconds", "peak_gpu_mem_gb", "pooling")},
        "diagnostics": diagnostics,
        "baseline": {"val": base_val, "test": base_test},
        "val_grid": grid,
        "selected_on_val": selected,
        "test": test_results,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "analyze_seconds": time.time() - start,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
    }
    with open(os.path.join(args.run_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        f"# Rapid pilot: {meta['task']}",
        "",
        f"Gate A (val inconsistent ≥ {GATE_A_MIN_INCONSISTENT:.0%}): **{'PASS' if gate_a['pass'] else 'FAIL'}** "
        f"(val {gate_a['val_inconsistent']:.3f}, test {gate_a['test_inconsistent']:.3f})",
        f"Gate B (this task): **{'PASS' if gate_b.get('pass_this_task') else 'FAIL'}** {json.dumps(gate_b)}",
        "",
        "| Test | layer | rank | consistency | all-correct | acc | macro-F1 | Δcons [95% CI] | Δall-correct [95% CI] | Δacc [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| baseline | – | – | {base_test['group_consistency']:.4f} | {base_test['group_all_correct']:.4f} | "
        f"{base_test['acc_all']:.4f} | {base_test['macro_f1_all']:.4f} | | | |",
    ]
    for method, r in test_results.items():
        cells = []
        for key in ("group_consistency", "group_all_correct", "acc_all"):
            v = r["vs_baseline"][key]
            cells.append(f"{v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
        lines.append(f"| {method} | {r['layer']} | {r['rank']} | {r['group_consistency']:.4f} | "
                     f"{r['group_all_correct']:.4f} | {r['acc_all']:.4f} | {r['macro_f1_all']:.4f} | " + " | ".join(cells) + " |")
    with open(os.path.join(args.run_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
