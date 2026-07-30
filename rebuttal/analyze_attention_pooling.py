#!/usr/bin/env python
"""Analyze token weights from the custom attention-pooling head.

This is a qualitative/quantitative helper for rebuttal analysis. It visualizes
the learned pooling weights, not transformer self-attention.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CONFIG
from models.custom_llama_classification import LlamaClassificationHead


def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_num_labels(classifier_state: Dict[str, torch.Tensor]) -> int:
    weight = classifier_state.get("classifier.weight")
    if weight is None:
        raise ValueError("classifier.pt does not contain classifier.weight")
    return int(weight.shape[0])


def load_attention_model(args):
    model_path = args.model_path or CONFIG["paths"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = CONFIG["model"].get("pad_token", tokenizer.eos_token)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.torch_dtype, torch.float32)

    base_model = AutoModel.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map=args.device_map,
    )
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    base_model = PeftModel.from_pretrained(base_model, args.adapter_path)

    classifier_path = Path(args.adapter_path) / "classifier.pt"
    if not classifier_path.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {classifier_path}")

    classifier_state = torch.load(classifier_path, map_location=base_model.device)
    num_labels = args.num_labels or infer_num_labels(classifier_state)
    classifier = LlamaClassificationHead(
        config=base_model.config,
        num_labels=num_labels,
        pooling_strategy="attention",
        use_fft=args.use_fft,
    ).to(base_model.device)
    classifier.load_state_dict(classifier_state, strict=True)
    classifier.eval()
    base_model.eval()
    return base_model, tokenizer, classifier


def normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def find_clean_span(noisy_text: str, clean_text: Optional[str]) -> Optional[Tuple[int, int]]:
    if not clean_text:
        return None
    noisy_lower = noisy_text.lower()
    clean = re.sub(r"\s+", " ", str(clean_text)).strip()
    candidates = [clean, clean.rstrip(".?!,;:")]
    for candidate in candidates:
        if not candidate:
            continue
        start = noisy_lower.find(candidate.lower())
        if start >= 0:
            return start, start + len(candidate)
    return None


def token_records(
    tokenizer,
    input_ids: torch.Tensor,
    offsets: Iterable[Tuple[int, int]],
    attention_mask: torch.Tensor,
    weights: torch.Tensor,
    clean_span: Optional[Tuple[int, int]],
) -> List[Dict]:
    ids = input_ids.detach().cpu().tolist()
    mask = attention_mask.detach().cpu().tolist()
    w = weights.detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    rows = []
    for idx, (tok, offset, is_attended, weight) in enumerate(zip(tokens, offsets, mask, w)):
        start, end = int(offset[0]), int(offset[1])
        if not is_attended:
            region = "pad"
        elif start == end:
            region = "special"
        elif clean_span is None:
            region = "content"
        else:
            midpoint = (start + end) / 2.0
            region = "original" if clean_span[0] <= midpoint < clean_span[1] else "noise"
        rows.append(
            {
                "position": idx,
                "token": tok,
                "char_start": start,
                "char_end": end,
                "region": region,
                "weight": float(weight),
            }
        )
    return rows


def analyze_one(base_model, tokenizer, classifier, text: str, clean_text: Optional[str], max_length: int):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping").squeeze(0).tolist()
    encoded = {k: v.to(base_model.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = base_model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        hidden_states = outputs.last_hidden_state
        if classifier.use_fft:
            hidden_states = classifier.apply_fft_filter(hidden_states)
        scores = classifier.attention_weights(hidden_states)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        scores = scores + (1 - mask) * (-1e9)
        weights = torch.softmax(scores, dim=1).squeeze(0).squeeze(-1)
        logits = classifier.classifier((hidden_states * weights.view(1, -1, 1)).sum(dim=1))
        pred = int(logits.argmax(dim=-1).item())

    clean_span = find_clean_span(text, clean_text)
    rows = token_records(
        tokenizer=tokenizer,
        input_ids=encoded["input_ids"].squeeze(0),
        offsets=offsets,
        attention_mask=encoded["attention_mask"].squeeze(0),
        weights=weights,
        clean_span=clean_span,
    )
    return pred, clean_span, rows


def summarize_example(rows: List[Dict], top_k: int) -> Dict:
    content = [r for r in rows if r["region"] not in {"pad", "special"}]
    original = [r for r in content if r["region"] == "original"]
    noise = [r for r in content if r["region"] == "noise"]
    top = sorted(content, key=lambda r: r["weight"], reverse=True)[:top_k]

    def mass(items):
        return sum(r["weight"] for r in items)

    def avg(items):
        return mean([r["weight"] for r in items]) if items else None

    return {
        "content_tokens": len(content),
        "original_tokens": len(original),
        "noise_tokens": len(noise),
        "original_mass": mass(original),
        "noise_mass": mass(noise),
        "special_mass": mass([r for r in rows if r["region"] == "special"]),
        "original_mean_weight": avg(original),
        "noise_mean_weight": avg(noise),
        "top_k_original_fraction": (
            sum(1 for r in top if r["region"] == "original") / len(top) if top else None
        ),
        "top_tokens": " ".join(f"{r['token']}:{r['weight']:.4f}:{r['region']}" for r in top),
    }


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, help="Path to a trained custom_head directory.")
    parser.add_argument("--dataset", required=True, help="Noisy/evaluation JSON list with text/label fields.")
    parser.add_argument("--clean-dataset", default=None, help="Aligned clean JSON list. Enables original-vs-noise analysis.")
    parser.add_argument("--output-dir", default="rebuttal/attention_pooling_analysis")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num-labels", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-fft", action="store_true")
    parser.add_argument("--device-map", default=CONFIG["model"].get("device_map", "auto"))
    parser.add_argument("--torch-dtype", default=CONFIG["model"].get("torch_dtype", "float32"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noisy = load_json(args.dataset)
    clean = load_json(args.clean_dataset) if args.clean_dataset else None
    if clean is not None and len(clean) != len(noisy):
        raise ValueError("--clean-dataset must be aligned with --dataset and have the same length")

    base_model, tokenizer, classifier = load_attention_model(args)

    summary_rows = []
    token_rows = []
    max_samples = min(args.max_samples, len(noisy))
    matched_spans = 0

    for idx in range(max_samples):
        text = str(noisy[idx]["text"])
        clean_text = str(clean[idx]["text"]) if clean is not None else None
        pred, clean_span, rows = analyze_one(base_model, tokenizer, classifier, text, clean_text, args.max_length)
        if clean_span is not None:
            matched_spans += 1
        ex_summary = summarize_example(rows, args.top_k)
        ex_summary.update(
            {
                "index": idx,
                "label": noisy[idx].get("label"),
                "pred": pred,
                "clean_span_found": clean_span is not None,
                "text": text,
                "clean_text": clean_text or "",
            }
        )
        summary_rows.append(ex_summary)
        for row in rows:
            row = dict(row)
            row["index"] = idx
            token_rows.append(row)

    write_csv(out_dir / "attention_summary.csv", summary_rows)
    write_csv(out_dir / "attention_tokens.csv", token_rows)

    comparable = [r for r in summary_rows if r["clean_span_found"] and r["noise_tokens"] > 0]
    report = {
        "adapter_path": args.adapter_path,
        "dataset": args.dataset,
        "clean_dataset": args.clean_dataset,
        "samples_analyzed": max_samples,
        "clean_spans_found": matched_spans,
        "comparable_original_noise_examples": len(comparable),
    }
    for field in ["original_mass", "noise_mass", "original_mean_weight", "noise_mean_weight", "top_k_original_fraction"]:
        values = [r[field] for r in comparable if r[field] is not None]
        report[field] = mean(values) if values else None

    with open(out_dir / "attention_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = [
        "# Attention Pooling Analysis",
        "",
        "These are pooling-head weights, not transformer self-attention explanations.",
        "",
        f"- Samples analyzed: {max_samples}",
        f"- Clean spans found: {matched_spans}",
        f"- Comparable original/noise examples: {len(comparable)}",
    ]
    if comparable:
        md.extend(
            [
                f"- Mean attention mass on original query span: {report['original_mass']:.4f}",
                f"- Mean attention mass on injected noise: {report['noise_mass']:.4f}",
                f"- Mean per-token weight on original span: {report['original_mean_weight']:.6f}",
                f"- Mean per-token weight on injected noise: {report['noise_mean_weight']:.6f}",
                f"- Mean top-{args.top_k} original-token fraction: {report['top_k_original_fraction']:.4f}",
            ]
        )
    md.extend(["", "## Top-Token Examples", ""])
    for row in summary_rows[: min(10, len(summary_rows))]:
        md.extend(
            [
                f"### Example {row['index']}",
                "",
                f"- Text: {row['text']}",
                f"- Label/pred: {row['label']} / {row['pred']}",
                f"- Top tokens: `{row['top_tokens']}`",
                "",
            ]
        )
    (out_dir / "attention_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Wrote {out_dir / 'attention_report.md'}")
    print(f"Wrote {out_dir / 'attention_summary.csv'}")
    print(f"Wrote {out_dir / 'attention_tokens.csv'}")


if __name__ == "__main__":
    main()
