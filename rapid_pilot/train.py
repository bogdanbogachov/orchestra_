"""Train the baseline LoRA classifier on singles (questions with no duplicate link).

Usage:
    python -m rapid_pilot.train --task_dir rapid_pilot/data/site --out rapid_pilot/runs/site
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import get_linear_schedule_with_warmup

from rapid_pilot.common import (
    Classifier, build_head, collate, load_backbone, load_tokenizer, make_batches, precision_setup,
    read_jsonl, run_model, tokenize,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="downloaded_models/downloaded_3_2_1b")
    parser.add_argument("--pooling", default="attention", choices=["attention", "mean", "last", "max"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train", type=int, default=None, help="Optional cap on training singles (smoke tests)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype, autocast_dtype = precision_setup()
    print(f"device={device} weights={weight_dtype} autocast={autocast_dtype}", flush=True)

    meta = json.load(open(os.path.join(args.task_dir, "meta.json")))
    num_labels = len(meta["label_names"])
    train = read_jsonl(os.path.join(args.task_dir, "train.jsonl"))[:args.max_train]
    val = read_jsonl(os.path.join(args.task_dir, "val_singles.jsonl"))

    tokenizer = load_tokenizer(args.model)
    train_ids, val_ids = tokenize(train, tokenizer, args.max_length), tokenize(val, tokenizer, args.max_length)
    train_labels = torch.tensor([r["label"] for r in train])
    val_labels = np.array([r["label"] for r in val])
    lengths = np.array([len(t) for t in train_ids])
    print(f"train={len(train)} val={len(val)} labels={num_labels} tokens median={np.median(lengths):.0f} "
          f"p90={np.percentile(lengths, 90):.0f} truncated={np.mean(lengths >= args.max_length):.3f}", flush=True)

    backbone = load_backbone(args.model, weight_dtype, device)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(backbone, lora)
    head = build_head(backbone.config, num_labels, args.pooling).to(device)
    model = Classifier(backbone, head)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params={sum(p.numel() for p in trainable):,}", flush=True)

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * steps_per_epoch * args.epochs),
                                                steps_per_epoch * args.epochs)
    scaler = torch.amp.GradScaler(enabled=autocast_dtype == torch.float16)

    log = {"args": vars(args), "epochs": []}
    best_acc, start = -1.0, time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        epoch_start, losses = time.time(), []
        for step, batch in enumerate(make_batches(train_ids, args.batch_size, shuffle=True, seed=args.seed + epoch)):
            input_ids, attention_mask = collate(train_ids, batch, tokenizer.pad_token_id, device)
            labels = train_labels[batch].to(device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                loss = model(input_ids, attention_mask, labels=labels)["loss"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{steps_per_epoch} loss {np.mean(losses[-100:]):.4f} "
                      f"elapsed {time.time() - start:.0f}s", flush=True)

        preds, _ = run_model(model, backbone, val_ids, tokenizer.pad_token_id, device,
                             args.eval_batch_size, autocast_dtype)
        acc = float((preds == val_labels).mean())
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": acc,
                 "epoch_seconds": time.time() - epoch_start}
        log["epochs"].append(entry)
        print(json.dumps(entry), flush=True)
        if acc > best_acc:
            best_acc = acc
            backbone.save_pretrained(os.path.join(args.out, "adapter"))
            torch.save(head.state_dict(), os.path.join(args.out, "head.pt"))

    log.update({
        "best_val_acc": best_acc,
        "train_seconds": time.time() - start,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
        "num_labels": num_labels,
        "pooling": args.pooling,
    })
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps({k: v for k, v in log.items() if k != "epochs"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
