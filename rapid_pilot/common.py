"""Shared pieces for the rapid pilot: data, model, residual-stream hooks and consistency metrics."""
import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer

from models.custom_llama_classification import LlamaClassificationHead

PAD_TOKEN = "<|reserved_special_token_15|>"


# ----------------------------
# Data
# ----------------------------

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        if PAD_TOKEN in tokenizer.get_vocab():
            tokenizer.pad_token = PAD_TOKEN
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize(records, tokenizer, max_length):
    encoded = tokenizer([r["text"] for r in records], truncation=True, max_length=max_length)
    return encoded["input_ids"]


def make_batches(token_ids, batch_size, shuffle, seed=0, bucket=50):
    """Index batches with little padding: sort by length inside shuffled buckets of `bucket` batches."""
    order = list(range(len(token_ids)))
    if shuffle:
        random.Random(seed).shuffle(order)
        chunk = batch_size * bucket
        batches = []
        for start in range(0, len(order), chunk):
            part = sorted(order[start:start + chunk], key=lambda i: len(token_ids[i]))
            batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
        random.Random(seed + 1).shuffle(batches)
        return batches
    order.sort(key=lambda i: len(token_ids[i]))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def collate(token_ids, batch, pad_id, device):
    width = max(len(token_ids[i]) for i in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, i in enumerate(batch):
        ids = token_ids[i]
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :len(ids)] = 1
    return input_ids.to(device), attention_mask.to(device)


# ----------------------------
# Model
# ----------------------------

def precision_setup():
    """(weight dtype, autocast dtype or None). bf16 where supported; fp32 weights + fp16 autocast otherwise."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, torch.bfloat16
        return torch.float32, torch.float16
    return torch.float32, None


def load_backbone(model_path, dtype, device):
    backbone = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
    return backbone.to(device)


class Classifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden_states=hidden.float(), attention_mask=attention_mask, labels=labels)


def build_head(config, num_labels, pooling):
    return LlamaClassificationHead(config=config, num_labels=num_labels, pooling_strategy=pooling, use_fft=False)


def decoder_layers(backbone):
    """The ModuleList of decoder layers of a (possibly PEFT-wrapped) LlamaModel."""
    for name, module in backbone.named_modules():
        if name.endswith("layers") and isinstance(module, nn.ModuleList):
            return module
    raise ValueError("No decoder layer list found")


def final_norm(backbone):
    for name, module in backbone.named_modules():
        if name.split(".")[-1] == "norm":
            return module
    raise ValueError("No final norm found")


def layer_module(backbone, layer):
    """`layer` is 1-indexed decoder-layer output, or "final" for the final-norm output the head reads."""
    if layer == "final":
        return final_norm(backbone)
    return decoder_layers(backbone)[int(layer) - 1]


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output, hidden):
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


# ----------------------------
# Residual-stream hooks
# ----------------------------

class PooledCapture:
    """Masked mean of a layer's output over real tokens, excluding position 0 (BOS / attention sink)."""

    def __init__(self):
        self.mask = None
        self.chunks = []

    def __call__(self, module, inputs, output):
        h = _hidden(output)[:, 1:].float()
        m = self.mask[:, 1:].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        self.chunks.append(pooled.cpu())

    def result(self):
        return torch.cat(self.chunks).numpy().astype(np.float64)


class ResidualProjection:
    """h <- h - Q Qᵀ (h - mu) for every token except position 0. Q: (d, r) orthonormal."""

    def __init__(self, basis, mean, device):
        self.q = torch.as_tensor(basis, dtype=torch.float32, device=device)
        self.mu = torch.as_tensor(mean, dtype=torch.float32, device=device)

    def __call__(self, module, inputs, output):
        h = _hidden(output)
        x = h[:, 1:].float()
        x = x - ((x - self.mu) @ self.q) @ self.q.T
        return _replace_hidden(output, torch.cat([h[:, :1], x.to(h.dtype)], dim=1))


@torch.no_grad()
def run_model(model, backbone, token_ids, pad_id, device, batch_size, autocast_dtype,
              captures=None, projection=None):
    """Predictions for `token_ids` in input order. captures: {layer: PooledCapture}; projection: (layer, hook)."""
    handles = []
    captures = captures or {}
    for layer, cap in captures.items():
        handles.append(layer_module(backbone, layer).register_forward_hook(cap))
    if projection is not None:
        handles.append(layer_module(backbone, projection[0]).register_forward_hook(projection[1]))

    batches = make_batches(token_ids, batch_size, shuffle=False)
    preds = np.empty(len(token_ids), dtype=np.int64)
    order = []
    try:
        model.eval()
        for batch in batches:
            input_ids, attention_mask = collate(token_ids, batch, pad_id, device)
            for cap in captures.values():
                cap.mask = attention_mask
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model(input_ids, attention_mask)["logits"]
            preds[batch] = logits.argmax(-1).cpu().numpy()
            order += batch
    finally:
        for handle in handles:
            handle.remove()

    # Captured rows follow batch order; restore input order.
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[np.asarray(order)] = np.arange(len(order))
    states = {layer: cap.result()[inverse] for layer, cap in captures.items()}
    return preds, states


# ----------------------------
# Metrics
# ----------------------------

def group_index(records):
    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[rec["group"]].append(i)
    return [np.asarray(v) for v in groups.values() if len(v) >= 2]


def group_outcomes(preds, labels, groups):
    """Per-group (consistent, all_correct) as float arrays."""
    consistent = np.array([float(len(set(preds[g].tolist())) == 1) for g in groups])
    all_correct = np.array([float(np.all(preds[g] == labels[g])) for g in groups])
    return consistent, all_correct


def score(group_preds, group_labels, groups, single_preds, single_labels):
    consistent, all_correct = group_outcomes(group_preds, group_labels, groups)
    preds = np.concatenate([group_preds, single_preds])
    labels = np.concatenate([group_labels, single_labels])
    return {
        "group_consistency": float(consistent.mean()),
        "group_all_correct": float(all_correct.mean()),
        "acc_group_members": float((group_preds == group_labels).mean()),
        "acc_singles": float((single_preds == single_labels).mean()),
        "acc_all": float((preds == labels).mean()),
        "macro_f1_all": float(f1_score(labels, preds, average="macro")),
        "n_groups": len(groups),
        "n_group_members": int(len(group_preds)),
        "n_singles": int(len(single_preds)),
    }


def paired_bootstrap(base, new, groups, group_labels, single_labels, n_boot=1000, seed=0):
    """95% CIs of new - base. base/new: dicts with 'group_preds', 'single_preds'. Resamples groups and singles."""
    rng = np.random.default_rng(seed)
    bc, ba = group_outcomes(base["group_preds"], group_labels, groups)
    nc, na = group_outcomes(new["group_preds"], group_labels, groups)
    b_single = (base["single_preds"] == single_labels).astype(float)
    n_single = (new["single_preds"] == single_labels).astype(float)
    b_member = np.array([(base["group_preds"][g] == group_labels[g]).sum() for g in groups], dtype=float)
    n_member = np.array([(new["group_preds"][g] == group_labels[g]).sum() for g in groups], dtype=float)
    sizes = np.array([len(g) for g in groups], dtype=float)

    diffs = defaultdict(list)
    for _ in range(n_boot):
        gi = rng.integers(0, len(groups), len(groups))
        si = rng.integers(0, len(single_labels), len(single_labels))
        diffs["group_consistency"].append(nc[gi].mean() - bc[gi].mean())
        diffs["group_all_correct"].append(na[gi].mean() - ba[gi].mean())
        total = sizes[gi].sum() + len(si)
        diffs["acc_all"].append(((n_member[gi].sum() + n_single[si].sum()) - (b_member[gi].sum() + b_single[si].sum())) / total)
    point = {
        "group_consistency": nc.mean() - bc.mean(),
        "group_all_correct": na.mean() - ba.mean(),
        "acc_all": ((n_member.sum() + n_single.sum()) - (b_member.sum() + b_single.sum())) / (sizes.sum() + len(single_labels)),
    }
    return {
        k: {"diff": float(point[k]), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
        for k, v in diffs.items()
    }
