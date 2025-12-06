import torch
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig


class LlamaClassificationHead(nn.Module):
    def __init__(self, config: LlamaConfig, num_labels: int, pooling_strategy: str = "mean"):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy
        self.hidden_size = config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_labels, bias=False)
        if pooling_strategy == "attention":
            self.attention_weights = nn.Linear(self.hidden_size, 1, bias=False)
    
    def pool_hidden_states(self, hidden_states, attention_mask=None):
        if self.pooling_strategy == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                masked_states = hidden_states * mask
                sum_states = masked_states.sum(dim=1)  # [B, H]
                seq_lengths = mask.sum(dim=1)  # [B, 1]
                pooled = sum_states / (seq_lengths + 1e-9)
            else:
                pooled = hidden_states.mean(dim=1)  # [B, H]
        
        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                masked_states = hidden_states * mask + (1 - mask) * (-1e9)
                pooled = masked_states.max(dim=1)[0]  # [B, H]
            else:
                pooled = hidden_states.max(dim=1)[0]  # [B, H]
        
        elif self.pooling_strategy == "last":
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths]  # [B, H]
            else:
                pooled = hidden_states[:, -1, :]  # [B, H]
        
        elif self.pooling_strategy == "attention":
            attention_scores = self.attention_weights(hidden_states)  # [B, T, 1]
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                attention_scores = attention_scores + (1 - mask) * (-1e9)
            attention_weights = torch.softmax(attention_scores, dim=1)  # [B, T, 1]
            pooled = (hidden_states * attention_weights).sum(dim=1)  # [B, H]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled
    
    def forward(self, hidden_states, attention_mask=None, labels=None):
        pooled = self.pool_hidden_states(hidden_states, attention_mask)  # [B, H]
        logits = self.classifier(pooled)  # [B, num_labels]
        probs = torch.nn.functional.softmax(logits, dim=-1)  # [B, num_labels]
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'logits': logits,
            'probs': probs,
            'loss': loss
        }
