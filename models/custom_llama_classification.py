import torch
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig


class LlamaClassificationHead(nn.Module):
    def __init__(self, config: LlamaConfig, num_labels: int, pooling_strategy: str = "mean", use_fft: bool = False):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy
        self.use_fft = use_fft
        self.hidden_size = config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_labels, bias=False)
        if pooling_strategy == "attention":
            self.attention_weights = nn.Linear(self.hidden_size, 1, bias=False)

    @staticmethod
    def apply_fft_filter(hidden_states):
        fft_result = torch.fft.fft(hidden_states, dim=1)

        seq_length = hidden_states.size(1)
        rows_to_keep = max(1, int(seq_length * 0.3))
        
        mask = torch.zeros(seq_length, device=hidden_states.device, dtype=torch.float32)
        mask[:rows_to_keep] = 1.0
        
        if seq_length > rows_to_keep:
            neg_start = seq_length - rows_to_keep + 1
            if neg_start < seq_length:
                mask[neg_start:] = 1.0

        mask = mask.unsqueeze(0).unsqueeze(-1)
        fft_filtered = fft_result * mask
        
        filtered_states = torch.fft.ifft(fft_filtered, dim=1)
        filtered_states = filtered_states.real
        
        return filtered_states
    
    def pool_hidden_states(self, hidden_states, attention_mask=None, input_ids=None):
        if self.pooling_strategy == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                masked_states = hidden_states * mask
                sum_states = masked_states.sum(dim=1)
                seq_lengths = mask.sum(dim=1)
                pooled = sum_states / (seq_lengths + 1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
        
        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                masked_states = hidden_states * mask + (1 - mask) * (-1e9)
                pooled = masked_states.max(dim=1)[0]
            else:
                pooled = hidden_states.max(dim=1)[0]
        
        elif self.pooling_strategy == "last":
            # Match default head behavior: use pad_token_id to find last non-padding token
            if input_ids is not None and self.config.pad_token_id is not None:
                batch_size = input_ids.shape[0]
                non_pad_mask = (input_ids != self.config.pad_token_id).to(hidden_states.device, torch.int32)
                token_indices = torch.arange(input_ids.shape[-1], device=hidden_states.device, dtype=torch.int32)
                last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
                batch_indices = torch.arange(batch_size, device=hidden_states.device)
                pooled = hidden_states[batch_indices, last_non_pad_token]
            elif attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths]
            else:
                pooled = hidden_states[:, -1, :]
        
        elif self.pooling_strategy == "attention":
            attention_scores = self.attention_weights(hidden_states)
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                attention_scores = attention_scores + (1 - mask) * (-1e9)
            attention_weights = torch.softmax(attention_scores, dim=1)
            pooled = (hidden_states * attention_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled
    
    def forward(self, hidden_states, attention_mask=None, labels=None, input_ids=None):
        if self.use_fft:
            hidden_states = self.apply_fft_filter(hidden_states)
        
        pooled = self.pool_hidden_states(hidden_states, attention_mask, input_ids)
        logits = self.classifier(pooled)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'logits': logits,
            'probs': probs,
            'loss': loss
        }
