from typing import Optional
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
        
        # Classification linear layer
        self.classifier = nn.Linear(self.hidden_size, self.num_labels, bias=False)
        
        # Attention pooling weights (if using attention pooling)
        if pooling_strategy == "attention":
            self.attention_weights = nn.Linear(self.hidden_size, 1, bias=False)
    
    def pool_hidden_states(self, hidden_states, attention_mask=None):
        print(f"\n{'='*60}")
        print(f"[Classification Head] Pooling Strategy: {self.pooling_strategy}")
        print(f"[Classification Head] Input hidden states shape: {hidden_states.shape}")
        print(f"[Classification Head] Number of tokens being considered: {hidden_states.shape[1]}")
        
        if attention_mask is not None:
            num_valid_tokens = attention_mask.sum(dim=1).item() if attention_mask.shape[0] == 1 else attention_mask.sum(dim=1).tolist()
            print(f"[Classification Head] Valid (non-padding) tokens: {num_valid_tokens}")
        
        if self.pooling_strategy == "mean":
            if attention_mask is not None:
                # Mask out padding tokens
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                masked_states = hidden_states * mask
                sum_states = masked_states.sum(dim=1)  # [B, H]
                seq_lengths = mask.sum(dim=1)  # [B, 1]
                pooled = sum_states / (seq_lengths + 1e-9)
                print(f"[Classification Head] MEAN pooling: Averaging over {seq_lengths.item():.0f} valid tokens (all tokens contribute)")
            else:
                pooled = hidden_states.mean(dim=1)  # [B, H]
                print(f"[Classification Head] MEAN pooling: Averaging over ALL {hidden_states.shape[1]} tokens")
        
        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                # Set padding to very negative values
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                masked_states = hidden_states * mask + (1 - mask) * (-1e9)
                pooled = masked_states.max(dim=1)[0]  # [B, H]
                num_valid = attention_mask.sum(dim=1).item() if attention_mask.shape[0] == 1 else attention_mask.sum(dim=1).tolist()
                print(f"[Classification Head] MAX pooling: Taking max over {num_valid} valid tokens (all tokens considered)")
            else:
                pooled = hidden_states.max(dim=1)[0]  # [B, H]
                print(f"[Classification Head] MAX pooling: Taking max over ALL {hidden_states.shape[1]} tokens")
        
        elif self.pooling_strategy == "last":
            # Get last non-padding token
            if attention_mask is not None:
                # Find last non-pad token index
                seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths]  # [B, H]
                last_token_idx = seq_lengths.item() if seq_lengths.shape[0] == 1 else seq_lengths.tolist()
                print(f"[Classification Head] LAST pooling: Using token at index {last_token_idx} (only last token, NOT all tokens)")
            else:
                pooled = hidden_states[:, -1, :]  # [B, H]
                print(f"[Classification Head] LAST pooling: Using token at index {hidden_states.shape[1]-1} (only last token, NOT all tokens)")
        
        elif self.pooling_strategy == "attention":
            # Learned attention pooling
            attention_scores = self.attention_weights(hidden_states)  # [B, T, 1]
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                attention_scores = attention_scores + (1 - mask) * (-1e9)
            attention_weights = torch.softmax(attention_scores, dim=1)  # [B, T, 1]
            pooled = (hidden_states * attention_weights).sum(dim=1)  # [B, H]
            num_valid = attention_mask.sum(dim=1).item() if attention_mask is not None and attention_mask.shape[0] == 1 else (attention_mask.sum(dim=1).tolist() if attention_mask is not None else hidden_states.shape[1])
            print(f"[Classification Head] ATTENTION pooling: Weighted sum over {num_valid} tokens (all tokens contribute with learned weights)")
            if attention_weights.shape[0] == 1:
                print(f"[Classification Head] Attention weights (first 5 tokens): {attention_weights[0, :5, 0].detach().cpu().tolist()}")
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        print(f"[Classification Head] Pooled output shape: {pooled.shape}")
        print(f"{'='*60}\n")
        
        return pooled
    
    def forward(self, hidden_states, attention_mask=None, labels=None):
        print(f"\n{'#'*60}")
        print(f"[Classification Head] Forward pass started")
        print(f"[Classification Head] Input: {hidden_states.shape[1]} tokens, each with {hidden_states.shape[2]} dimensions")
        print(f"{'#'*60}")
        
        # Pool hidden states
        pooled = self.pool_hidden_states(hidden_states, attention_mask)  # [B, H]
        
        # Classification logits
        logits = self.classifier(pooled)  # [B, num_labels]
        
        print(f"[Classification Head] Logits shape: {logits.shape} (from pooled {pooled.shape[1]}-dim vector)")
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)  # [B, num_labels]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'logits': logits,
            'probs': probs,
            'loss': loss
        }

