import torch
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig


class LlamaClassificationHead(nn.Module):
    def __init__(self, config: LlamaConfig, num_labels: int, pooling_strategy: str = "mean", use_fft: bool = False, fft_adaptive: bool = True):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.pooling_strategy = pooling_strategy
        self.use_fft = use_fft
        self.fft_adaptive = fft_adaptive  # True for learnable adaptive, False for fixed 50% cutoff
        self.hidden_size = config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_labels, bias=False)
        if pooling_strategy == "attention":
            self.attention_weights = nn.Linear(self.hidden_size, 1, bias=False)
        
        # Learnable adaptive FFT parameters (only if adaptive is enabled)
        if use_fft and fft_adaptive:
            # Small network to predict cutoff ratio from sequence statistics
            # Input: [mean_magnitude, std_magnitude, normalized_seq_len]
            # Output: cutoff ratio [0, 1] via sigmoid
            self.fft_adaptive_network = nn.Sequential(
                nn.Linear(3, 16),  # Input: [mean_mag, std_mag, seq_len_norm]
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # Output: cutoff ratio [0, 1]
            )

    @staticmethod
    def apply_fft_filter(hidden_states):
        """
        Static FFT filter with fixed 50% cutoff (kept for backward compatibility).
        """
        fft_result = torch.fft.fft(hidden_states, dim=1)

        seq_length = hidden_states.size(1)
        rows_to_keep = max(1, int(seq_length * 0.5))

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
    
    def apply_adaptive_fft_filter_learnable(self, hidden_states):
        """
        Learnable adaptive FFT filter.
        Uses a small network to predict optimal cutoff ratio based on sequence statistics.
        The network learns to adapt the frequency cutoff dynamically for each input.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            Filtered hidden states with same shape
        """
        fft_result = torch.fft.fft(hidden_states, dim=1)
        
        # Compute statistics from FFT magnitude
        magnitude = torch.abs(fft_result)  # [batch, seq_len, hidden_dim]
        mean_mag = magnitude.mean(dim=[1, 2])  # [batch] - average magnitude across all frequencies and dimensions
        std_mag = magnitude.std(dim=[1, 2])    # [batch] - std of magnitude
        seq_len_norm = hidden_states.size(1) / 512.0  # Normalize by typical max length
        
        # Predict cutoff ratio for each sample in batch
        # Stack statistics: [mean_magnitude, std_magnitude, normalized_seq_len]
        stats = torch.stack([mean_mag, std_mag, torch.full_like(mean_mag, seq_len_norm)], dim=1)  # [batch, 3]
        cutoff_ratios = self.fft_adaptive_network(stats).squeeze(-1)  # [batch] - cutoff ratio in [0, 1]
        
        # Create adaptive mask for each sample
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        mask = torch.zeros(batch_size, seq_len, device=hidden_states.device, dtype=torch.float32)
        
        for b in range(batch_size):
            rows_to_keep = max(1, int(seq_len * cutoff_ratios[b].item()))
            mask[b, :rows_to_keep] = 1.0
            
            # Keep negative frequencies (symmetric in FFT)
            if seq_len > rows_to_keep:
                neg_start = seq_len - rows_to_keep + 1
                if neg_start < seq_len:
                    mask[b, neg_start:] = 1.0
        
        mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        fft_filtered = fft_result * mask
        
        filtered_states = torch.fft.ifft(fft_filtered, dim=1)
        return filtered_states.real
    
    def pool_hidden_states(self, hidden_states, attention_mask=None):
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
            if attention_mask is not None:
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
    
    def forward(self, hidden_states, attention_mask=None, labels=None):
        if self.use_fft:
            if self.fft_adaptive:
                # Use learnable adaptive filtering
                hidden_states = self.apply_adaptive_fft_filter_learnable(hidden_states)
            else:
                # Use fixed 50% cutoff filtering
                hidden_states = self.apply_fft_filter(hidden_states)
        
        # Custom head style: pool first, then apply linear layer
        pooled = self.pool_hidden_states(hidden_states, attention_mask)
        logits = self.classifier(pooled)  # [batch, num_labels]
        
        # Softmax identical to default head: torch.nn.functional.softmax(logits, dim=-1)
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
