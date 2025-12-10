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
            # Learnable attention weight vector (shape [1, hidden_size]) that computes importance scores for tokens
            self.attention_weights = nn.Linear(self.hidden_size, 1, bias=False)

    @staticmethod
    def apply_fft_filter(hidden_states):
        fft_result = torch.fft.fft(hidden_states, dim=1)

        seq_length = hidden_states.size(1)
        rows_to_keep = max(1, int(seq_length * 0.3))
        
        mask = torch.zeros(seq_length, device=hidden_states.device, dtype=torch.float32)
        mask[:rows_to_keep] = 1.0
        
        # Keep corresponding negative frequencies to maintain Hermitian symmetry
        # Negative frequencies are at indices N-k, where k are the positive frequencies
        # We need to keep indices from seq_length-rows_to_keep+1 to seq_length-1
        # But avoid double-counting if there's overlap
        if seq_length > rows_to_keep:
            neg_start = seq_length - rows_to_keep + 1
            if neg_start < seq_length:
                mask[neg_start:] = 1.0

        mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
        fft_filtered = fft_result * mask
        
        # Apply inverse FFT to convert back to time domain
        filtered_states = torch.fft.ifft(fft_filtered, dim=1)  # [B, T, H] (complex)
        
        # Take the real part (should be real since input was real and symmetry is maintained)
        filtered_states = filtered_states.real  # [B, T, H]
        
        return filtered_states
    
    def pool_hidden_states(self, hidden_states, attention_mask=None):
        if self.pooling_strategy == "mean":
            if attention_mask is not None:
                # Reshape mask to [B, T, 1] for broadcasting with hidden_states [B, T, H]
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                # Zero out padding positions (multiply by 0) to ignore them in pooling
                masked_states = hidden_states * mask
                # Sum all real tokens along sequence dimension → [B, H]
                sum_states = masked_states.sum(dim=1)  # [B, H]
                # Count number of real tokens per sequence (not padded length) → [B, 1]
                seq_lengths = mask.sum(dim=1)  # [B, 1]
                # Divide by actual token count to get mean (1e-9 prevents division by zero)
                pooled = sum_states / (seq_lengths + 1e-9)
            else:
                pooled = hidden_states.mean(dim=1)  # [B, H]
        
        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                # Reshape mask to [B, T, 1] for broadcasting with hidden_states [B, T, H]
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                # Zero padding positions and set them to very negative value so max() ignores them
                masked_states = hidden_states * mask + (1 - mask) * (-1e9)
                # Take maximum along sequence dimension → [B, H]
                pooled = masked_states.max(dim=1)[0]  # [B, H]
            else:
                pooled = hidden_states.max(dim=1)[0]  # [B, H]
        
        elif self.pooling_strategy == "last":
            if attention_mask is not None:
                # Calculate index of last real token (count - 1 for 0-indexing) → [B]
                seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
                # Create batch indices [0, 1, 2, ...] for advanced indexing
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                # Extract last real token for each sequence using advanced indexing → [B, H]
                pooled = hidden_states[batch_indices, seq_lengths]  # [B, H]
            else:
                pooled = hidden_states[:, -1, :]  # [B, H]
        
        elif self.pooling_strategy == "attention":
            # Compute raw importance score for each token by applying learnable attention vector → [B, T, 1]
            attention_scores = self.attention_weights(hidden_states)  # [B, T, 1]
            if attention_mask is not None:
                # Reshape mask to [B, T, 1] for broadcasting with attention_scores
                mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
                # Set padding positions to very negative value so they get 0 weight after softmax
                attention_scores = attention_scores + (1 - mask) * (-1e9)
            # Convert scores to probabilities (sum to 1.0 per sequence) → [B, T, 1]
            attention_weights = torch.softmax(attention_scores, dim=1)  # [B, T, 1]
            # Weighted sum: multiply each token's hidden state by its weight, then sum → [B, H]
            pooled = (hidden_states * attention_weights).sum(dim=1)  # [B, H]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        return pooled
    
    def forward(self, hidden_states, attention_mask=None, labels=None):
        if self.use_fft:
            hidden_states = self.apply_fft_filter(hidden_states)
        
        pooled = self.pool_hidden_states(hidden_states, attention_mask)  # [B, H]
        # Convert pooled hidden features to raw class scores → [B, num_labels]
        logits = self.classifier(pooled)  # [B, num_labels]
        # Convert logits to probability distribution over classes (sums to 1.0) → [B, num_labels]
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
