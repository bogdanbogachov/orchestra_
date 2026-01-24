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
            # Small network to predict cutoff ratio from noise-detection statistics
            # Input: [high_freq_ratio, spectral_flatness, low_freq_concentration, normalized_seq_len]
            # Output: cutoff ratio [0, 1] via sigmoid
            self.fft_adaptive_network = nn.Sequential(
                nn.Linear(4, 16),  # Input: [high_freq_ratio, spectral_flatness, low_freq_concentration, seq_len_norm]
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # Output: cutoff ratio [0, 1]
            )
            # Initialize to output ~0.5 (similar to fixed filtering)
            # This gives a reasonable starting point instead of random initialization
            with torch.no_grad():
                # Set bias of final linear layer so sigmoid outputs ~0.5 initially
                # sigmoid(0) ≈ 0.5, so we want the input to sigmoid to be ~0
                self.fft_adaptive_network[-2].bias.data.fill_(0.0)
                # Initialize weights to be small so output is close to 0.5
                self.fft_adaptive_network[-2].weight.data.normal_(0.0, 0.01)

    @staticmethod
    def apply_fft_filter(hidden_states):
        """
        Static FFT filter with fixed 50% cutoff (kept for backward compatibility).
        Returns: (filtered_states, cutoff_ratio)
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

        cutoff_ratio = 0.5  # Fixed for this method
        return filtered_states, cutoff_ratio
    
    def apply_adaptive_fft_filter_learnable(self, hidden_states):
        """
        Learnable adaptive FFT filter.
        Uses a small network to predict optimal cutoff ratio based on noise-detection statistics.
        Uses shared cutoff per batch for stability, harder mask, and no epsilon.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        
        Returns:
            tuple: (filtered_states, cutoff_ratio) where cutoff_ratio is a scalar for regularization
        """
        fft_result = torch.fft.fft(hidden_states, dim=1)
        
        # Compute noise-detection statistics from FFT magnitude
        magnitude = torch.abs(fft_result)  # [batch, seq_len, hidden_dim]
        seq_len = hidden_states.size(1)
        
        # Average across hidden dimensions to get frequency profile per sample
        magnitude_avg = magnitude.mean(dim=2)  # [batch, seq_len] - average across hidden dims
        
        # 1. High-frequency energy ratio: noise typically has more energy in high frequencies
        # Split frequencies into low (first half) and high (second half)
        mid_point = seq_len // 2
        low_freq_energy = magnitude_avg[:, :mid_point].sum(dim=1)  # [batch]
        high_freq_energy = magnitude_avg[:, mid_point:].sum(dim=1)  # [batch]
        # Avoid division by zero, add small epsilon
        total_energy = low_freq_energy + high_freq_energy + 1e-8
        high_freq_ratio = high_freq_energy / total_energy  # [batch] - higher = more noise
        
        # 2. Spectral flatness: measures how "noisy" vs "tonal" the spectrum is
        # Flatness = geometric_mean / arithmetic_mean (in log domain for numerical stability)
        # Noise has high flatness (flat spectrum), signal has low flatness (peaky spectrum)
        log_mag = torch.log(magnitude_avg + 1e-8)  # [batch, seq_len]
        geometric_mean = log_mag.mean(dim=1)  # [batch] - mean of logs = log of geometric mean
        arithmetic_mean = magnitude_avg.mean(dim=1)  # [batch]
        log_arithmetic_mean = torch.log(arithmetic_mean + 1e-8)  # [batch]
        # Spectral flatness = exp(geometric_mean - log_arithmetic_mean)
        # Higher flatness (closer to 1) = more noise-like
        spectral_flatness = torch.exp(geometric_mean - log_arithmetic_mean)  # [batch]
        
        # 3. Low-frequency concentration: how much energy is in the lowest 25% of frequencies
        # Signal typically concentrates in low frequencies, noise is more spread out
        low_quarter = max(1, seq_len // 4)
        low_quarter_energy = magnitude_avg[:, :low_quarter].sum(dim=1)  # [batch]
        low_freq_concentration = low_quarter_energy / (total_energy + 1e-8)  # [batch] - higher = more signal-like
        
        seq_len_norm = seq_len / 512.0  # Normalize by typical max length
        
        # Use batch-averaged statistics for shared cutoff (more stable than per-sample)
        batch_high_freq_ratio = high_freq_ratio.mean()
        batch_spectral_flatness = spectral_flatness.mean()
        batch_low_freq_concentration = low_freq_concentration.mean()
        
        stats = torch.stack([
            batch_high_freq_ratio,
            batch_spectral_flatness,
            batch_low_freq_concentration,
            torch.tensor(seq_len_norm, device=hidden_states.device, dtype=hidden_states.dtype)
        ]).unsqueeze(0)  # [1, 4]
        
        # Predict single cutoff ratio for entire batch
        # Network already has sigmoid, so don't apply it again
        cutoff_ratio = self.fft_adaptive_network(stats).squeeze()  # scalar
        
        # Create harder mask with much smaller temperature for sharper cutoff
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        tau = hidden_states.new_tensor(0.01)  # Much smaller temperature for harder mask (was 0.1)
        
        # Create index tensor: [0, 1, 2, ..., seq_len-1]
        indices = torch.arange(seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
        indices = indices.unsqueeze(0).expand(batch_size, -1)  # [batch, seq_len]
        
        # Compute cutoff threshold (shared across batch)
        cutoff_threshold = cutoff_ratio * seq_len  # scalar
        
        # Harder mask for positive frequencies: sigmoid((threshold - index) / tau)
        # With smaller tau, this approaches a hard cutoff while remaining differentiable
        mask_pos = torch.sigmoid((cutoff_threshold - indices) / tau)  # [batch, seq_len]
        
        # Also handle negative frequencies (symmetric in FFT)
        neg_indices = seq_len - indices - 1  # Reverse indices
        mask_neg = torch.sigmoid((cutoff_threshold - neg_indices) / tau)  # [batch, seq_len]
        
        # Combine both masks: keep if either condition is true
        mask = torch.maximum(mask_pos, mask_neg)  # [batch, seq_len]
        
        # Removed epsilon - it was interfering with learning
        # The sigmoid mask already prevents complete filtering naturally
        
        mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        fft_filtered = fft_result * mask
        
        filtered_states = torch.fft.ifft(fft_filtered, dim=1)
        return filtered_states.real, cutoff_ratio
    
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
                seq_lengths = attention_mask.long().sum(dim=1) - 1
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
        regularization_loss = None
        fft_cutoff_ratio = None
        
        if self.use_fft:
            if self.fft_adaptive:
                # Use learnable adaptive filtering
                hidden_states, cutoff_ratio = self.apply_adaptive_fft_filter_learnable(hidden_states)
                
                # Light regularization to keep cutoff near 0.5 (target value)
                # Reduced weight (0.001) to allow more adaptation while maintaining stability
                regularization_loss = 0.001 * (cutoff_ratio - 0.5) ** 2
                fft_cutoff_ratio = cutoff_ratio.item() if isinstance(cutoff_ratio, torch.Tensor) else cutoff_ratio
            else:
                # Use fixed 50% cutoff filtering
                hidden_states, cutoff_ratio = self.apply_fft_filter(hidden_states)
                fft_cutoff_ratio = cutoff_ratio
        
        # Custom head style: pool first, then apply linear layer
        pooled = self.pool_hidden_states(hidden_states, attention_mask)
        logits = self.classifier(pooled)  # [batch, num_labels]
        
        # Softmax identical to default head: torch.nn.functional.softmax(logits, dim=-1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            # Add regularization if adaptive filtering is used
            if regularization_loss is not None:
                loss = loss + regularization_loss
        
        output_dict = {
            'logits': logits,
            'probs': probs,
            'loss': loss
        }
        
        # Add FFT cutoff ratio if available
        if fft_cutoff_ratio is not None:
            output_dict['fft_cutoff_ratio'] = fft_cutoff_ratio
        
        return output_dict
