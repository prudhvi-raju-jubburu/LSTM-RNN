"""
Location-Aware Additive Attention Mechanism for TTS alignment.
Aligns encoder linguistic representations with temporal decoder acoustic frames.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
from config import ModelConfig, default_config


class LocationLayer(nn.Module):
    """
    Extracts positional alignment features from previous attention weights using 1D convolution.
    """
    def __init__(self, attention_n_filters: int, attention_kernel_size: int, attention_dim: int):
        super().__init__()
        padding = (attention_kernel_size - 1) // 2
        self.location_conv = nn.Conv1d(
            2,  # Cumulative attention + Previous attention
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
        )
        self.location_dense = nn.Linear(attention_n_filters, attention_dim, bias=False)

    def forward(self, attention_weights_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_weights_cat: [batch_size, 2, max_text_len]
        Returns:
            location_features: [batch_size, max_text_len, attention_dim]
        """
        # [batch, 2, max_text_len] -> [batch, attention_n_filters, max_text_len]
        processed = self.location_conv(attention_weights_cat)
        # [batch, max_text_len, attention_n_filters] -> [batch, max_text_len, attention_dim]
        processed = processed.transpose(1, 2)
        return self.location_dense(processed)


class LocationAwareAttention(nn.Module):
    """
    Additive Attention with Location Awareness:
    score(s_i, h_j) = v^T tanh(W s_i + V h_j + U f_{i, j} + b)
    """
    def __init__(self, cfg: ModelConfig = default_config.model):
        super().__init__()
        encoder_dim = cfg.encoder_lstm_dim * 2  # 512
        decoder_dim = cfg.decoder_lstm_dim      # 512
        attn_dim = cfg.attention_dim            # 128

        self.query_layer = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            cfg.attention_location_n_filters,
            cfg.attention_location_kernel_size,
            attn_dim,
        )
        self.score_mask_value = -1e4

    def get_initial_weights(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns initial zero attention weights and cumulative weights: shape [batch_size, max_text_len]."""
        batch_size, max_text_len, _ = memory.shape
        attn_weights = torch.zeros(batch_size, max_text_len, device=memory.device)
        attn_weights[:, 0] = 1.0  # initialize focus at the beginning of the text
        cumulative = attn_weights.clone()
        return attn_weights, cumulative

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        prev_weights: torch.Tensor,
        cumulative_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Current decoder state [batch_size, decoder_dim]
            memory: Raw encoder outputs [batch_size, max_text_len, encoder_dim]
            processed_memory: Pre-computed linear projection of memory [batch_size, max_text_len, attn_dim]
            prev_weights: Previous step attention weights [batch_size, max_text_len]
            cumulative_weights: Cumulative attention weights [batch_size, max_text_len]
            mask: Optional boolean mask for padded tokens [batch_size, max_text_len]
        Returns:
            context_vector: [batch_size, encoder_dim]
            weights: [batch_size, max_text_len]
            cumulative_weights: [batch_size, max_text_len]
        """
        # 1. Location features from previous and cumulative alignment
        weights_cat = torch.stack([prev_weights, cumulative_weights], dim=1)  # [batch, 2, max_text_len]
        location_features = self.location_layer(weights_cat)  # [batch, max_text_len, attn_dim]

        # 2. Query projection
        # [batch, decoder_dim] -> [batch, 1, attn_dim]
        processed_query = self.query_layer(query).unsqueeze(1)

        # 3. Additive energy computation
        # [batch, max_text_len, attn_dim]
        energy = self.v(torch.tanh(processed_query + processed_memory + location_features)).squeeze(-1)

        # 4. Mask padded tokens
        if mask is not None:
            energy.masked_fill_(mask, self.score_mask_value)

        # 5. Softmax to obtain alignment distribution
        weights = torch.softmax(energy, dim=-1)

        # 6. Cumulative update
        new_cumulative_weights = cumulative_weights + weights

        # 7. Compute context vector as weighted sum of encoder memory
        # [batch, 1, max_text_len] x [batch, max_text_len, encoder_dim] -> [batch, encoder_dim]
        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)

        return context, weights, new_cumulative_weights
