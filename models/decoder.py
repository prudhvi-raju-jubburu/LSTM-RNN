"""
Autoregressive LSTM Decoder with PreNet, Attention context, and Stop Token prediction.
"""
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from models.attention import LocationAwareAttention
from config import ModelConfig, AudioConfig, default_config


class PreNet(nn.Module):
    """
    Two-layer bottleneck network with active Dropout (0.5) even during inference,
    forcing the model to generate coherent speech rather than simply copying the previous frame.
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=self.dropout, training=True)
        x = self.relu(self.fc2(x))
        x = nn.functional.dropout(x, p=self.dropout, training=True)
        return x


class Decoder(nn.Module):
    """
    Autoregressive Decoder:
    - PreNet processes acoustic frame [batch_size, n_mels]
    - LocationAwareAttention aligns encoder memory
    - 2-layer Uni-directional LSTM maintains autoregressive acoustic state
    - Linear projection outputs next Mel frame
    - Stop Token projection outputs end-of-speech probability
    """
    def __init__(
        self,
        model_cfg: ModelConfig = default_config.model,
        audio_cfg: AudioConfig = default_config.audio,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.audio_cfg = audio_cfg
        self.n_mels = audio_cfg.n_mels
        self.encoder_dim = model_cfg.encoder_lstm_dim * 2  # 512
        self.decoder_dim = model_cfg.decoder_lstm_dim      # 512

        # 1. PreNet
        self.prenet = PreNet(self.n_mels, model_cfg.prenet_dim, model_cfg.prenet_dropout)

        # 2. Attention
        self.attention = LocationAwareAttention(model_cfg)

        # 3. Two-layer Uni-directional LSTM
        # LSTM 1 receives [prenet_dim + encoder_dim]
        self.lstm1 = nn.LSTMCell(model_cfg.prenet_dim + self.encoder_dim, self.decoder_dim)
        # LSTM 2 receives [decoder_dim + encoder_dim]
        self.lstm2 = nn.LSTMCell(self.decoder_dim + self.encoder_dim, self.decoder_dim)

        # 4. Projections
        # Mel projection from concatenated [lstm2_output, context]
        self.linear_mel = nn.Linear(self.decoder_dim + self.encoder_dim, self.n_mels)
        # Stop token prediction
        self.linear_stop = nn.Linear(self.decoder_dim + self.encoder_dim, 1)

    def initialize_decoder_states(
        self, memory: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Initializes zero hidden/cell states, attention weights, context vector, and initial frame.
        """
        batch_size = memory.size(0)
        device = memory.device

        h1 = torch.zeros(batch_size, self.decoder_dim, device=device)
        c1 = torch.zeros(batch_size, self.decoder_dim, device=device)
        h2 = torch.zeros(batch_size, self.decoder_dim, device=device)
        c2 = torch.zeros(batch_size, self.decoder_dim, device=device)

        attn_weights, cumulative_weights = self.attention.get_initial_weights(memory)
        context = torch.zeros(batch_size, self.encoder_dim, device=device)
        initial_frame = torch.zeros(batch_size, self.n_mels, device=device)

        return (h1, c1), (h2, c2), attn_weights, cumulative_weights, context, initial_frame

    def decode_step(
        self,
        current_frame: torch.Tensor,
        state1: Tuple[torch.Tensor, torch.Tensor],
        state2: Tuple[torch.Tensor, torch.Tensor],
        prev_weights: torch.Tensor,
        cumulative_weights: torch.Tensor,
        context: torch.Tensor,
        memory: torch.Tensor,
        processed_memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Perform a single decoder time step.
        """
        # Pass acoustic frame through PreNet
        prenet_out = self.prenet(current_frame)  # [batch, prenet_dim]

        # LSTM 1
        lstm1_in = torch.cat([prenet_out, context], dim=-1)
        h1, c1 = self.lstm1(lstm1_in, state1)

        # Attention step using query h1
        context, attn_weights, new_cumulative = self.attention(
            query=h1,
            memory=memory,
            processed_memory=processed_memory,
            prev_weights=prev_weights,
            cumulative_weights=cumulative_weights,
            mask=mask,
        )

        # LSTM 2
        lstm2_in = torch.cat([h1, context], dim=-1)
        h2, c2 = self.lstm2(lstm2_in, state2)

        # Output projection
        out_cat = torch.cat([h2, context], dim=-1)
        mel_frame = self.linear_mel(out_cat)
        stop_token = torch.sigmoid(self.linear_stop(out_cat)).squeeze(-1)

        return (
            mel_frame,
            stop_token,
            (h1, c1),
            (h2, c2),
            attn_weights,
            new_cumulative,
            context,
        )
