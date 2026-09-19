"""
Text Encoder: Character Embedding + 1D Convolutions + Bidirectional LSTM.
"""
from typing import Tuple
import torch
import torch.nn as nn
from text.symbols import NUM_SYMBOLS
from config import ModelConfig, default_config


class Encoder(nn.Module):
    """
    Encodes character sequence into continuous linguistic representations:
    1. Character embedding lookup
    2. 3x 1D Convolutions (kernel size 5) with BatchNorm and ReLU
    3. Bidirectional LSTM
    """
    def __init__(self, cfg: ModelConfig = default_config.model, num_symbols: int = NUM_SYMBOLS):
        super().__init__()
        self.cfg = cfg

        # 1. Embedding layer
        self.embedding = nn.Embedding(num_symbols, cfg.embedding_dim)

        # 2. Convolutional blocks
        conv_layers = []
        in_channels = cfg.embedding_dim
        for _ in range(cfg.encoder_conv_layers):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        cfg.encoder_conv_channels,
                        kernel_size=cfg.encoder_conv_kernel,
                        stride=1,
                        padding=(cfg.encoder_conv_kernel - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(cfg.encoder_conv_channels),
                    nn.ReLU(),
                    nn.Dropout(cfg.encoder_dropout),
                )
            )
            in_channels = cfg.encoder_conv_channels
        self.convolutions = nn.ModuleList(conv_layers)

        # 3. Bidirectional LSTM
        # Input dim: encoder_conv_channels, Hidden dim: encoder_lstm_dim
        # Output dim will be 2 * encoder_lstm_dim (e.g. 256 * 2 = 512)
        self.lstm = nn.LSTM(
            cfg.encoder_conv_channels,
            cfg.encoder_lstm_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, text_seq: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_seq: LongTensor of shape [batch_size, max_text_len]
            text_lengths: LongTensor of shape [batch_size] containing sequence lengths
        Returns:
            encoder_outputs: FloatTensor of shape [batch_size, max_text_len, encoder_lstm_dim * 2]
        """
        # [batch_size, max_text_len] -> [batch_size, max_text_len, embedding_dim]
        embedded = self.embedding(text_seq)

        # Conv1D expects [batch_size, channels, length]
        out = embedded.transpose(1, 2)
        for conv in self.convolutions:
            out = conv(out)

        # Transpose back to [batch_size, length, channels]
        out = out.transpose(1, 2)

        # Pack padded sequence for dynamic LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            out, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=text_seq.size(1)
        )

        return encoder_outputs
