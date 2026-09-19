"""
PostNet: 5-layer 1D Convolutional Residual Network for Mel-Spectrogram Refinement.
"""
import torch
import torch.nn as nn
from config import ModelConfig, AudioConfig, default_config


class PostNet(nn.Module):
    """
    Predicts fine residual corrections to the initial decoder Mel-spectrogram output:
    Mel_final = Mel_initial + PostNet(Mel_initial)
    """
    def __init__(
        self,
        model_cfg: ModelConfig = default_config.model,
        audio_cfg: AudioConfig = default_config.audio,
    ):
        super().__init__()
        self.n_mels = audio_cfg.n_mels
        channels = model_cfg.postnet_conv_channels
        kernel_size = model_cfg.postnet_conv_kernel
        padding = (kernel_size - 1) // 2

        layers = []
        # First layer
        layers.append(
            nn.Sequential(
                nn.Conv1d(self.n_mels, channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(model_cfg.postnet_dropout),
            )
        )

        # Intermediate layers (3 layers)
        for _ in range(model_cfg.postnet_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.Tanh(),
                    nn.Dropout(model_cfg.postnet_dropout),
                )
            )

        # Final layer (linear projection back to n_mels)
        layers.append(
            nn.Sequential(
                nn.Conv1d(channels, self.n_mels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(self.n_mels),
                nn.Dropout(model_cfg.postnet_dropout),
            )
        )

        self.convolutions = nn.ModuleList(layers)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: [batch_size, n_mels, num_frames] or [batch_size, num_frames, n_mels]
        Returns:
            residual: [batch_size, n_mels, num_frames]
        """
        # Ensure shape is [batch_size, n_mels, num_frames]
        transpose_needed = False
        if mel_spec.shape[1] != self.n_mels and mel_spec.shape[2] == self.n_mels:
            mel_spec = mel_spec.transpose(1, 2)
            transpose_needed = True

        out = mel_spec
        for conv in self.convolutions:
            out = conv(out)

        if transpose_needed:
            out = out.transpose(1, 2)

        return out
