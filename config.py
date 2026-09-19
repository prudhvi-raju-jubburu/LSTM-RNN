"""
Central configuration for AI Text-to-Speech Voice Generation System (LSTM + Attention).
"""
import os
from dataclasses import dataclass, field
import torch


@dataclass
class AudioConfig:
    # Sampling rate & FFT properties
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0

    # Audio amplitude dynamic range / normalization
    ref_level_db: float = 20.0
    min_level_db: float = -100.0
    max_abs_value: float = 4.0

    # Griffin-Lim reconstruction iterations
    griffin_lim_iters: int = 60


@dataclass
class ModelConfig:
    # Text embedding
    embedding_dim: int = 256

    # Encoder (Conv1D + BiLSTM)
    encoder_conv_layers: int = 3
    encoder_conv_kernel: int = 5
    encoder_conv_channels: int = 256
    encoder_lstm_dim: int = 256  # Bidirectional -> output is 256 * 2 = 512
    encoder_dropout: float = 0.5

    # Attention (Location-aware)
    attention_dim: int = 128
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31

    # Decoder (PreNet + 2-layer LSTM)
    prenet_dim: int = 256
    prenet_dropout: float = 0.5
    decoder_lstm_dim: int = 512
    decoder_layers: int = 2
    decoder_dropout: float = 0.1
    max_decoder_steps: int = 1000
    stop_threshold: float = 0.5

    # PostNet (5-layer Conv1D residual)
    postnet_layers: int = 5
    postnet_conv_channels: int = 256
    postnet_conv_kernel: int = 5
    postnet_dropout: float = 0.5


@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 1.0
    epochs: int = 50
    save_interval: int = 5
    checkpoint_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


default_config = Config()
