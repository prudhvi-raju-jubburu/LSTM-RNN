"""
End-to-end AI Text-to-Speech Voice Generation Model (LSTM + Attention).
Unites Encoder, Attention, Decoder, and PostNet into a unified PyTorch Module.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.postnet import PostNet
from config import Config, default_config


class TextToSpeechLSTM(nn.Module):
    """
    End-to-End LSTM TTS Model:
    Text -> Character Embeddings -> Conv1D + BiLSTM -> Location-Aware Attention ->
    Autoregressive Decoder LSTM -> Mel Projection -> PostNet Refinement -> Waveform Vocoder.
    """
    def __init__(self, config: Config = default_config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.model)
        self.decoder = Decoder(config.model, config.audio)
        self.postnet = PostNet(config.model, config.audio)

    def forward(
        self,
        text_seq: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_targets: Optional[torch.Tensor] = None,
        max_decoder_steps: Optional[int] = None,
        stop_threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting both Teacher Forcing (when mel_targets is provided)
        and Autoregressive generation (when mel_targets is None).

        Args:
            text_seq: [batch_size, max_text_len]
            text_lengths: [batch_size]
            mel_targets: [batch_size, num_frames, n_mels] (optional for training)
            max_decoder_steps: maximum time steps for inference
            stop_threshold: stop-token threshold for early termination during inference

        Returns:
            Dictionary containing:
                - 'mel_initial': [batch_size, num_frames, n_mels]
                - 'mel_postnet': [batch_size, num_frames, n_mels]
                - 'stop_tokens': [batch_size, num_frames]
                - 'alignments': [batch_size, num_frames, max_text_len]
        """
        batch_size, max_text_len = text_seq.shape
        device = text_seq.device

        # 1. Encode text sequence
        # encoder_outputs: [batch_size, max_text_len, encoder_dim]
        memory = self.encoder(text_seq, text_lengths)

        # Pre-compute memory projections for attention
        processed_memory = self.decoder.attention.memory_layer(memory)

        # Create padding mask for encoder memory (True where padded)
        # [batch_size, max_text_len]
        mask = torch.arange(max_text_len, device=device).unsqueeze(0) >= text_lengths.unsqueeze(1)

        # 2. Initialize decoder states
        state1, state2, attn_weights, cumulative_weights, context, current_frame = (
            self.decoder.initialize_decoder_states(memory)
        )

        mel_outputs = []
        stop_outputs = []
        alignments = []

        is_training = mel_targets is not None
        steps = mel_targets.size(1) if is_training else (max_decoder_steps or self.config.model.max_decoder_steps)
        threshold = stop_threshold or self.config.model.stop_threshold

        # 3. Autoregressive / Teacher-Forced Decoding Loop
        for t in range(steps):
            (
                mel_frame,
                stop_token,
                state1,
                state2,
                attn_weights,
                cumulative_weights,
                context,
            ) = self.decoder.decode_step(
                current_frame=current_frame,
                state1=state1,
                state2=state2,
                prev_weights=attn_weights,
                cumulative_weights=cumulative_weights,
                context=context,
                memory=memory,
                processed_memory=processed_memory,
                mask=mask,
            )

            mel_outputs.append(mel_frame)
            stop_outputs.append(stop_token)
            alignments.append(attn_weights)

            # Next input frame: teacher forcing or autoregressive feedback
            if is_training:
                current_frame = mel_targets[:, t, :]
            else:
                current_frame = mel_frame
                # Check for stop condition during single-item inference
                if batch_size == 1 and stop_token.item() > threshold and t > 10:
                    break

        # Stack outputs
        # [batch_size, num_frames, n_mels]
        mel_initial = torch.stack(mel_outputs, dim=1)
        # [batch_size, num_frames]
        stop_tokens = torch.stack(stop_outputs, dim=1)
        # [batch_size, num_frames, max_text_len]
        alignment_matrix = torch.stack(alignments, dim=1)

        # 4. PostNet refinement
        # mel_initial shape: [batch, frames, n_mels]
        # PostNet input expects [batch, n_mels, frames]
        mel_residual = self.postnet(mel_initial.transpose(1, 2)).transpose(1, 2)
        mel_postnet = mel_initial + mel_residual

        return {
            "mel_initial": mel_initial,
            "mel_postnet": mel_postnet,
            "stop_tokens": stop_tokens,
            "alignments": alignment_matrix,
        }

    @torch.no_grad()
    def generate(
        self,
        text_seq: torch.Tensor,
        text_lengths: torch.Tensor,
        max_steps: int = 800,
        stop_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience inference method running in evaluation mode.
        """
        self.eval()
        return self.forward(
            text_seq=text_seq,
            text_lengths=text_lengths,
            mel_targets=None,
            max_decoder_steps=max_steps,
            stop_threshold=stop_threshold,
        )
