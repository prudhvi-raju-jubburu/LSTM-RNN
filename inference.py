"""
Inference pipeline for AI Text-to-Speech Voice Generation System (LSTM + Attention).
Converts raw text into natural-sounding speech audio waveforms.
"""
import os
import argparse
from typing import Tuple, Optional, Dict
import numpy as np
import torch

from config import Config, default_config
from text.cleaner import clean_text
from text.tokenizer import text_to_tensor
from audio.audio_processing import AudioProcessor
from audio.visualizer import plot_waveform_and_spectrogram, plot_attention_alignment
from models.tts_lstm import TextToSpeechLSTM


class TTSSynthesizer:
    """
    High-level Text-to-Speech synthesis interface.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Config = default_config,
        device: Optional[str] = None,
    ):
        self.config = config
        self.device = device or config.train.device
        self.audio_processor = AudioProcessor(config.audio)

        # Initialize model
        self.model = TextToSpeechLSTM(config).to(self.device)
        self.model.eval()

        # Load checkpoint if available
        self.checkpoint_info = None
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Loads model weights from checkpoint file."""
        if checkpoint_path is None:
            # Check default locations
            default_best = os.path.join(self.config.train.checkpoint_dir, "best_model.pt")
            default_latest = os.path.join(self.config.train.checkpoint_dir, "checkpoint_latest.pt")
            if os.path.exists(default_best):
                checkpoint_path = default_best
            elif os.path.exists(default_latest):
                checkpoint_path = default_latest

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[*] Loading model checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.checkpoint_info = {
                    "epoch": checkpoint.get("epoch", "?"),
                    "val_loss": checkpoint.get("val_loss", None),
                }
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print("[!] Notice: No checkpoint found or provided. Running with initialized neural weights.")

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        griffin_lim_iters: int = 50,
        max_steps: int = 800,
        stop_threshold: float = 0.5,
    ) -> Dict[str, object]:
        """
        Synthesize speech from input text string.

        Returns:
            Dictionary containing:
                - 'wav': np.ndarray (1D audio waveform)
                - 'sample_rate': int
                - 'mel': np.ndarray (80, frames)
                - 'alignment': np.ndarray (frames, text_len)
                - 'cleaned_text': str
        """
        cleaned = clean_text(text)
        if not cleaned:
            cleaned = "silence"

        # Tokenize text
        text_tensor = text_to_tensor(cleaned, add_eos=True, device=self.device).unsqueeze(0)  # [1, L]
        text_lengths = torch.tensor([text_tensor.size(1)], dtype=torch.long, device=self.device)

        # Autoregressive generation
        out = self.model.generate(
            text_seq=text_tensor,
            text_lengths=text_lengths,
            max_steps=max_steps,
            stop_threshold=stop_threshold,
        )

        mel_postnet = out["mel_postnet"].squeeze(0).transpose(0, 1).cpu()  # [n_mels, num_frames]
        alignment = out["alignments"].squeeze(0).cpu().numpy()            # [num_frames, text_len]

        # Waveform vocoder synthesis via Griffin-Lim
        raw_wav = self.audio_processor.mel_to_wav(mel_postnet, iters=griffin_lim_iters)

        # Speed and Pitch post-processing
        final_wav = self.audio_processor.adjust_pitch_speed(
            raw_wav, speed_factor=speed, pitch_semitones=pitch_shift
        )

        return {
            "wav": final_wav,
            "sample_rate": self.config.audio.sample_rate,
            "mel": mel_postnet.numpy(),
            "alignment": alignment,
            "cleaned_text": cleaned,
        }

    def save_wav(self, wav: np.ndarray, output_path: str):
        """Save synthesized waveform to WAV file."""
        self.audio_processor.save_audio(wav, output_path)
        print(f"[*] Audio saved successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize Speech from Text")
    parser.add_argument("--text", type=str, default="Artificial intelligence text to speech synthesis using LSTM.", help="Input text to speak")
    parser.add_argument("--output", type=str, default="output_speech.wav", help="Output .wav path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed factor (e.g. 1.0, 1.2)")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch shift in semitones (e.g. 0.0, 2.0, -2.0)")
    parser.add_argument("--iters", type=int, default=50, help="Griffin-Lim iterations")
    args = parser.parse_args()

    synthesizer = TTSSynthesizer(checkpoint_path=args.checkpoint)
    result = synthesizer.synthesize(
        text=args.text,
        speed=args.speed,
        pitch_shift=args.pitch,
        griffin_lim_iters=args.iters,
    )
    synthesizer.save_wav(result["wav"], args.output)
