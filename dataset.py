"""
LJSpeech Dataset loader, processor, and dynamic collator.
Supports both the full LJ Speech dataset and an auto-generated sample dataset for rapid prototyping.
"""
import os
import glob
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from text.tokenizer import text_to_sequence
from text.symbols import PAD_ID
from audio.audio_processing import AudioProcessor
from config import Config, default_config


class LJSpeechDataset(Dataset):
    """
    PyTorch Dataset for LJ Speech:
    Reads metadata.csv (ID|raw_text|normalized_text) and loads audio wavs into Mel-spectrograms.
    """
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        config: Config = default_config,
        max_samples: Optional[int] = None,
        cache_mels: bool = True,
    ):
        self.config = config
        self.audio_processor = AudioProcessor(config.audio)
        self.cache_mels = cache_mels
        self.mel_cache: Dict[str, torch.Tensor] = {}

        # Discover or locate dataset path
        self.dataset_path = self._find_dataset_path(dataset_path)
        self.items: List[Tuple[str, str, str]] = self._load_metadata()

        if max_samples is not None and len(self.items) > max_samples:
            self.items = self.items[:max_samples]

    def _find_dataset_path(self, user_path: Optional[str]) -> str:
        """Find either user-provided path, Kagglehub cache, or create sample dataset."""
        candidates = []
        if user_path:
            candidates.append(user_path)

        # Check standard kagglehub cache locations
        kaggle_base = os.path.expanduser("~/.cache/kagglehub/datasets/mathurinache/the-lj-speech-dataset")
        if os.path.exists(kaggle_base):
            for root, dirs, files in os.walk(kaggle_base):
                if "metadata.csv" in files:
                    candidates.append(root)

        # Check local data folder
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "LJSpeech-1.1")
        candidates.append(local_dir)
        local_sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_ljspeech")
        candidates.append(local_sample)

        for path in candidates:
            if os.path.exists(path) and os.path.isfile(os.path.join(path, "metadata.csv")):
                return path

        # If none found, generate sample dataset
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_ljspeech")
        self.create_sample_dataset(sample_dir)
        return sample_dir

    def _load_metadata(self) -> List[Tuple[str, str, str]]:
        """Parse metadata.csv file."""
        metadata_file = os.path.join(self.dataset_path, "metadata.csv")
        items = []

        if not os.path.exists(metadata_file):
            return items

        with open(metadata_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    file_id = parts[0]
                    raw_text = parts[1]
                    norm_text = parts[2] if len(parts) >= 3 else raw_text

                    # Verify wav exists
                    wav_candidates = [
                        os.path.join(self.dataset_path, "wavs", f"{file_id}.wav"),
                        os.path.join(self.dataset_path, f"{file_id}.wav"),
                    ]
                    for w in wav_candidates:
                        if os.path.exists(w):
                            items.append((file_id, norm_text, w))
                            break

        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_id, text, wav_path = self.items[index]

        # 1. Text to sequence tensor
        text_seq = text_to_sequence(text, add_eos=True)
        text_tensor = torch.tensor(text_seq, dtype=torch.long)

        # 2. Audio to mel spectrogram
        if self.cache_mels and file_id in self.mel_cache:
            mel = self.mel_cache[file_id]
        else:
            wav = self.audio_processor.load_audio(wav_path)
            mel = self.audio_processor.wav_to_mel(wav)  # [n_mels, num_frames]
            # Transpose to [num_frames, n_mels] for decoder
            mel = mel.transpose(0, 1)
            if self.cache_mels:
                self.mel_cache[file_id] = mel

        return {
            "file_id": file_id,
            "text": text_tensor,
            "mel": mel,
            "text_len": len(text_tensor),
            "mel_len": mel.size(0),
        }

    @staticmethod
    def create_sample_dataset(output_dir: str, num_samples: int = 12):
        """
        Generates sample LJSpeech-formatted dataset with multi-formant audio signals
        for instant development and training verification.
        """
        os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
        ap = AudioProcessor(default_config.audio)
        sample_texts = [
            "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts.",
            "The author of the invention is not certainly known.",
            "The inventor was probably John Gutenberg of Mainz.",
            "He was born about the year fourteen hundred.",
            "In fourteen fifty he entered into a partnership with John Fust.",
            "Artificial intelligence speech synthesis provides natural voice interactions.",
            "Voice assistants help millions of people with daily accessibility tasks.",
            "Deep learning with long short term memory models captures sequence dynamics.",
            "The text to speech system converts written words into expressive acoustic waveforms.",
            "Natural sounding audio improves user comprehension and digital accessibility.",
            "Modern speech models utilize attention mechanisms to align phonetic units.",
            "Thank you for listening to this neural text to speech demonstration.",
        ]

        metadata_lines = []
        for i, text in enumerate(sample_texts[:num_samples]):
            file_id = f"LJ001-{i+1:04d}"
            wav_path = os.path.join(output_dir, "wavs", f"{file_id}.wav")

            # Generate synthetic speech-like formant audio
            duration = max(1.5, len(text) * 0.055)
            t = np.linspace(0, duration, int(default_config.audio.sample_rate * duration), endpoint=False)

            # Pitch fundamental (F0 ~ 180 Hz with natural inflection)
            f0 = 180.0 + 25.0 * np.sin(2 * np.pi * 1.5 * t)
            phase = 2 * np.pi * np.cumsum(f0) / default_config.audio.sample_rate

            # Formants simulating vowel resonance
            sig = 0.4 * np.sin(phase) + 0.25 * np.sin(2.5 * phase) + 0.15 * np.sin(4.2 * phase)
            # Amplitude envelope
            env = np.sin(np.pi * t / duration) ** 0.5
            wav = (sig * env * 0.9).astype(np.float32)

            ap.save_audio(wav, wav_path)
            metadata_lines.append(f"{file_id}|{text}|{text}")

        with open(os.path.join(output_dir, "metadata.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines) + "\n")


class TTSCollate:
    """
    Collate function to dynamically pad text and mel frames in a mini-batch.
    """
    def __init__(self, pad_id: int = PAD_ID):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)

        # Extract lengths
        text_lens = [b["text_len"] for b in batch]
        mel_lens = [b["mel_len"] for b in batch]
        max_text_len = max(text_lens)
        max_mel_len = max(mel_lens)
        n_mels = batch[0]["mel"].size(1)

        # Allocate padded tensors
        padded_texts = torch.full((batch_size, max_text_len), self.pad_id, dtype=torch.long)
        padded_mels = torch.zeros((batch_size, max_mel_len, n_mels), dtype=torch.float32)
        stop_targets = torch.zeros((batch_size, max_mel_len), dtype=torch.float32)

        for i, item in enumerate(batch):
            t_len = item["text_len"]
            m_len = item["mel_len"]
            padded_texts[i, :t_len] = item["text"]
            padded_mels[i, :m_len, :] = item["mel"]
            # Stop token is 1.0 from m_len - 1 onwards
            stop_targets[i, m_len - 1 :] = 1.0

        return {
            "text": padded_texts,
            "text_lengths": torch.tensor(text_lens, dtype=torch.long),
            "mel": padded_mels,
            "mel_lengths": torch.tensor(mel_lens, dtype=torch.long),
            "stop_targets": stop_targets,
        }
