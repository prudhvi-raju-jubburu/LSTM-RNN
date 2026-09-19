"""
Audio processing pipeline: Mel-spectrogram extraction, Griffin-Lim phase reconstruction,
audio normalization, and file I/O.
"""
import math
from typing import Optional, Tuple
import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample

from config import AudioConfig, default_config


def hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to Mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """Convert Mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    n_mels: int = 80,
    n_fft: int = 1024,
    sample_rate: int = 22050,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> torch.Tensor:
    """
    Construct triangular Mel filterbank matrix of shape [n_mels, n_fft // 2 + 1].
    """
    if fmax is None:
        fmax = float(sample_rate) / 2.0

    num_bins = n_fft // 2 + 1
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    weights = np.zeros((n_mels, num_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus and k < num_bins:
                weights[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)

        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m and k < num_bins:
                weights[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    # Normalize triangular area
    enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
    weights *= enorm[:, np.newaxis]

    return torch.from_numpy(weights).float()


class AudioProcessor:
    """
    Unified Audio Processing Engine for TTS:
    - Mel-spectrogram computation
    - Griffin-Lim Mel-to-waveform vocoder
    - Waveform I/O & normalization
    """
    def __init__(self, cfg: Optional[AudioConfig] = None):
        self.cfg = cfg or default_config.audio
        self.mel_basis = create_mel_filterbank(
            n_mels=self.cfg.n_mels,
            n_fft=self.cfg.n_fft,
            sample_rate=self.cfg.sample_rate,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
        )
        # Pseudoinverse for Griffin-Lim mel-to-linear conversion
        inv_mel = np.linalg.pinv(self.mel_basis.numpy())
        self.inv_mel_basis = torch.from_numpy(inv_mel).float()

    def load_audio(self, path: str) -> np.ndarray:
        """Load audio file, resample to target sample rate, and return mono float array in [-1, 1]."""
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)  # Convert stereo to mono

        if sr != self.cfg.sample_rate:
            target_len = int(len(wav) * self.cfg.sample_rate / sr)
            wav = resample(wav, target_len)

        # Normalize amplitude peak
        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / peak * 0.95

        return wav.astype(np.float32)

    def save_audio(self, wav: np.ndarray, path: str):
        """Save float waveform numpy array to wav file."""
        wav = np.clip(wav, -1.0, 1.0)
        sf.write(path, wav, self.cfg.sample_rate, subtype="PCM_16")

    def wav_to_mel(self, wav: np.ndarray) -> torch.Tensor:
        """
        Convert 1D audio waveform into 80-channel log-Mel Spectrogram.
        Returns: Tensor of shape [n_mels, num_frames]
        """
        wav_t = torch.from_numpy(wav).unsqueeze(0).float()  # [1, T]
        window = torch.hann_window(self.cfg.win_length)

        stft_res = torch.stft(
            wav_t,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
            return_complex=True,
        )
        magnitudes = torch.abs(stft_res).squeeze(0)  # [n_fft//2 + 1, num_frames]

        mel_spec = torch.matmul(self.mel_basis, magnitudes)  # [n_mels, num_frames]
        mel_spec = torch.clamp(mel_spec, min=1e-5)
        log_mel = torch.log(mel_spec)

        # Dynamic range normalization
        log_mel = (log_mel + 5.0) / 5.0  # roughly normalizes to [0, 1] range
        return log_mel

    def mel_to_wav(self, mel: torch.Tensor, iters: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct audio waveform from log-Mel Spectrogram via Griffin-Lim algorithm.
        Input mel shape: [n_mels, num_frames]
        """
        iters = iters or self.cfg.griffin_lim_iters
        if mel.dim() == 3:
            mel = mel.squeeze(0)

        # Denormalize
        log_mel = mel * 5.0 - 5.0
        mel_linear = torch.exp(log_mel)

        # Project Mel back to Linear magnitude spectrogram
        linear_mag = torch.matmul(self.inv_mel_basis, mel_linear)
        linear_mag = torch.clamp(linear_mag, min=1e-6)

        num_freq_bins, num_frames = linear_mag.shape
        window = torch.hann_window(self.cfg.win_length)

        # Initialize random phase
        angles = torch.empty_like(linear_mag).uniform_(-math.pi, math.pi)
        complex_spec = torch.polar(linear_mag, angles)

        # Griffin-Lim iterative phase recovery
        for _ in range(iters):
            wav = torch.istft(
                complex_spec.unsqueeze(0),
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=window,
            )
            # Re-compute STFT
            re_stft = torch.stft(
                wav,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                window=window,
                return_complex=True,
            ).squeeze(0)

            # Keep target magnitude, update phase
            angles = torch.angle(re_stft)
            # Ensure frequency dimension match
            if angles.shape[0] != linear_mag.shape[0] or angles.shape[1] != linear_mag.shape[1]:
                angles = angles[:linear_mag.shape[0], :linear_mag.shape[1]]
            complex_spec = torch.polar(linear_mag, angles)

        # Final inverse STFT to audio waveform
        final_wav = torch.istft(
            complex_spec.unsqueeze(0),
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length,
            window=window,
        ).squeeze(0).numpy()

        # Peak normalization
        peak = np.max(np.abs(final_wav))
        if peak > 0:
            final_wav = final_wav / peak * 0.95

        return final_wav.astype(np.float32)

    def adjust_pitch_speed(self, wav: np.ndarray, speed_factor: float = 1.0, pitch_semitones: float = 0.0) -> np.ndarray:
        """
        Adjust playback speed and pitch of audio waveform.
        - speed_factor: > 1.0 is faster, < 1.0 is slower.
        - pitch_semitones: > 0 is higher pitch, < 0 is lower pitch.
        """
        out_wav = wav.copy()

        # Speed modification via resampled length
        if speed_factor != 1.0 and speed_factor > 0.1:
            new_length = int(len(out_wav) / speed_factor)
            out_wav = resample(out_wav, new_length)

        # Pitch modification via sampling interpolation
        if pitch_semitones != 0.0:
            factor = 2.0 ** (pitch_semitones / 12.0)
            resampled_len = int(len(out_wav) / factor)
            temp = resample(out_wav, resampled_len)
            out_wav = resample(temp, len(out_wav))

        # Peak normalize
        peak = np.max(np.abs(out_wav))
        if peak > 0:
            out_wav = out_wav / peak * 0.95

        return out_wav.astype(np.float32)
