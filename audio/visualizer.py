"""
Audio and Attention visualization utilities using matplotlib.
"""
import io
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_waveform_and_spectrogram(
    wav: np.ndarray,
    mel: np.ndarray,
    sample_rate: int = 22050,
    title: str = "Synthesized Speech",
) -> plt.Figure:
    """
    Generate a two-panel figure showing waveform and mel-spectrogram.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [1, 2]})

    # Waveform plot
    time_axis = np.linspace(0, len(wav) / sample_rate, len(wav))
    ax1.plot(time_axis, wav, color="#2563EB", lw=1.0)
    ax1.set_title(f"{title} - Waveform", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else 1)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Spectrogram plot
    if mel.ndim == 3:
        mel = mel[0]
    im = ax2.imshow(mel, aspect="auto", origin="lower", interpolation="none", cmap="magma")
    ax2.set_title("80-Channel Log-Mel Spectrogram", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Acoustic Frames")
    ax2.set_ylabel("Mel Frequency Channels")
    fig.colorbar(im, ax=ax2, orientation="horizontal", pad=0.2, label="Normalized Magnitude")

    plt.tight_layout()
    return fig


def plot_attention_alignment(
    alignment: np.ndarray,
    text_chars: Optional[list] = None,
    title: str = "Attention Alignment Matrix",
) -> plt.Figure:
    """
    Plot attention weight alignment between encoder text tokens and decoder audio frames.
    Shape of alignment: [decoder_frames, encoder_steps] or [encoder_steps, decoder_frames]
    """
    if alignment.ndim == 3:
        alignment = alignment[0]

    # Ensure alignment is oriented [encoder_steps, decoder_frames]
    if alignment.shape[0] > alignment.shape[1]:
        alignment = alignment.T

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Decoder Frames (Time)")
    ax.set_ylabel("Encoder Tokens (Characters)")

    if text_chars and len(text_chars) == alignment.shape[0]:
        ax.set_yticks(range(len(text_chars)))
        ax.set_yticklabels(text_chars, fontsize=8)

    fig.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    return fig


def figure_to_image_bytes(fig: plt.Figure) -> bytes:
    """Convert matplotlib figure to PNG image bytes for web apps."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img_bytes = buf.getvalue()
    plt.close(fig)
    return img_bytes
