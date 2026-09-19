"""
NeuralVoice Studio - AI Text-to-Speech Voice Generation Web Application
Powered by LSTM + Location-Aware Attention & Griffin-Lim Vocoder.
"""
import os
import time
import io
import numpy as np
import torch
import streamlit as st

from config import Config, default_config
from text.cleaner import clean_text
from text.symbols import NUM_SYMBOLS
from inference import TTSSynthesizer
from audio.visualizer import plot_waveform_and_spectrogram, plot_attention_alignment
from dataset import LJSpeechDataset
import soundfile as sf

# Set Streamlit page layout
st.set_page_config(
    page_title="NeuralVoice AI Studio | LSTM Text-to-Speech",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphic Dark UI Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(17, 24, 39, 1) 0%, rgba(10, 15, 29, 1) 90%);
        color: #F3F4F6;
    }

    /* Top glowing header banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.25);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        max-width: 800px;
        line-height: 1.5;
    }

    /* Status badge pill */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(129, 140, 248, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #C7D2FE;
        margin-right: 8px;
    }

    /* Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px -5px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }

    /* Metric counter box */
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-size: 1.05rem !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_synthesizer():
    """Cache TTS Synthesizer model instance."""
    return TTSSynthesizer()


synthesizer = get_synthesizer()

# --- HEADER SECTION ---
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎙️ NeuralVoice AI Studio</div>
        <div class="hero-subtitle">
            Next-generation Text-to-Speech synthesis system powered by 
            <strong>Bidirectional LSTM</strong>, <strong>Location-Aware Attention</strong>, and <strong>Griffin-Lim Acoustic Vocoder</strong>.
            Trained on the LJSpeech dataset for natural, intelligible voice generation.
        </div>
        <div style="margin-top: 14px;">
            <span class="badge-pill">⚡ Architecture: BiLSTM + Attention</span>
            <span class="badge-pill">🔊 Mel-Spectrogram: 80-Channels</span>
            <span class="badge-pill">⏱️ Sample Rate: 22,050 Hz</span>
            <span class="badge-pill">🎛️ PostNet: 5-Layer Residual Conv1D</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### 🎛️ Acoustic Control Center")
    st.markdown("Fine-tune acoustic rendering, prosody, and vocoder properties:")

    speed = st.slider(
        "⚡ Speaking Rate (Speed)",
        min_value=0.6,
        max_value=1.5,
        value=1.0,
        step=0.05,
        help="Accelerate or decelerate spoken speech playback rate.",
    )

    pitch = st.slider(
        "🎵 Pitch Inflection (Semitones)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        help="Transpose the fundamental vocal frequency higher or lower.",
    )

    iters = st.slider(
        "🔬 Griffin-Lim Phase Iterations",
        min_value=20,
        max_value=100,
        value=40,
        step=5,
        help="Higher iterations improve phase fidelity and acoustic clarity.",
    )

    with st.expander("🛠️ Advanced Decoder Settings", expanded=False):
        max_steps = st.number_input("Max Decoder Frames", min_value=100, max_value=1500, value=600, step=50)
        stop_thresh = st.slider("Stop Token Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    st.markdown("---")
    st.markdown("### 📦 Checkpoint Status")
    chk_info = synthesizer.checkpoint_info
    if chk_info:
        st.success(f"✅ Loaded Checkpoint (Epoch {chk_info['epoch']})")
        if chk_info["val_loss"]:
            st.caption(f"Validation Loss: `{chk_info['val_loss']:.4f}`")
    else:
        st.info("ℹ️ Neural model active with default weights.")

    if st.button("🔄 Reload Model Checkpoint"):
        st.cache_resource.clear()
        st.rerun()

    # Hardware status
    st.markdown("---")
    device_name = "CUDA GPU" if torch.cuda.is_available() else "Host CPU"
    st.markdown(f"**Hardware Acceleration:** `{device_name}`")
    total_params = sum(p.numel() for p in synthesizer.model.parameters())
    st.markdown(f"**Model Parameters:** `{total_params:,}`")


# --- MAIN TABS ---
tab_synth, tab_arch, tab_train = st.tabs(["🎙️ Voice Synthesis", "🧠 Model Architecture", "📊 LJSpeech & Training"])

with tab_synth:
    st.markdown("#### 📝 Input Text to Synthesize")

    # Quick prompt presets
    preset_cols = st.columns(4)
    preset_text = None
    with preset_cols[0]:
        if st.button("🤖 Assistant", use_container_width=True):
            preset_text = "Good morning! All smart home systems are online and running smoothly."
    with preset_cols[1]:
        if st.button("♿ Accessibility", use_container_width=True):
            preset_text = "The elevator is arriving at floor four. Doors opening on the right."
    with preset_cols[2]:
        if st.button("📖 Storytelling", use_container_width=True):
            preset_text = "The ancient clock tower echoed through the quiet midnight streets."
    with preset_cols[3]:
        if st.button("🔔 Notification", use_container_width=True):
            preset_text = "Reminder: Your team review meeting starts in fifteen minutes."

    default_prompt = preset_text or "Artificial intelligence text to speech synthesis converts written words into natural speech."

    text_input = st.text_area(
        label="Speech Script",
        value=default_prompt,
        height=110,
        placeholder="Type the message you would like the neural voice to speak...",
    )

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        generate_clicked = st.button("✨ Generate Natural Speech", use_container_width=True)

    with col_info:
        cleaned_preview = clean_text(text_input)
        st.caption(f"Normalized phonetic sequence: `{cleaned_preview[:80]}...` ({len(text_input)} chars)")

    if generate_clicked and text_input.strip():
        with st.spinner("🧠 Generating neural mel-spectrogram and synthesizing waveform..."):
            t0 = time.time()
            res = synthesizer.synthesize(
                text=text_input,
                speed=speed,
                pitch_shift=pitch,
                griffin_lim_iters=iters,
                max_steps=max_steps,
                stop_threshold=stop_thresh,
            )
            synth_time = time.time() - t0

            wav = res["wav"]
            sr = res["sample_rate"]
            mel = res["mel"]
            alignment = res["alignment"]
            duration = len(wav) / sr

        # Metrics row
        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(
                f"<div class='metric-box'><div class='metric-value'>{duration:.2f}s</div><div class='metric-label'>Audio Duration</div></div>",
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"<div class='metric-box'><div class='metric-value'>{synth_time:.2f}s</div><div class='metric-label'>Inference Latency</div></div>",
                unsafe_allow_html=True,
            )
        with m_col3:
            rtf = synth_time / max(0.01, duration)
            st.markdown(
                f"<div class='metric-box'><div class='metric-value'>{rtf:.2f}x</div><div class='metric-label'>Real-Time Factor</div></div>",
                unsafe_allow_html=True,
            )
        with m_col4:
            st.markdown(
                f"<div class='metric-box'><div class='metric-value'>{mel.shape[1]}</div><div class='metric-label'>Acoustic Frames</div></div>",
                unsafe_allow_html=True,
            )

        # Audio Player & Download
        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("### 🔊 Audio Playback")

        # Convert to WAV in-memory bytes
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        audio_bytes = buf.getvalue()

        audio_col1, audio_col2 = st.columns([3, 1])
        with audio_col1:
            st.audio(audio_bytes, format="audio/wav")
        with audio_col2:
            st.download_button(
                label="📥 Download .WAV",
                data=audio_bytes,
                file_name="neural_speech.wav",
                mime="audio/wav",
                use_container_width=True,
            )

        # Visualizations
        st.markdown("---")
        st.markdown("### 📊 Acoustic Analysis & Alignment")
        v_tab1, v_tab2 = st.tabs(["Spectrogram & Waveform", "Attention Alignment Matrix"])

        with v_tab1:
            fig_audio = plot_waveform_and_spectrogram(wav, mel, sample_rate=sr, title=f"'{text_input[:30]}...'")
            st.pyplot(fig_audio, clear_figure=True)

        with v_tab2:
            fig_align = plot_attention_alignment(alignment, title="Text-to-Frame Attention Alignment")
            st.pyplot(fig_align, clear_figure=True)
            st.caption("A sharp monotonic diagonal indicates strong phonetic-acoustic alignment between text characters and audio frames.")

with tab_arch:
    st.markdown("### 🏛️ Neural TTS Architecture (LSTM + Attention)")
    st.markdown(
        """
        The voice generation system is based on an **Encoder-Decoder neural architecture with Attention**, 
        custom-engineered for natural human speech synthesis:
        """
    )

    st.markdown(
        """
        ```
        Input Text String
              │
              ▼
        ┌────────────────────────────────────────────────────────┐
        │  1. Text Normalizer & Phonetic Character Tokenizer     │
        └────────────────────────────────────────────────────────┘
              │ Token IDs
              ▼
        ┌────────────────────────────────────────────────────────┐
        │  2. Text Encoder (Conv1D + Bidirectional LSTM)         │
        │     - 256-dim Character Embeddings                     │
        │     - 3x Conv1D layers (5x1 kernel, BatchNorm, ReLU)   │
        │     - 1x Bidirectional LSTM (512-dim Context Memory)   │
        └────────────────────────────────────────────────────────┘
              │ Context Memory [Batch, Length, 512]
              ▼
        ┌────────────────────────────────────────────────────────┐
        │  3. Location-Aware Additive Attention Alignment        │
        │     - Computes monotonic alignment α_t across frames   │
        │     - 1D Convolution over cumulative attention history │
        └────────────────────────────────────────────────────────┘
              │ Context Vectors [Batch, 512]
              ▼
        ┌────────────────────────────────────────────────────────┐
        │  4. Autoregressive Acoustic Decoder                    │
        │     - PreNet (2x Linear with 0.5 active Dropout)       │
        │     - 2-Layer Uni-directional LSTM (512 hidden state)  │
        │     - Mel Projection: Linear -> 80 Log-Mel channels    │
        │     - Stop Token Head: Sigmoid end-of-utterance        │
        └────────────────────────────────────────────────────────┘
              │ Initial 80-Channel Mel Spectrogram
              ▼
        ┌────────────────────────────────────────────────────────┐
        │  5. PostNet Residual Refinement                        │
        │     - 5-Layer 1D Convolutions (Tanh & BatchNorm)       │
        │     - Predicts residual corrections: Mel_final = M + R │
        └────────────────────────────────────────────────────────┘
              │ Refined Log-Mel Spectrogram
              ▼
        ┌────────────────────────────────────────────────────────┐
        │  6. Griffin-Lim Acoustic Vocoder                       │
        │     - Iterative phase estimation from Mel Magnitudes   │
        │     - Reconstructs continuous 22.05 kHz audio waveform │
        └────────────────────────────────────────────────────────┘
              │
              ▼
        Natural Speech Audio (.wav)
        ```
        """
    )

    st.markdown("#### 🔬 Detailed Module Specifications")
    arch_cols = st.columns(3)
    with arch_cols[0]:
        st.markdown(
            """
            **Encoder Specifications:**
            - Embedding Vocab: `100` characters & symbols
            - Embedding Dim: `256`
            - Convolutions: `3` layers (Kernel 5)
            - LSTM: `Bidirectional` (256x2 = 512)
            - Dropout: `0.5`
            """
        )
    with arch_cols[1]:
        st.markdown(
            """
            **Decoder & Attention:**
            - Attention Type: `Location-Aware Additive`
            - Attention Dim: `128`
            - PreNet: `256` units with 0.5 Dropout
            - Decoder LSTMs: `2` uni-directional layers (512)
            - Mel Channels: `80` Log-Mel bands
            """
        )
    with arch_cols[2]:
        st.markdown(
            """
            **Vocoder & PostNet:**
            - PostNet Layers: `5` Conv1D (256 channels)
            - Vocoder: `Griffin-Lim Phase Estimation`
            - FFT Window: `1024`
            - Hop Length: `256`
            - Sampling Rate: `22,050 Hz`
            """
        )

with tab_train:
    st.markdown("### 📊 LJSpeech Dataset & Model Training")
    st.markdown(
        """
        The **LJ Speech Dataset** contains 13,100 high-quality audio clips of non-fiction passages 
        read by a single female speaker, totaling ~24 hours of speech.
        """
    )

    ds = LJSpeechDataset(max_samples=20)
    st.info(f"📁 Active Dataset Directory: `{ds.dataset_path}` (Found {len(ds)} utterances)")

    st.markdown("#### 🎧 Sample Utterances from Dataset")
    sample_cols = st.columns(3)
    for idx in range(min(3, len(ds.items))):
        fid, txt, wpath = ds.items[idx]
        with sample_cols[idx]:
            st.markdown(f"**{fid}**")
            st.caption(f"\"{txt[:60]}...\"")
            if os.path.exists(wpath):
                st.audio(wpath)

    st.markdown("---")
    st.markdown("#### 🚀 In-App Training / Fine-Tuning")
    train_c1, train_c2, train_c3 = st.columns(3)
    with train_c1:
        num_epochs = st.number_input("Epochs", min_value=1, max_value=50, value=3)
    with train_c2:
        train_batch = st.number_input("Batch Size", min_value=1, max_value=16, value=4)
    with train_c3:
        train_lr = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)

    if st.button("▶️ Launch Training Run"):
        train_status = st.empty()
        progress_bar = st.progress(0)
        from train import train as run_training

        with st.spinner("Training model on utterances..."):
            train_status.info("🚀 Training in progress. Tracking losses across epochs...")
            try:
                run_training(
                    epochs=int(num_epochs),
                    batch_size=int(train_batch),
                    learning_rate=float(train_lr),
                    dataset_path=ds.dataset_path,
                    save_interval=1,
                )
                progress_bar.progress(100)
                train_status.success("✅ Training completed! Best checkpoint updated in `checkpoints/best_model.pt`")
                st.cache_resource.clear()
            except Exception as e:
                train_status.error(f"Training error: {e}")
