# 🗣️ AI Text-to-Speech (TTS) Voice Generation System (LSTM + Attention)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-LJ%20Speech-success.svg)](https://keithito.com/LJ-Speech-Dataset/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end **Neural Text-to-Speech (TTS) Voice Generation System** that converts written English text into natural, intelligible audio speech. Designed for accessibility, virtual assistants, dynamic storytelling, and custom synthetic voice applications.

Built with a **Bidirectional LSTM Encoder**, **Location-Aware Additive Attention Alignment**, an **Autoregressive LSTM Decoder**, and a **5-Layer Residual PostNet**, backed by a fast **Griffin-Lim Acoustic Vocoder** and an interactive **Streamlit Voice Generation Studio**.

---

## ✨ Key Features

- **End-to-End Neural Pipeline**: Fully differentiable sequence-to-sequence neural architecture translating character sequences into 80-channel log-Mel spectrograms.
- **Location-Aware Attention**: Incorporates cumulative attention history convolutions to maintain monotonic text-to-speech alignment and prevent word skipping or repeating.
- **PostNet Residual Enhancement**: 5-layer 1D convolutional residual network refining high-frequency acoustic details and spectral fidelity.
- **Griffin-Lim Phase Retrieval**: Efficient, lightweight phase estimation for on-device waveform synthesis without requiring complex neural vocoder setups.
- **Interactive Streamlit Web Studio**: Full-featured graphical studio with text presets, interactive waveform and spectrogram visualizers, attention alignment heatmaps, and audio modulation (speed, pitch, Griffin-Lim iterations).
- **CLI & Scriptable Inference**: Fast standalone Python inference script for batch processing and terminal usage.
- **Multi-Task Loss Training**: Jointly optimizes initial Mel reconstruction (MSE), PostNet-refined Mel reconstruction (MSE), and binary stop token prediction (BCE) with gradient clipping.
- **LJ Speech Compatibility**: Native integration with the standard single-speaker LJ Speech Dataset (`mathurinache/the-lj-speech-dataset`).

---

## 🏛️ System Architecture

```text
Raw Text Input (e.g. "Dr. Smith had 42 apples!")
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  1. Text Normalizer & Phonetic Character Tokenizer     │
│     - Expands numbers: "42" -> "forty two"             │
│     - Expands abbreviations: "Dr." -> "Doctor"         │
│     - Maps characters to token IDs with EOS token      │
└────────────────────────────────────────────────────────┘
       │ Token IDs [Batch, Text_Len]
       ▼
┌────────────────────────────────────────────────────────┐
│  2. Text Encoder (Conv1D + Bidirectional LSTM)         │
│     - Character Embeddings (256 dims)                  │
│     - 3x 1D Convolutions (kernel 5x1, BatchNorm, ReLU) │
│     - 1x Bidirectional LSTM (256x2 = 512 context dim)  │
└────────────────────────────────────────────────────────┘
       │ Context Memory [Batch, Text_Len, 512]
       ▼
┌────────────────────────────────────────────────────────┐
│  3. Location-Aware Additive Attention Alignment        │
│     - Aligns text phonemes with temporal audio frames  │
│     - 1D Convolution over cumulative attention history │
│     - Computes monotonic alignment distribution α_t    │
└────────────────────────────────────────────────────────┘
       │ Context Vectors [Batch, 512]
       ▼
┌────────────────────────────────────────────────────────┐
│  4. Autoregressive Acoustic Decoder                    │
│     - PreNet: 2-layer FC with 0.5 dropout bottleneck   │
│     - 2-Layer Uni-directional LSTM (512 hidden state)  │
│     - Mel Linear Projection: 80 Log-Mel channels       │
│     - Stop Token Head: Sigmoid end-of-speech detector  │
└────────────────────────────────────────────────────────┘
       │ Initial 80-Channel Mel Spectrogram
       ▼
┌────────────────────────────────────────────────────────┐
│  5. PostNet Residual Refinement                        │
│     - 5-Layer 1D Convolutions with Tanh & BatchNorm    │
│     - Mel_final = Mel_initial + PostNet(Mel_initial)   │
└────────────────────────────────────────────────────────┘
       │ Refined Log-Mel Spectrogram
       ▼
┌────────────────────────────────────────────────────────┐
│  6. Griffin-Lim Acoustic Vocoder Engine                │
│     - Iterative phase estimation from Mel Magnitudes   │
│     - Pitch shift & playback rate adjustment           │
└────────────────────────────────────────────────────────┘
       │
       ▼
Natural Audio Speech Waveform (.wav, 22,050 Hz)
```

---

## 📁 Repository Structure

```
LSTM-RNN/
├── config.py             # Audio, model, and training hyperparameters
├── text/
│   ├── __init__.py
│   ├── symbols.py        # Character vocabulary and special token mappings
│   ├── cleaner.py        # Text normalization (numbers, abbreviations, punctuation)
│   └── tokenizer.py      # Text-to-sequence and sequence-to-text conversion
├── audio/
│   ├── __init__.py
│   ├── audio_processing.py # Mel filterbank, STFT, Griffin-Lim vocoder, audio I/O
│   └── visualizer.py     # Matplotlib waveform, spectrogram, and alignment plotting
├── models/
│   ├── __init__.py
│   ├── encoder.py        # Character embedding + 3x Conv1D + BiLSTM
│   ├── attention.py      # Location-aware additive attention mechanism
│   ├── decoder.py        # PreNet + 2-layer Uni-directional LSTM + Stop Token head
│   ├── postnet.py        # 5-layer residual 1D convolutional refinement network
│   └── tts_lstm.py       # Full end-to-end TextToSpeechLSTM model
├── dataset.py            # LJSpeech loader, dynamic batch collator, sample generator
├── train.py              # Multi-task loss training loop with gradient clipping & checkpoints
├── inference.py          # High-level speech synthesis pipeline & CLI tool
├── app.py                # Streamlit Web Studio (Interactive Voice Generation & Visualizer)
├── test_system.py        # Comprehensive unit & pipeline test suite
├── checkpoints/          # Saved model weights (e.g. best_model.pt)
├── requirements.txt      # Dependency specification
└── README.md             # Project documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/prudhvi-raju-jubburu/LSTM-RNN.git
cd LSTM-RNN

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

Run the automated test suite to ensure all components (tokenizer, audio processing, model modules, and synthetic forward pass) function properly:

```bash
python test_system.py
```

---

## 🖥️ Interactive Web Studio

Launch the interactive Streamlit Voice Generation Studio:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The studio features:
- **Text Input & Presets**: Pre-populated examples for speech synthesis, numbers, questions, and narrative paragraphs.
- **Acoustic Modulation Controls**: Sliders for playback speed (0.5x to 2.0x), pitch semitone adjustment (-6 to +6 st), and Griffin-Lim iterations (20 to 100).
- **Spectrogram & Waveform Visualizer**: Real-time plotting of the generated audio waveform and the 80-channel Mel spectrogram.
- **Attention Alignment Heatmap**: Inspect how the attention weights track character phonemes across time frames.
- **Audio Download**: One-click download of the synthesized `.wav` file.

---

## 🎙️ Command-Line Inference

Generate speech directly from your terminal:

```bash
python inference.py --text "Artificial intelligence voice generation using LSTM." --output speech.wav
```

### Synthesis Arguments & Flags:
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--text` | `"Hello world!"` | Input text string to synthesize |
| `--output` | `output.wav` | Destination file path for generated audio |
| `--speed` | `1.0` | Speed multiplier (`0.5` = half speed, `1.5` = 50% faster) |
| `--pitch` | `0.0` | Pitch shift in semitones (`+2.0` = higher, `-2.0` = lower) |
| `--iters` | `60` | Number of Griffin-Lim phase recovery iterations |
| `--checkpoint`| `checkpoints/best_model.pt` | Path to trained model weights |
| `--device` | `cuda` (if available) | Computation device (`cpu` or `cuda`) |

---

## 🏋️ Training the Model

### Training with Synthetic / Demo Data
To test training without downloading the full dataset:
```bash
python train.py --epochs 10 --batch_size 4 --lr 0.001
```

### Training on the LJ Speech Dataset
1. Download or locate your LJ Speech dataset folder containing `metadata.csv` and the `wavs/` directory.
2. Run the training script:
```bash
python train.py --data_path "path/to/LJSpeech-1.1" --epochs 50 --batch_size 16 --lr 0.001
```

Checkpoints will automatically be saved to `checkpoints/best_model.pt` and `checkpoints/checkpoint_epoch_X.pt`.

---

## 🔬 Mathematical Formulation

### 1. Location-Aware Attention
The alignment score $e_{i,j}$ at decoder time step $i$ for encoder output $h_j$ is given by:
$$e_{i,j} = v_a^\top \tanh\left(W_s s_i + W_h h_j + W_f f_{i,j} + b_a\right)$$
where $f_i = F * \alpha_{i-1}$ is the result of convolving the cumulative attention weights with 32 1D filters, ensuring monotonic progression.

### 2. Multi-Task Loss Objective
The training loss combines linear Mel prediction, PostNet residual prediction, and the end-of-speech stop token loss:
$$\mathcal{L}_{\text{total}} = \text{MSE}(\hat{M}_{\text{init}}, M) + \text{MSE}(\hat{M}_{\text{post}}, M) + \text{BCE}(\hat{S}, S)$$

Gradient clipping with $\|\mathbf{g}\|_2 \le 1.0$ is applied to stabilize backpropagation through time across recurrent layers.

---

## 📦 Dependencies

- **PyTorch** $\ge$ 2.0.0
- **NumPy** $\ge$ 1.24.0
- **SciPy** $\ge$ 1.10.0
- **SoundFile** $\ge$ 0.12.0
- **Matplotlib** $\ge$ 3.7.0
- **Streamlit** $\ge$ 1.30.0
- **kagglehub** $\ge$ 0.2.0

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.
