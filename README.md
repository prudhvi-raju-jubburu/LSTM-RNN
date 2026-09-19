# AI Text-to-Speech (TTS) Voice Generation System (LSTM + Attention)

An end-to-end Neural Text-to-Speech Voice Generation System that converts written English text into natural, intelligible audio speech for accessibility, virtual assistants, storytelling, and digital voice applications.

Built with an **LSTM-based Encoder-Decoder Architecture with Location-Aware Attention**, trained on the **LJ Speech Dataset** (`mathurinache/the-lj-speech-dataset`), and powered by an interactive **Streamlit Voice Generation Studio**.

---

## 🏛️ System Architecture

```
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
LSTM/
├── config.py             # Audio, model, and training hyperparameters
├── text/
│   ├── symbols.py        # Character vocabulary and special token mappings
│   ├── cleaner.py        # Text normalization (numbers, abbreviations, punctuation)
│   └── tokenizer.py      # Text-to-sequence and sequence-to-text conversion
├── audio/
│   ├── audio_processing.py # Mel filterbank, STFT, Griffin-Lim vocoder, audio I/O
│   └── visualizer.py     # Matplotlib waveform, spectrogram, and alignment plotting
├── models/
│   ├── encoder.py        # Character embedding + 3x Conv1D + BiLSTM
│   ├── attention.py      # Location-aware additive attention mechanism
│   ├── decoder.py        # PreNet + 2-layer Uni-directional LSTM + Stop Token head
│   ├── postnet.py        # 5-layer residual 1D convolutional refinement network
│   └── tts_lstm.py       # Full end-to-end TextToSpeechLSTM model
├── dataset.py            # LJSpeech loader, dynamic batch collator, sample generator
├── train.py              # Multi-task loss training loop with gradient clipping & checkpoints
├── inference.py          # High-level speech synthesis pipeline & CLI tool
├── app.py                # Streamlit Web Studio (Interactive Voice Generation & Visualizer)
├── checkpoints/          # Saved model weights (best_model.pt)
├── requirements.txt      # Dependency specification
└── README.md             # Project documentation
```

---

## 🚀 Quick Start

### 1. Installation
Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Studio
Run the Streamlit application:
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

---

## 🎙️ Command-Line Speech Synthesis

Generate speech directly from the terminal:
```bash
python inference.py --text "Artificial intelligence voice generation using LSTM." --output speech.wav
```

### Optional Acoustic Flags:
- `--speed 1.2`: Speeds up playback by 20%.
- `--pitch 2.0`: Increases pitch by +2 semitones.
- `--iters 60`: Number of Griffin-Lim phase estimation iterations (higher = crisper audio).
- `--checkpoint checkpoints/best_model.pt`: Custom model weights path.

---

## 🏋️ Training the Model

Train the LSTM acoustic model with multi-task loss ($\mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{postnet}} + \mathcal{L}_{\text{stop}}$):
```bash
python train.py --epochs 20 --batch_size 4 --lr 0.001
```

To train directly on the full LJSpeech dataset after downloading:
```bash
python train.py --data_path "C:/Users/jubbu/.cache/kagglehub/datasets/mathurinache/the-lj-speech-dataset" --epochs 50 --batch_size 16
```

---

## 🔬 Multi-Task Loss Objective

$$\mathcal{L}_{\text{total}} = \text{MSE}(\text{Mel}_{\text{init}}, \text{Mel}_{\text{target}}) + \text{MSE}(\text{Mel}_{\text{post}}, \text{Mel}_{\text{target}}) + \text{BCE}(\text{Stop}_{\text{pred}}, \text{Stop}_{\text{target}})$$

Gradient clipping ($\|\mathbf{g}\| \le 1.0$) is enforced during backpropagation to prevent exploding gradients through the recurrent LSTM unrolling steps.
