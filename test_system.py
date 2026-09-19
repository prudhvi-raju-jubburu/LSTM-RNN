"""
Comprehensive Automated Verification Suite for Neural TTS System.
"""
import os
import torch
import numpy as np
from config import default_config
from text.cleaner import clean_text
from text.tokenizer import text_to_sequence, sequence_to_text
from audio.audio_processing import AudioProcessor
from audio.visualizer import plot_waveform_and_spectrogram, plot_attention_alignment
from models.tts_lstm import TextToSpeechLSTM
from dataset import LJSpeechDataset, TTSCollate
from inference import TTSSynthesizer


def test_text_frontend():
    print("[1/5] Testing Text Frontend...")
    raw = "Dr. Jekyll had 100 experiments on Sept. 1st!"
    cleaned = clean_text(raw)
    seq = text_to_sequence(raw, add_eos=True)
    reconstructed = sequence_to_text(seq)

    assert "doctor" in cleaned, f"Abbreviation expansion failed: {cleaned}"
    assert "one hundred" in cleaned, f"Number expansion failed: {cleaned}"
    assert len(seq) > 0, "Sequence empty"
    print("      [OK] Cleaned text:", cleaned)
    print("      [OK] Token sequence length:", len(seq))


def test_audio_processing():
    print("[2/5] Testing Audio Processing & Griffin-Lim Vocoder...")
    ap = AudioProcessor(default_config.audio)
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(default_config.audio.sample_rate * duration_sec), endpoint=False)
    synth_wav = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    mel = ap.wav_to_mel(synth_wav)
    assert mel.shape[0] == 80, f"Expected 80 mel channels, got {mel.shape[0]}"

    reconstructed_wav = ap.mel_to_wav(mel, iters=10)
    assert len(reconstructed_wav) > 0, "Reconstructed audio is empty"
    print("      [OK] Generated Mel shape:", tuple(mel.shape))
    print("      [OK] Inverted Waveform shape:", reconstructed_wav.shape)


def test_model_architecture():
    print("[3/5] Testing Model Architecture & Attention...")
    model = TextToSpeechLSTM(default_config)
    batch_size = 2
    seq_len = 15
    text_seq = torch.randint(0, 30, (batch_size, seq_len))
    text_lengths = torch.tensor([seq_len, seq_len - 3])
    mel_targets = torch.randn(batch_size, 35, 80)

    # 1. Training mode
    train_out = model(text_seq, text_lengths, mel_targets=mel_targets)
    assert train_out["mel_initial"].shape == (batch_size, 35, 80)
    assert train_out["mel_postnet"].shape == (batch_size, 35, 80)
    assert train_out["stop_tokens"].shape == (batch_size, 35)
    assert train_out["alignments"].shape == (batch_size, 35, seq_len)

    # 2. Inference mode
    infer_out = model.generate(text_seq[:1], text_lengths[:1], max_steps=20)
    assert infer_out["mel_postnet"].shape[0] == 1
    assert infer_out["mel_postnet"].shape[2] == 80
    print("      [OK] Model train & inference forward passes succeeded.")


def test_dataset_and_collator():
    print("[4/5] Testing Dataset & Batch Collator...")
    ds = LJSpeechDataset(max_samples=4)
    assert len(ds) >= 1, "Dataset empty"
    item = ds[0]
    assert "text" in item and "mel" in item

    collate_fn = TTSCollate()
    batch = collate_fn([ds[i] for i in range(min(2, len(ds)))])
    assert "text" in batch and "mel" in batch and "stop_targets" in batch
    print(f"      [OK] Dataset contains {len(ds)} items. Collated batch successfully.")


def test_end_to_end_synthesis():
    print("[5/5] Testing End-to-End Synthesizer Pipeline...")
    synthesizer = TTSSynthesizer()
    out_path = "test_verification.wav"
    result = synthesizer.synthesize(
        text="Hello world from neural text to speech.",
        speed=1.1,
        pitch_shift=1.0,
        griffin_lim_iters=15,
        max_steps=50,
    )

    wav = result["wav"]
    sr = result["sample_rate"]
    assert len(wav) > 0, "Synthesized wav is empty"
    synthesizer.save_wav(wav, out_path)
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
    os.remove(out_path)
    print("      [OK] Synthesized and verified audio output.")


if __name__ == "__main__":
    print("=" * 60)
    print("[*] Running Neural TTS Verification Suite")
    print("=" * 60)
    test_text_frontend()
    test_audio_processing()
    test_model_architecture()
    test_dataset_and_collator()
    test_end_to_end_synthesis()
    print("=" * 60)
    print("[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
