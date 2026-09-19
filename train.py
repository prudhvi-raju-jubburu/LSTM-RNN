"""
Training script for AI Text-to-Speech Voice Generation System (LSTM + Attention).
Supports multi-task loss, gradient clipping, checkpointing, and evaluation.
"""
import os
import argparse
import time
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import Config, default_config
from dataset import LJSpeechDataset, TTSCollate
from models.tts_lstm import TextToSpeechLSTM


class TTSLoss(nn.Module):
    """
    Multi-task loss:
    1. MSE Loss on initial decoder Mel-spectrogram
    2. MSE Loss on PostNet refined Mel-spectrogram
    3. Binary Cross-Entropy Loss on Stop Token prediction
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        model_out: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mel_target = targets["mel"]
        stop_target = targets["stop_targets"]

        loss_initial = self.mse_loss(model_out["mel_initial"], mel_target)
        loss_postnet = self.mse_loss(model_out["mel_postnet"], mel_target)
        loss_stop = self.bce_loss(model_out["stop_tokens"], stop_target)

        total_loss = loss_initial + loss_postnet + loss_stop

        return {
            "total_loss": total_loss,
            "loss_initial": loss_initial,
            "loss_postnet": loss_postnet,
            "loss_stop": loss_stop,
        }


def train(
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    dataset_path: str = None,
    max_samples: int = None,
    checkpoint_dir: str = None,
    device: str = None,
    save_interval: int = 5,
):
    cfg = default_config
    device = device or cfg.train.device
    checkpoint_dir = checkpoint_dir or cfg.train.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"[*] Initializing AI TTS (LSTM + Attention) Training on device: {device}")

    # 1. Dataset & DataLoader
    dataset = LJSpeechDataset(dataset_path=dataset_path, config=cfg, max_samples=max_samples)
    print(f"[*] Loaded dataset with {len(dataset)} utterances.")

    if len(dataset) > 4:
        val_size = max(1, int(len(dataset) * 0.1))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, dataset

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TTSCollate(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TTSCollate(),
    )

    # 2. Model, Loss, Optimizer
    model = TextToSpeechLSTM(cfg).to(device)
    criterion = TTSLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    start_time = time.time()

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_init_loss = 0.0
        train_post_loss = 0.0
        train_stop_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            text = batch["text"].to(device)
            text_lens = batch["text_lengths"].to(device)
            mel = batch["mel"].to(device)
            stop_targets = batch["stop_targets"].to(device)

            optimizer.zero_grad()

            # Forward pass with teacher forcing
            out = model(text, text_lens, mel_targets=mel)

            # Compute losses
            loss_dict = criterion(out, {"mel": mel, "stop_targets": stop_targets})
            total_loss = loss_dict["total_loss"]

            total_loss.backward()

            # Gradient clipping to stabilize LSTM training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip_thresh)
            optimizer.step()

            train_total_loss += total_loss.item()
            train_init_loss += loss_dict["loss_initial"].item()
            train_post_loss += loss_dict["loss_postnet"].item()
            train_stop_loss += loss_dict["loss_stop"].item()

        scheduler.step()

        num_batches = max(1, len(train_loader))
        avg_train_loss = train_total_loss / num_batches

        # Validation
        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                text = batch["text"].to(device)
                text_lens = batch["text_lengths"].to(device)
                mel = batch["mel"].to(device)
                stop_targets = batch["stop_targets"].to(device)

                out = model(text, text_lens, mel_targets=mel)
                loss_dict = criterion(out, {"mel": mel, "stop_targets": stop_targets})
                val_total_loss += loss_dict["total_loss"].item()

        avg_val_loss = val_total_loss / max(1, len(val_loader))

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss: {avg_train_loss:.4f} (Init: {train_init_loss/num_batches:.3f}, Post: {train_post_loss/num_batches:.3f}, Stop: {train_stop_loss/num_batches:.3f}) | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # Checkpoint Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "config": cfg,
                },
                best_path,
            )

        if epoch % save_interval == 0 or epoch == epochs:
            latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": cfg,
                },
                latest_path,
            )

    elapsed = time.time() - start_time
    print(f"[*] Training finished in {elapsed:.1f}s. Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Text-to-Speech LSTM Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data_path", type=str, default=None, help="Path to LJSpeech dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of dataset samples")
    parser.add_argument("--save_interval", type=int, default=5, help="Checkpoint saving interval")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dataset_path=args.data_path,
        max_samples=args.max_samples,
        save_interval=args.save_interval,
    )
