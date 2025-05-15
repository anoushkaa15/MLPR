import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import math

from train.config import (
    BASE_PATH, DEVICE, FRAME_RATE, INTERVAL_FRAMES, STRIDE_FRAMES,
    MEL_SIZE, SSM_TARGET_SIZE, BATCH_SIZE, EPOCHS, VAL_SPLIT, CHECKPOINT_PATH
)

from utils.dataset import MultiFolderGuitarToDrumDataset, MelSSMDataset
from models.transformer import DrumTransformer


# ========== TRAINING ==========
def train_model():
    raw = MultiFolderGuitarToDrumDataset()
    total_songs = len(raw.samples)
    val_count = math.floor(VAL_SPLIT * total_songs)
    train_count = total_songs - val_count

    train_samples = raw.samples[:train_count]
    val_samples = raw.samples[train_count:]

    train_ds = MelSSMDataset(train_samples)
    val_ds = MelSSMDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"Songs: {total_songs} | Train: {train_count} | Val: {val_count}")
    print(f"Segments: Train {len(train_ds)} | Val {len(val_ds)}")

    model = DrumTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()
    scaler = GradScaler()
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        model.train()
        total_tr = 0
        for gm, gs, dm in train_loader:
            gm, gs, dm = gm.to(DEVICE), gs.to(DEVICE), dm.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                pm = model(gm, gs)
                loss = 0.5 * mse_loss_fn(pm, dm) + 0.5 * mae_loss_fn(pm, dm)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_tr += loss.item()

        model.eval()
        total_vl, total_mae = 0, 0
        with torch.no_grad():
            for gm, gs, dm in val_loader:
                gm, gs, dm = gm.to(DEVICE), gs.to(DEVICE), dm.to(DEVICE)
                with autocast():
                    pm = model(gm, gs)
                    loss = 0.5 * mse_loss_fn(pm, dm) + 0.5 * mae_loss_fn(pm, dm)
                total_vl += loss.item()
                total_mae += F.l1_loss(pm, dm).item()

        avg_tr = total_tr / len(train_loader)
        avg_vl = total_vl / len(val_loader)
        avg_ma = total_mae / len(val_loader)

        print(f"Train Loss: {avg_tr:.4f} | Val Loss: {avg_vl:.4f} | MAE: {avg_ma:.4f}")
        if epoch == 1 or (avg_vl < best_val):
            best_val = avg_vl
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved checkpoint (val loss = {avg_vl:.4f})")

if __name__ == "__main__":
    train_model()
