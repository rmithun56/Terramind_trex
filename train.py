"""
train.py — T-Rex crop type mapper training script
Usage:
    python train.py                         # train on data/ with default settings
    python train.py --epochs 20 --lr 5e-4
    python train.py --s1 data/s1.tif --s2 data/s2.tif --epochs 15
"""

from model import TerraMindUNet
import argparse
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rasterio

# ── Model (same as infer.py — single source of truth via src/) ────────────
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ── Dataset ───────────────────────────────────────────────────────────────


class SentinelCropDataset(Dataset):
    """
    Patch-based dataset built from a fused (C, H, W) tensor and a (H, W)
    label map.  Extracts overlapping patches at the given stride.
    """

    def __init__(self, image: np.ndarray, labels: np.ndarray,
                 patch_size: int = 64, stride: int = 32):
        self.patches: list = []
        self.label_patches: list = []
        _, H, W = image.shape
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                self.patches.append(image[:, y:y+patch_size, x:x+patch_size])
                self.label_patches.append(
                    labels[y:y+patch_size, x:x+patch_size])

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, i):
        return (
            torch.tensor(self.patches[i],       dtype=torch.float32),
            torch.tensor(self.label_patches[i], dtype=torch.long),
        )


# ── Data helpers ──────────────────────────────────────────────────────────

def load_and_normalize(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    mean = data.mean(axis=(1, 2), keepdims=True)
    std = data.std(axis=(1, 2),  keepdims=True) + 1e-6
    return (data - mean) / std


def fuse_modalities(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Concatenate S1 (2 ch) and S2 (12 ch) → 14-channel tensor."""
    return np.concatenate([s1, s2], axis=0)


def make_synthetic_labels(H: int, W: int, num_classes: int = 7,
                          seed: int = 42) -> np.ndarray:
    """
    Placeholder labels used when no ground-truth annotation is available.
    Replace with real LPIS / CDL data for production training.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_classes, size=(H, W), dtype=np.int64)


# ── Training loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = out.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += lbls.numel()
    return total_loss / len(loader), correct / total * 100


def compute_miou(model, loader, device, num_classes: int = 7) -> float:
    model.eval()
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs).argmax(dim=1)
            for c in range(num_classes):
                pred_c = preds == c
                label_c = lbls == c
                intersection[c] += (pred_c & label_c).sum().item()
                union[c] += (pred_c | label_c).sum().item()
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="T-Rex crop type mapper training")
    parser.add_argument("--s1",         type=str, default=None,
                        help="Path to Sentinel-1 GeoTIFF")
    parser.add_argument("--s2",         type=str, default=None,
                        help="Path to Sentinel-2 GeoTIFF")
    parser.add_argument("--labels",     type=str, default=None,
                        help="Path to labels .npy (H,W) int array")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride",     type=int, default=32)
    parser.add_argument("--out",        type=str, default="outputs/trex_model.pth",
                        help="Where to save trained weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== T-Rex Training ===")
    print(f"  Device : {device}")
    print(
        f"  Epochs : {args.epochs}  |  LR: {args.lr}  |  Batch: {args.batch_size}\n")

    # ── Load data ────────────────────────────────────────────────────────
    s1_candidates = list(Path("data").glob("*s1*.tif")) + \
        list(Path("sample_input").glob("*s1*.tif"))
    s2_candidates = list(Path("data").glob("*s2*.tif")) + \
        list(Path("sample_input").glob("*s2*.tif"))

    s1_path = args.s1 or (str(s1_candidates[0]) if s1_candidates else None)
    s2_path = args.s2 or (str(s2_candidates[0]) if s2_candidates else None)

    if s1_path and s2_path:
        print(f"  S1: {s1_path}")
        print(f"  S2: {s2_path}")
        s1 = load_and_normalize(s1_path)
        s2 = load_and_normalize(s2_path)
        # Pad S1 to 2 ch, S2 to 12 ch if needed
        if s1.shape[0] < 2:
            s1 = np.tile(s1, (2, 1, 1))[:2]
        if s2.shape[0] < 12:
            repeats = (12 // s2.shape[0]) + 1
            s2 = np.tile(s2, (repeats, 1, 1))[:12]
        fused = fuse_modalities(s1, s2)
    else:
        print("  No S1/S2 found — using synthetic 14-channel input")
        fused = np.random.rand(14, 448, 448).astype(np.float32)

    _, H, W = fused.shape
    print(f"  Fused tensor: {fused.shape}")

    # ── Labels ───────────────────────────────────────────────────────────
    if args.labels and Path(args.labels).exists():
        labels = np.load(args.labels).astype(np.int64)
        print(f"  Labels loaded from {args.labels}")
    else:
        print("  Using synthetic labels (replace with LPIS/CDL for real training)")
        labels = make_synthetic_labels(H, W)

    # ── Dataset / loader ─────────────────────────────────────────────────
    dataset = SentinelCropDataset(fused, labels,
                                  patch_size=args.patch_size,
                                  stride=args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    print(f"  Patches: {len(dataset)}  |  Batches/epoch: {len(loader)}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model = TerraMindUNet(in_channels=14, num_classes=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training ──────────────────────────────────────────────────────────
    best_miou = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, acc = train_one_epoch(
            model, loader, optimizer, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 2 == 0 or epoch == args.epochs:
            miou = compute_miou(model, loader, device)
            flag = " ← best" if miou > best_miou else ""
            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), args.out)
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {loss:.4f} | Acc: {acc:.1f}% | "
                  f"mIoU: {miou:.3f} | {elapsed:.1f}s{flag}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {loss:.4f} | Acc: {acc:.1f}% | {elapsed:.1f}s")

    print(f"\n  Best mIoU : {best_miou:.3f}")
    print(f"  Weights   : {args.out}")
    print("=== Training complete ===\n")


if __name__ == "__main__":
    main()
