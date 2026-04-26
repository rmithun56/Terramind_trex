"""
infer.py — T-Rex crop type mapper entry point
Usage:
    python infer.py --demo                   # uses sample_input/sample_s1.tif
    python infer.py --tif path/to/file.tif   # any Sentinel GeoTIFF
    python infer.py --tif file.tif --weights model.pth
"""

import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

CLASSES = ["Background", "Wheat", "Corn", "Soybean", "Sunflower", "Rice", "Other"]
COLORS  = ["#000000", "#f5c518", "#00cc44", "#7b2d8b", "#ff8800", "#0088ff", "#aaaaaa"]
NUM_CLASSES = len(CLASSES)


class TerraMindUNet(nn.Module):
    """
    Lightweight UNet segmentation head in the style of TerraMind TiM.
    Encoder: 3-stage conv (64 -> 128 -> 256).
    Decoder: skip connections back to original resolution.
    """
    def __init__(self, in_channels: int = 14, num_classes: int = 7):
        super().__init__()
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.dec2 = self._block(256 + 128, 128)
        self.dec1 = self._block(128 + 64, 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat([e3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.head(d1)


def load_tif(path: str) -> np.ndarray:
    """Load a GeoTIFF and return float32 array (C, H, W)."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    print(f"  Loaded {Path(path).name}: shape={data.shape}, "
          f"range=[{data.min():.2f}, {data.max():.2f}]")
    return data


def normalize(data: np.ndarray) -> np.ndarray:
    """Per-band zero-mean unit-variance normalisation."""
    mean = data.mean(axis=(1, 2), keepdims=True)
    std  = data.std(axis=(1, 2), keepdims=True) + 1e-6
    return (data - mean) / std


def to_14ch(data: np.ndarray) -> np.ndarray:
    """Pad or trim to exactly 14 channels (S1+S2 fused)."""
    c = data.shape[0]
    if c < 14:
        repeats = (14 // c) + 1
        data = np.tile(data, (repeats, 1, 1))
    return data[:14]


def run_inference(tif_path: str, weights_path: str = None) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    raw  = load_tif(tif_path)
    data = normalize(to_14ch(raw))

    model = TerraMindUNet(in_channels=14, num_classes=NUM_CLASSES).to(device)
    if weights_path and Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"  Weights loaded from {weights_path}")
    else:
        print("  No weights file — running with random init (demo mode)")

    model.eval()
    tensor = torch.tensor(data).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).squeeze().cpu().numpy()

    return pred


def save_outputs(pred: np.ndarray, out_dir: str = "outputs"):
    Path(out_dir).mkdir(exist_ok=True)

    # Save raw prediction
    npy_path = f"{out_dir}/demo_prediction.npy"
    np.save(npy_path, pred)

    # Save colour map
    cmap = mcolors.ListedColormap(COLORS)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(pred, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1)
    cbar = plt.colorbar(im, ax=ax, ticks=range(NUM_CLASSES))
    cbar.ax.set_yticklabels(CLASSES)
    ax.set_title("T-Rex Crop Type Map (TerraMind-style UNet)", fontsize=12)
    ax.axis("off")
    png_path = f"{out_dir}/demo_prediction.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved: {npy_path}")
    print(f"  Saved: {png_path}")


def print_distribution(pred: np.ndarray):
    unique, counts = np.unique(pred, return_counts=True)
    print("\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        label = CLASSES[cls] if cls < NUM_CLASSES else f"Class {cls}"
        pct   = cnt / pred.size * 100
        bar   = "█" * int(pct / 2)
        print(f"  {label:<12} {pct:5.1f}%  {bar}")


def main():
    parser = argparse.ArgumentParser(description="T-Rex crop type mapper")
    parser.add_argument("--demo",    action="store_true",
                        help="Run on bundled sample_input/sample_s1.tif")
    parser.add_argument("--tif",     type=str, default=None,
                        help="Path to a Sentinel GeoTIFF")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to .pth weights file (optional)")
    parser.add_argument("--out",     type=str, default="outputs",
                        help="Output directory (default: outputs/)")
    args = parser.parse_args()

    if args.demo:
        candidates = list(Path("sample_input").glob("*.tif")) + \
                     list(Path("data").glob("*.tif"))
        if not candidates:
            print("No .tif found in sample_input/ or data/ — generating synthetic input")
            Path("sample_input").mkdir(exist_ok=True)
            dummy = np.random.rand(14, 256, 256).astype(np.float32)
            with rasterio.open(
                "sample_input/synthetic.tif", "w",
                driver="GTiff", height=256, width=256, count=14, dtype="float32"
            ) as dst:
                dst.write(dummy)
            tif_path = "sample_input/synthetic.tif"
        else:
            tif_path = str(candidates[0])
    elif args.tif:
        tif_path = args.tif
    else:
        parser.print_help()
        return

    print(f"\n=== T-Rex Crop Type Mapper ===")
    print(f"  Input: {tif_path}")

    pred = run_inference(tif_path, args.weights)
    print_distribution(pred)
    save_outputs(pred, args.out)

    print("\n=== Done! ===")


if __name__ == "__main__":
    main()
