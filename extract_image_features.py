"""
Offline Image Feature Extraction for IDBS
===========================================
Extracts feature vectors from StateFarm images using a pretrained ResNet18.

This is a ONE-TIME pre-processing step:
  - Runs a frozen ResNet18 (no gradient) over all StateFarm images
  - Saves 512-d feature vectors to  data/processed/statefarm/image_features.pkl
  - Takes ~15-25 min on a typical CPU — but only needs to run ONCE
  - After this, all training scripts load flat tensors instead of raw images

Output format (image_features.pkl):
  {
    "features":  np.ndarray  (N, 512)   — L2-normalised feature vectors
    "labels":    np.ndarray  (N,)       — integer class labels  (0–9)
    "filepaths": list[str]             — source image paths
    "backbone":  str                   — backbone model name
  }

Usage:
  python extract_image_features.py                         # defaults
  python extract_image_features.py --batch_size 64         # faster if RAM allows
  python extract_image_features.py --backbone resnet34     # larger backbone
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Image dataset (reads paths from CSV or raw folder)
# ──────────────────────────────────────────────────────────────────────────────

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImagePathDataset(Dataset):
    """Simple dataset that loads images from (filepath, label) pairs."""

    def __init__(self, filepaths: List[str], labels: List[int]):
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        img = _TRANSFORM(img)
        return img, self.labels[idx], idx          # return idx for tracking


def collect_image_paths() -> Tuple[List[str], List[int]]:
    """
    Collect all StateFarm image paths and labels.
    Tries multiple data sources in priority order.
    """
    filepaths: List[str] = []
    labels: List[int] = []

    # ── Option 1: flat CSV ──────────────────────────────────────────────
    for csv_path in [
        Path("data/processed/statefarm/train_paths.csv"),
        Path("data/processed/statefarm/train/train_paths.csv"),
    ]:
        if csv_path.exists():
            print(f"  Reading paths from {csv_path}")
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                fp = str(row["filepath"])
                # Resolve relative paths
                p = Path(fp)
                if not p.is_absolute() and not p.exists():
                    # Try prefixing with project root
                    p = Path.cwd() / fp
                if p.exists():
                    filepaths.append(str(p))
                    labels.append(int(row["label"]))

    # Also grab val/test CSVs if they exist
    for csv_path in [
        Path("data/processed/statefarm/val/val_paths.csv"),
        Path("data/processed/statefarm/test/test_paths.csv"),
    ]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                fp = str(row["filepath"])
                p = Path(fp)
                if not p.is_absolute() and not p.exists():
                    p = Path.cwd() / fp
                if p.exists():
                    filepaths.append(str(p))
                    labels.append(int(row["label"]))

    if filepaths:
        print(f"  Collected {len(filepaths)} images from CSV(s)")
        return filepaths, labels

    # ── Option 2: raw ImageFolder ────────────────────────────────────────
    for raw_dir in [
        Path("data/raw/state_farm"),
        Path("data/raw/statefarm/imgs/train"),
        Path("data/raw/statefarm/train"),
    ]:
        if raw_dir.exists():
            print(f"  Scanning raw folder: {raw_dir}")
            class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
            for class_dir in class_dirs:
                # Class name like "c0", "c1", ...
                class_name = class_dir.name
                # Extract label integer from class name
                try:
                    label = int(class_name.replace("c", ""))
                except ValueError:
                    label = len(set(labels))  # fallback sequential numbering

                for img_path in sorted(class_dir.glob("*")):
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        filepaths.append(str(img_path))
                        labels.append(label)

            if filepaths:
                print(f"  Collected {len(filepaths)} images from {raw_dir}")
                return filepaths, labels

    raise FileNotFoundError(
        "No StateFarm images found. Expected one of:\n"
        "  data/processed/statefarm/train_paths.csv\n"
        "  data/raw/state_farm/c*/*.jpg\n"
        "  data/raw/statefarm/imgs/train/c*/*.jpg"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Backbone loader
# ──────────────────────────────────────────────────────────────────────────────

def load_backbone(name: str) -> Tuple[nn.Module, int]:
    """Load a pretrained backbone with classification head removed."""
    try:
        import timm
        model = timm.create_model(name, pretrained=True, num_classes=0)
        feat_dim = model.num_features
        print(f"  Backbone: {name}  feature_dim={feat_dim}")
        return model, feat_dim
    except Exception:
        pass

    # Fallback: torchvision
    import torchvision.models as tvm
    weights_map = {
        "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT, 512),
        "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.DEFAULT, 512),
        "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT, 2048),
    }
    if name in weights_map:
        fn, weights, dim = weights_map[name]
        model = fn(weights=weights)
        model.fc = nn.Identity()  # remove classification head
        print(f"  Backbone (torchvision): {name}  feature_dim={dim}")
        return model, dim

    raise RuntimeError(f"Could not load backbone '{name}' via timm or torchvision.")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Extraction loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    feat_dim: int,
    total: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on all images and collect feature vectors."""
    model.eval()
    all_features = np.zeros((total, feat_dim), dtype=np.float32)
    all_labels = np.zeros(total, dtype=np.int64)

    done = 0
    for images, labels, indices in tqdm(loader, desc="Extracting features"):
        features = model(images)                   # (B, feat_dim)
        features = features.cpu().numpy()

        # L2 normalise each feature vector
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        features = features / norms

        batch_size = features.shape[0]
        all_features[done : done + batch_size] = features
        all_labels[done : done + batch_size] = labels.numpy()
        done += batch_size

    return all_features[:done], all_labels[:done]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract image features from StateFarm using a pretrained backbone"
    )
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="Backbone model name (default: resnet18)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for extraction (default: 32)")
    parser.add_argument("--output", type=str,
                        default="data/processed/statefarm/image_features.pkl",
                        help="Output pickle file path")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  IDBS — Offline Image Feature Extraction")
    print(f"  backbone={args.backbone}  batch_size={args.batch_size}")
    print(f"{'='*60}\n")

    # Collect image paths
    print("Collecting image paths…")
    filepaths, labels = collect_image_paths()
    n_classes = len(set(labels))
    print(f"  Total images: {len(filepaths)}  Classes: {n_classes}")
    for cls in sorted(set(labels)):
        count = sum(1 for l in labels if l == cls)
        print(f"    class {cls}: {count} images")

    # Load backbone
    print("\nLoading backbone…")
    model, feat_dim = load_backbone(args.backbone)
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataset and loader
    dataset = ImagePathDataset(filepaths, labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,         # 0 for Windows compatibility
        pin_memory=False,
    )

    # Extract features
    print(f"\nExtracting {feat_dim}-d features from {len(dataset)} images…")
    t0 = time.time()
    features, label_arr = extract_features(model, loader, feat_dim, len(dataset))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/len(dataset)*1000:.1f} ms/image)")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "features": features,
        "labels": label_arr,
        "filepaths": filepaths,
        "backbone": args.backbone,
        "feat_dim": feat_dim,
        "n_classes": n_classes,
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Shape: features={features.shape}  labels={label_arr.shape}")
    print(f"\n✓ Feature extraction complete!")


if __name__ == "__main__":
    main()
