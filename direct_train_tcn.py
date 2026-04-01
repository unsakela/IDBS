"""
Direct TCN Training Script for IDBS  (Optimised)
===================================================
Trains a Temporal Convolutional Network on ALL datasets:

  Sensor datasets:
    - Mendeley Driving  — data/processed/mendeley_driving/processed.pkl/.csv
    - Mendeley Risky    — data/processed/mendeley_risky/processed.pkl/.csv
    - GitHub Driving    — data/processed/github_driving/processed.pkl/.csv

  Image features (auxiliary — from pre-extracted ResNet18 features):
    - StateFarm         — data/processed/statefarm/image_features.pkl
                          512-d feature vectors extracted offline.
                          If not found, runs without image data.

Input shape to TCN: (B, 1, F) where F = padded feature dimension.
Labels are dataset-tagged and globally re-encoded (e.g. statefarm_c0, mendeley_risky_1).

Optimisation features:
  ✓ Gradient accumulation (configurable, default 4 steps)
  ✓ Label smoothing (0.1)
  ✓ LR warmup (3 epochs) + cosine annealing
  ✓ Class-weighted loss (inverse frequency)
  ✓ CPU AMP (bfloat16 on supported hardware)
  ✓ Early stopping (patience 7)
  ✓ Gradient clipping (max norm 1.0)
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.sensor.tcn import TCN
from utils.device import clear_cache, get_batch_size, get_device


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Sensor loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_pkl(path: Path, tag: str) -> Tuple[torch.Tensor, np.ndarray]:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    feats = torch.FloatTensor(np.array(d["features"], dtype=np.float32))
    if feats.ndim == 3:
        feats = feats.reshape(feats.shape[0], -1)
    labels = np.array(d["labels"], dtype=np.int64).ravel()
    return feats, np.array([f"{tag}_{l}" for l in labels])


def _load_csv_sensor(path: Path, tag: str) -> Tuple[torch.Tensor, np.ndarray]:
    df = pd.read_csv(path)
    label_col = next(
        (c for c in ["label", "Label", "class", "Class", "target", "Target"]
         if c in df.columns), None,
    )
    if label_col is None:
        raise ValueError(f"No label column in {path}. Columns: {df.columns.tolist()}")
    raw = LabelEncoder().fit_transform(df[label_col].values)
    feat_cols = [c for c in df.columns
                 if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    feats = df[feat_cols].values.astype(np.float32)
    feats = (feats - feats.mean(0, keepdims=True)) / (feats.std(0, keepdims=True) + 1e-8)
    return torch.FloatTensor(feats), np.array([f"{tag}_{l}" for l in raw])


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Image feature loading (pre-extracted)
# ──────────────────────────────────────────────────────────────────────────────

def _load_image_features(path: Path, tag: str = "statefarm") -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    """Load pre-extracted image features from extract_image_features.py output."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        feats = torch.FloatTensor(np.array(d["features"], dtype=np.float32))
        labels = np.array(d["labels"], dtype=np.int64).ravel()
        str_labels = np.array([f"{tag}_{l}" for l in labels])
        print(f"  {tag} image features: {feats.shape}  ({len(set(labels))} classes)")
        return feats, str_labels
    except Exception as e:
        print(f"  {tag} image features: FAILED — {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Master loader — merges ALL datasets
# ──────────────────────────────────────────────────────────────────────────────

def load_all_data(include_images: bool = True) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load Mendeley Driving + Mendeley Risky + GitHub Driving sensor data,
    plus optional pre-extracted image features from StateFarm.
    Returns: features (N, F), labels (N,), num_classes
    """
    data_dir = Path("data/processed")
    all_feats:      List[torch.Tensor] = []
    all_str_labels: List[np.ndarray]   = []

    # ── Sensor sources ─────────────────────────────────────────────────
    sensor_sources = [
        ("mendeley_driving", data_dir / "mendeley_driving" / "processed.pkl", "pkl"),
        ("mendeley_driving", data_dir / "mendeley_driving" / "processed.csv", "csv"),
        ("mendeley_risky",   data_dir / "mendeley_risky"   / "processed.pkl", "pkl"),
        ("mendeley_risky",   data_dir / "mendeley_risky"   / "processed.csv", "csv"),
        ("github_driving",   data_dir / "github_driving"   / "processed.pkl", "pkl"),
        ("github_driving",   data_dir / "github_driving"   / "processed.csv", "csv"),
    ]
    seen: set = set()
    for tag, path, fmt in sensor_sources:
        if tag in seen or not path.exists():
            continue
        seen.add(tag)
        try:
            if fmt == "pkl":
                feats, str_labels = _load_pkl(path, tag)
            else:
                feats, str_labels = _load_csv_sensor(path, tag)
            all_feats.append(feats)
            all_str_labels.append(str_labels)
            print(f"  {tag}: {feats.shape}")
        except Exception as e:
            print(f"  {tag}: FAILED — {e}")

    # ── Image features (auxiliary) ─────────────────────────────────────
    if include_images:
        img_result = _load_image_features(
            data_dir / "statefarm" / "image_features.pkl", "statefarm"
        )
        if img_result is not None:
            all_feats.append(img_result[0])
            all_str_labels.append(img_result[1])

    if not all_feats:
        raise FileNotFoundError(
            "No data found. Run preprocessing scripts first:\n"
            "  python data/preprocessing/mendeley_driving_preprocess.py\n"
            "  python data/preprocessing/mendeley_risky_preprocess.py\n"
            "  python data/preprocessing/github_driving_preprocess.py\n"
            "  python extract_image_features.py  (optional)"
        )

    # ── Zero-pad to common width ────────────────────────────────────────
    max_f = max(f.shape[1] for f in all_feats)
    aligned = []
    for f in all_feats:
        if f.shape[1] < max_f:
            f = torch.cat([f, torch.zeros(f.shape[0], max_f - f.shape[1])], dim=1)
        aligned.append(f)

    combined_feats  = torch.vstack(aligned)
    combined_strlbl = np.concatenate(all_str_labels)

    # Global normalize + clean
    mean = combined_feats.mean(0, keepdim=True)
    std  = combined_feats.std(0, keepdim=True) + 1e-8
    combined_feats = (combined_feats - mean) / std
    combined_feats = torch.nan_to_num(combined_feats, nan=0.0, posinf=0.0, neginf=0.0)

    le              = LabelEncoder()
    combined_labels = torch.LongTensor(le.fit_transform(combined_strlbl))
    num_classes     = int(combined_labels.max().item()) + 1

    print(f"\nAll data combined:")
    print(f"  features  : {combined_feats.shape}")
    print(f"  labels    : {combined_labels.shape}  ({num_classes} classes)")
    for cls in range(num_classes):
        count = (combined_labels == cls).sum().item()
        name  = le.classes_[cls]
        print(f"  class {cls:2d} ({name}): {count} samples")

    return combined_feats, combined_labels, num_classes


# ──────────────────────────────────────────────────────────────────────────────
# 4.  DataLoaders
# ──────────────────────────────────────────────────────────────────────────────

def create_data_loaders(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    N   = len(features)
    idx = torch.randperm(N, generator=torch.Generator().manual_seed(42)).numpy()

    n_train   = max(1, int(0.70 * N))
    n_val     = max(1, int(0.15 * N))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    def _make(subset_idx, shuffle: bool) -> DataLoader:
        if len(subset_idx) == 0:
            ds = TensorDataset(
                torch.empty(0, features.shape[1]),
                torch.empty(0, dtype=torch.long),
            )
            return DataLoader(ds, batch_size=batch_size)
        ds = TensorDataset(features[subset_idx], labels[subset_idx])
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers,
            drop_last=(shuffle and len(subset_idx) >= batch_size),
        )

    loaders = _make(train_idx, True), _make(val_idx, False), _make(test_idx, False)
    print(f"Splits — train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    return loaders


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Class weights
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, capped at 10×."""
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes       # normalise to mean=1
    weights = weights.clamp(max=10.0)                     # cap extreme weights
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# 6.  LR scheduler with warmup
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup → cosine annealing."""
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=max(1, warmup_epochs))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ──────────────────────────────────────────────────────────────────────────────
# 7.  AMP helper
# ──────────────────────────────────────────────────────────────────────────────

def get_amp_context(device: torch.device):
    """Return an autocast context manager appropriate for the device."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    # CPU bfloat16 — available on modern CPUs, graceful fallback otherwise
    try:
        return torch.amp.autocast("cpu", dtype=torch.bfloat16)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Train / validate
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module, loader: DataLoader,
    optimizer: optim.Optimizer, criterion: nn.Module,
    device: torch.device, clip: float = 1.0,
    accum_steps: int = 4,
) -> Tuple[float, float]:
    model.train()
    if len(loader) == 0:
        return 0.0, 0.0

    amp_ctx = get_amp_context(device)
    total_loss, preds, gts = 0.0, [], []
    optimizer.zero_grad()

    for step, (feats, labels) in enumerate(tqdm(loader, desc="Training", leave=False)):
        feats  = feats.unsqueeze(1).to(device)   # (B, 1, F)
        labels = labels.to(device)

        with amp_ctx:
            out  = model(feats)
            loss = criterion(out, labels) / accum_steps

        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(labels.cpu().numpy())

    acc = accuracy_score(gts, preds) if gts else 0.0
    return total_loss / len(loader), acc


def validate_epoch(
    model: nn.Module, loader: DataLoader,
    criterion: nn.Module, device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    if len(loader) == 0:
        return 0.0, 0.0, 0.0

    amp_ctx = get_amp_context(device)
    total_loss, preds, gts = 0.0, [], []

    with torch.no_grad():
        for feats, labels in tqdm(loader, desc="Validation", leave=False):
            feats  = feats.unsqueeze(1).to(device)
            labels = labels.to(device)
            with amp_ctx:
                out = model(feats)
                total_loss += criterion(out, labels).item()
            preds.extend(out.argmax(1).cpu().numpy())
            gts.extend(labels.cpu().numpy())

    acc = accuracy_score(gts, preds) if gts else 0.0
    f1  = f1_score(gts, preds, average="weighted", zero_division=0) if gts else 0.0
    return total_loss / len(loader), acc, f1


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TCN on all IDBS datasets (optimised)")
    parser.add_argument("--epochs",           type=int,   default=4)
    parser.add_argument("--batch_size",       type=int,   default=32)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--accum_steps",      type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × accum_steps)")
    parser.add_argument("--warmup_epochs",    type=int,   default=3)
    parser.add_argument("--label_smoothing",  type=float, default=0.1)
    parser.add_argument("--patience",         type=int,   default=7,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--include_images",   action="store_true", default=True,
                        help="Include pre-extracted image features as auxiliary data")
    parser.add_argument("--no_images",        action="store_true", default=False,
                        help="Exclude image features (sensor only)")
    parser.add_argument("--checkpoint_dir",   type=str,   default="checkpoints")
    args = parser.parse_args()

    include_images = args.include_images and not args.no_images

    device      = get_device()
    batch_size  = get_batch_size(args.batch_size)
    num_workers = 0   # always 0 — prevents Windows multiprocessing deadlocks

    print(f"Device={device}  batch_size={batch_size}  accum={args.accum_steps}  "
          f"effective_batch={batch_size * args.accum_steps}  workers={num_workers}\n")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Load data (sensor + optional image features)
    features, labels, num_classes = load_all_data(include_images=include_images)

    print(f"\nFeature tensor : {features.shape}  dtype={features.dtype}")
    print(f"Label tensor   : {labels.shape}    dtype={labels.dtype}")
    print(f"Has nan: {features.isnan().any().item()}  "
          f"Has inf: {features.isinf().any().item()}")

    train_loader, val_loader, test_loader = create_data_loaders(
        features, labels, batch_size, num_workers
    )

    sample_x, sample_y = next(iter(train_loader))
    print(f"\nFirst batch — x: {sample_x.unsqueeze(1).shape}  "
          f"y: {sample_y.shape}  y_unique: {sample_y.unique().tolist()}\n")

    input_size = features.shape[-1]
    model = TCN(
        input_size       = input_size,
        output_size      = num_classes,
        num_channels     = [64, 128, 256, 512],
        kernel_size      = 3,
        dropout          = 0.2,
        return_sequences = False,
    ).to(device)

    # Class-weighted loss with label smoothing
    class_weights = compute_class_weights(labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    print(f"TCN params  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"input_size  : {input_size}")
    print(f"num_classes : {num_classes}")
    print(f"class_weights: {class_weights.cpu().tolist()}")
    print(f"label_smoothing: {args.label_smoothing}")
    print(f"warmup: {args.warmup_epochs} epochs  patience: {args.patience}\n")

    best_f1, best_epoch = 0.0, 0
    patience_counter = 0

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            accum_steps=args.accum_steps,
        )
        va_loss, va_acc, va_f1 = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val   loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | "
              f"lr={lr_now:.6f}")

        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch + 1
            patience_counter = 0
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":               va_f1,
                "val_acc":              va_acc,
                "num_classes":          num_classes,
                "input_size":           input_size,
            }, checkpoint_dir / "tcn_best.pt")
        else:
            patience_counter += 1

        clear_cache()

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest epoch {best_epoch} — val F1 {best_f1:.4f}")

    if best_f1 == 0.0:
        print("No valid checkpoint; skipping test evaluation.")
        return

    ckpt = torch.load(checkpoint_dir / "tcn_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te_loss, te_acc, te_f1 = validate_epoch(model, test_loader, criterion, device)
    print(f"Test — loss={te_loss:.4f}  acc={te_acc:.4f}  f1={te_f1:.4f}")


if __name__ == "__main__":
    main()