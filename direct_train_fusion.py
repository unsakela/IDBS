"""
Fusion Training Script for IDBS  (Optimised)
==============================================
Late-fusion model:  TCN (sensor) + Image features → fusion head.

Two modes of operation:
  1. PRE-EXTRACTED features (fast, CPU-friendly) — default
     Image branch uses pre-extracted 512-d ResNet18 features.
     No CNN backbone loaded at all — just a projection MLP.

  2. LIVE image processing (slow, needs GPU ideally) — fallback
     Loads raw StateFarm images, runs ResNet/EfficientNet per batch.
     Activated automatically if image_features.pkl is not found.

Sensor branch : Mendeley Driving + Mendeley Risky + GitHub Driving (no WESAD)
Vision branch : StateFarm 10-class distracted driver images

FusionDataset pairs each sensor window with a randomly sampled image/feature
(unaligned pairing — replace with aligned data once available).

Optimisation features:
  ✓ Gradient accumulation (configurable, default 4 steps)
  ✓ Label smoothing (0.1)
  ✓ LR warmup (3 epochs) + cosine annealing
  ✓ Class-weighted loss (inverse frequency)
  ✓ CPU AMP (bfloat16 on supported hardware)
  ✓ Early stopping (patience 7)
  ✓ Gradient clipping (max norm 1.0)

Usage
-----
  python direct_train_fusion.py --tcn_ckpt checkpoints/tcn_best.pt
  python direct_train_fusion.py   # trains from scratch if no checkpoints
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.sensor.tcn import TCN
from utils.device import clear_cache, get_batch_size, get_device


# ──────────────────────────────────────────────────────────────────────────────
# Transforms (only used in live-image fallback mode)
# ──────────────────────────────────────────────────────────────────────────────

_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
])
_VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), transforms.Normalize(_MEAN, _STD),
])


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Sensor loading (no WESAD)
# ──────────────────────────────────────────────────────────────────────────────

def _load_pkl(path: Path, tag: str) -> Tuple[torch.Tensor, np.ndarray]:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    feats = torch.FloatTensor(np.array(d["features"], dtype=np.float32))
    if feats.ndim == 3:
        feats = feats.reshape(feats.shape[0], -1)
    labels = np.array(d["labels"], dtype=np.int64).ravel()
    return feats, np.array([f"{tag}_{l}" for l in labels])


def _load_csv(path: Path, tag: str) -> Tuple[torch.Tensor, np.ndarray]:
    df = pd.read_csv(path)
    lc = next((c for c in ["label", "Label", "class", "Class", "target", "Target"]
               if c in df.columns), None)
    if lc is None:
        raise ValueError(f"No label column in {path}")
    raw = LabelEncoder().fit_transform(df[lc].values)
    fc  = [c for c in df.columns if c != lc and pd.api.types.is_numeric_dtype(df[c])]
    f   = df[fc].values.astype(np.float32)
    f   = (f - f.mean(0, keepdims=True)) / (f.std(0, keepdims=True) + 1e-8)
    return torch.FloatTensor(f), np.array([f"{tag}_{l}" for l in raw])


def load_all_sensor() -> Tuple[torch.Tensor, torch.Tensor, int]:
    data_dir = Path("data/processed")
    sources  = [
        ("mendeley_driving", data_dir / "mendeley_driving/processed.pkl", "pkl"),
        ("mendeley_driving", data_dir / "mendeley_driving/processed.csv", "csv"),
        ("mendeley_risky",   data_dir / "mendeley_risky/processed.pkl",   "pkl"),
        ("mendeley_risky",   data_dir / "mendeley_risky/processed.csv",   "csv"),
        ("github_driving",   data_dir / "github_driving/processed.pkl",   "pkl"),
        ("github_driving",   data_dir / "github_driving/processed.csv",   "csv"),
    ]
    all_feats: List[torch.Tensor] = []
    all_strlbl: List[np.ndarray]  = []
    seen: set = set()

    for tag, path, fmt in sources:
        if tag in seen or not path.exists():
            continue
        seen.add(tag)
        try:
            f, sl = _load_pkl(path, tag) if fmt == "pkl" else _load_csv(path, tag)
            all_feats.append(f); all_strlbl.append(sl)
            print(f"  {tag}: {f.shape}")
        except Exception as e:
            print(f"  {tag}: failed — {e}")

    if not all_feats:
        raise FileNotFoundError("No sensor data found. Run preprocessing first.")

    max_f = max(f.shape[1] for f in all_feats)
    aligned = [torch.cat([f, torch.zeros(f.shape[0], max_f - f.shape[1])], 1)
               if f.shape[1] < max_f else f for f in all_feats]

    feats  = torch.nan_to_num(torch.vstack(aligned), nan=0.0, posinf=0.0, neginf=0.0)
    strlbl = np.concatenate(all_strlbl)
    le     = LabelEncoder()
    labels = torch.LongTensor(le.fit_transform(strlbl))
    n_cls  = int(labels.max().item()) + 1
    print(f"  Sensor combined: {feats.shape}  ({n_cls} classes)")
    return feats, labels, n_cls


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# 2a.  Pre-extracted image features
# ──────────────────────────────────────────────────────────────────────────────

def load_preextracted_features() -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]:
    """Returns (features, labels, n_classes, feat_dim) or None."""
    feat_path = Path("data/processed/statefarm/image_features.pkl")
    if not feat_path.exists():
        return None
    with open(feat_path, "rb") as fh:
        d = pickle.load(fh)
    features = torch.FloatTensor(np.array(d["features"], dtype=np.float32))
    labels   = torch.LongTensor(np.array(d["labels"], dtype=np.int64))
    feat_dim = int(features.shape[1])
    n_cls    = int(labels.max().item()) + 1
    print(f"  Image features: {features.shape}  ({n_cls} classes, backbone={d.get('backbone', '?')})")
    return features, labels, n_cls, feat_dim


# ──────────────────────────────────────────────────────────────────────────────
# 2b.  StateFarm image loading (live fallback)
# ──────────────────────────────────────────────────────────────────────────────

class _ImgDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tf):
        self.df, self.tf = df.reset_index(drop=True), tf
    def __len__(self):  return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        return self.tf(Image.open(row["filepath"]).convert("RGB")), int(row["label"])


class _SubsetImg(Dataset):
    def __init__(self, base, idxs, tf):
        self.base, self.idxs, self.tf = base, list(idxs), tf
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        p, l = self.base.samples[self.idxs[i]]
        return self.tf(Image.open(p).convert("RGB")), l


def load_statefarm() -> Tuple[Dataset, Dataset, Dataset, int]:
    flat = Path("data/processed/statefarm/train_paths.csv")
    if flat.exists():
        df = pd.read_csv(flat); n = int(df["label"].nunique())
        train_df, tmp = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"].tolist())
        val_df, test_df = train_test_split(tmp, test_size=0.50, random_state=42, stratify=tmp["label"].tolist())
        print(f"  StateFarm: train={len(train_df)} val={len(val_df)} test={len(test_df)} classes={n}")
        return _ImgDataset(train_df, _TRAIN_TF), _ImgDataset(val_df, _VAL_TF), _ImgDataset(test_df, _VAL_TF), n

    d = Path("data/processed/statefarm")
    tc, vc, xc = d / "train/train_paths.csv", d / "val/val_paths.csv", d / "test/test_paths.csv"
    if tc.exists() and vc.exists():
        tdf, vdf = pd.read_csv(tc), pd.read_csv(vc)
        xdf = pd.read_csv(xc) if xc.exists() else vdf.iloc[:0]
        n = int(tdf["label"].nunique())
        if len(tdf) < len(vdf): tdf, vdf = vdf, tdf
        return _ImgDataset(tdf, _TRAIN_TF), _ImgDataset(vdf, _VAL_TF), _ImgDataset(xdf, _VAL_TF), n

    for raw in [Path("data/raw/statefarm/imgs/train"),
                Path("data/raw/state_farm/imgs/train"),
                Path("data/raw/statefarm/train")]:
        if raw.exists():
            import torchvision.datasets as tvds
            full = tvds.ImageFolder(raw)
            ti, tmp = train_test_split(range(len(full)), test_size=0.20, random_state=42, stratify=full.targets)
            vi, xi  = train_test_split(tmp, test_size=0.50, random_state=42, stratify=[full.targets[i] for i in tmp])
            n = len(full.classes)
            return _SubsetImg(full, ti, _TRAIN_TF), _SubsetImg(full, vi, _VAL_TF), _SubsetImg(full, xi, _VAL_TF), n

    raise FileNotFoundError("No StateFarm data found.")


# ══════════════════════════════════════════════════════════════════════════════
#  FUSION DATASETS
# ══════════════════════════════════════════════════════════════════════════════

class FeatureFusionDataset(Dataset):
    """Pairs sensor windows with pre-extracted image features."""

    def __init__(self, sensor_feats, sensor_labels, img_features, img_labels):
        self.sf = sensor_feats
        self.sl = sensor_labels
        self.if_ = img_features
        self.il  = img_labels
        self.n   = len(sensor_feats)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s_feat  = self.sf[idx % len(self.sf)]
        s_label = self.sl[idx % len(self.sf)]
        img_idx = np.random.randint(0, len(self.if_))
        img_feat = self.if_[img_idx]
        return s_feat, img_feat, s_label


class LiveFusionDataset(Dataset):
    """Pairs sensor windows with randomly sampled StateFarm images (live)."""

    def __init__(self, sensor_feats, sensor_labels, img_dataset):
        self.sf   = sensor_feats
        self.sl   = sensor_labels
        self.imgs = img_dataset
        self.n    = min(len(sensor_feats), len(img_dataset))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s_feat  = self.sf[idx % len(self.sf)]
        s_label = self.sl[idx % len(self.sf)]
        img_idx = np.random.randint(0, len(self.imgs))
        image, _ = self.imgs[img_idx]
        return s_feat, image, s_label


# ══════════════════════════════════════════════════════════════════════════════
#  MODELS
# ══════════════════════════════════════════════════════════════════════════════

class _Proj(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o), nn.GELU(), nn.Dropout(0.2))
    def forward(self, x): return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Feature-based fusion model (no CNN — fast)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureFusionModel(nn.Module):
    """Fusion model using pre-extracted image features + TCN for sensor."""

    def __init__(self, sensor_input_size, num_classes, image_feat_dim, embed_dim=256):
        super().__init__()
        self.tcn = TCN(
            input_size=sensor_input_size, output_size=num_classes,
            num_channels=[64, 128, 256, 512], kernel_size=3, dropout=0.2,
            return_sequences=False,
        )
        self.tcn_proj = _Proj(512, embed_dim)
        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(0.1),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(embed_dim, num_classes),
        )

    def forward(self, sensor, img_feat):
        s_emb = self.tcn_proj(self.tcn(sensor))
        v_emb = self.image_proj(img_feat)
        return self.fusion_head(torch.cat([s_emb, v_emb], dim=1))

    def load_pretrained_tcn(self, tcn_ckpt, device):
        if tcn_ckpt and Path(tcn_ckpt).exists() and Path(tcn_ckpt).stat().st_size > 0:
            try:
                ckpt = torch.load(tcn_ckpt, map_location=device)
                miss, unex = self.tcn.load_state_dict(ckpt["model_state_dict"], strict=False)
                print(f"  TCN ckpt loaded (missing={len(miss)}, unexpected={len(unex)})")
            except Exception as e:
                print(f"  TCN ckpt failed: {e} — training from scratch.")
        else:
            print("  TCN ckpt: not found — training from scratch.")


# ──────────────────────────────────────────────────────────────────────────────
# Live-image fusion model (CNN backbone — slow fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _load_efficientnet() -> Tuple[nn.Module, int]:
    try:
        import timm
    except ImportError:
        raise ImportError("pip install timm")
    for name in ["resnet18", "resnet34", "resnet50", "efficientnet_b0"]:
        try:
            m = timm.create_model(name, pretrained=False, num_classes=0)
            print(f"  Vision backbone: {name}  feature_dim={m.num_features}")
            return m, m.num_features
        except Exception:
            continue
    raise RuntimeError("No ResNet/EfficientNet available via timm.")


class LiveFusionModel(nn.Module):
    """Fusion model with CNN backbone for live image processing."""

    def __init__(self, sensor_input_size, num_sensor_classes, num_fusion_classes, embed_dim=256):
        super().__init__()
        self.tcn = TCN(
            input_size=sensor_input_size, output_size=num_sensor_classes,
            num_channels=[64, 128, 256, 512], kernel_size=3, dropout=0.2, return_sequences=False,
        )
        self.tcn_proj = _Proj(512, embed_dim)
        self.vision_backbone, vision_dim = _load_efficientnet()
        self.vision_proj = _Proj(vision_dim, embed_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(embed_dim, num_fusion_classes),
        )

    def forward(self, sensor, image):
        s_emb = self.tcn_proj(self.tcn(sensor))
        v_emb = self.vision_proj(self.vision_backbone(image))
        return self.fusion_head(torch.cat([s_emb, v_emb], dim=1))

    def load_pretrained_branches(self, tcn_ckpt, vision_ckpt, device):
        if tcn_ckpt and Path(tcn_ckpt).exists() and Path(tcn_ckpt).stat().st_size > 0:
            try:
                ckpt = torch.load(tcn_ckpt, map_location=device)
                miss, unex = self.tcn.load_state_dict(ckpt["model_state_dict"], strict=False)
                print(f"  TCN ckpt loaded (missing={len(miss)}, unexpected={len(unex)})")
            except Exception as e:
                print(f"  TCN ckpt failed: {e} — training from scratch.")
        else:
            print("  TCN ckpt: not found — training from scratch.")

        if vision_ckpt and Path(vision_ckpt).exists() and Path(vision_ckpt).stat().st_size > 0:
            try:
                ckpt   = torch.load(vision_ckpt, map_location=device)
                prefix = "vision_backbone."
                state  = {k[len(prefix):]: v for k, v in ckpt["model_state_dict"].items()
                          if k.startswith(prefix)}
                miss, unex = self.vision_backbone.load_state_dict(state, strict=False)
                print(f"  Vision ckpt loaded (missing={len(miss)}, unexpected={len(unex)})")
            except Exception as e:
                print(f"  Vision ckpt failed: {e} — training from scratch.")
        else:
            print("  Vision ckpt: not found — training from scratch.")


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return weights.clamp(max=10.0)


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=max(1, warmup_epochs))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def get_amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    try:
        return torch.amp.autocast("cpu", dtype=torch.bfloat16)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def train_fusion_epoch(model, loader, optimizer, criterion, device,
                       accum_steps=4, mode="feature"):
    model.train()
    amp_ctx = get_amp_context(device)
    total_loss, preds, gts = 0.0, [], []
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        t0 = time.time()
        s_feats = batch[0].unsqueeze(1).to(device)
        labels  = batch[2].to(device)

        if mode == "feature":
            img_input = batch[1].to(device)       # (B, feat_dim) — feature vector
        else:
            img_input = batch[1].to(device)       # (B, 3, 224, 224) — raw image

        with amp_ctx:
            out  = model(s_feats, img_input)
            loss = criterion(out, labels) / accum_steps

        loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        preds.extend(out.argmax(1).cpu().numpy())
        gts.extend(labels.cpu().numpy())

        if time.time() - t0 > 60:
            print(f"  WARNING: batch {i} took >{60}s")

    acc = accuracy_score(gts, preds) if gts else 0.0
    return total_loss / max(len(loader), 1), acc


def validate_fusion_epoch(model, loader, criterion, device, mode="feature"):
    model.eval()
    amp_ctx = get_amp_context(device)
    total_loss, preds, gts = 0.0, [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            s_feats = batch[0].unsqueeze(1).to(device)
            labels  = batch[2].to(device)

            if mode == "feature":
                img_input = batch[1].to(device)
            else:
                img_input = batch[1].to(device)

            with amp_ctx:
                out = model(s_feats, img_input)
                total_loss += criterion(out, labels).item()
            preds.extend(out.argmax(1).cpu().numpy())
            gts.extend(labels.cpu().numpy())

    acc = accuracy_score(gts, preds) if gts else 0.0
    f1  = f1_score(gts, preds, average="weighted", zero_division=0) if gts else 0.0
    return total_loss / max(len(loader), 1), acc, f1


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _split(n, seed=42):
    rng = np.random.default_rng(seed); idx = rng.permutation(n)
    n_tr = int(0.70 * n); n_va = int(0.15 * n)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",         type=int,   default=4)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--embed_dim",      type=int,   default=256)
    parser.add_argument("--freeze_epochs",  type=int,   default=3)
    parser.add_argument("--accum_steps",    type=int,   default=4)
    parser.add_argument("--warmup_epochs",  type=int,   default=3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience",       type=int,   default=7)
    parser.add_argument("--tcn_ckpt",       type=str,   default="checkpoints/tcn_best.pt")
    parser.add_argument("--vision_ckpt",    type=str,   default="checkpoints/vision_best.pt")
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    args = parser.parse_args()

    device      = get_device()
    batch_size  = get_batch_size(args.batch_size)
    num_workers = 0
    pin_memory  = False

    print(f"\n{'='*60}")
    print(f"  IDBS Fusion Training (Optimised)")
    print(f"  device={device}  batch_size={batch_size}  accum={args.accum_steps}")
    print(f"{'='*60}\n")

    # Checkpoint status
    tcn_ok    = Path(args.tcn_ckpt).exists()    and Path(args.tcn_ckpt).stat().st_size    > 0
    vision_ok = Path(args.vision_ckpt).exists() and Path(args.vision_ckpt).stat().st_size > 0
    print(f"Checkpoints:")
    print(f"  TCN:    {'FOUND' if tcn_ok    else 'NOT FOUND — scratch'}")
    print(f"  Vision: {'FOUND' if vision_ok else 'NOT FOUND — scratch'}\n")

    print("Loading sensor datasets…")
    s_feats, s_labels, n_sensor_cls = load_all_sensor()

    # ── Try pre-extracted features first ─────────────────────────────────
    print("\nLoading image data…")
    preextracted = load_preextracted_features()

    if preextracted is not None:
        # ═══════ FEATURE MODE ═══════
        mode = "feature"
        img_features, img_labels, n_vision_cls, image_feat_dim = preextracted
        print(f"\n✓ Using PRE-EXTRACTED features mode (fast, CPU-friendly)")

        tr_idx, va_idx, te_idx = _split(len(s_feats))
        train_ds = FeatureFusionDataset(s_feats[tr_idx], s_labels[tr_idx], img_features, img_labels)
        val_ds   = FeatureFusionDataset(s_feats[va_idx], s_labels[va_idx], img_features, img_labels)
        test_ds  = FeatureFusionDataset(s_feats[te_idx], s_labels[te_idx], img_features, img_labels)

        sensor_input_size = int(s_feats.shape[1])
        model = FeatureFusionModel(
            sensor_input_size = sensor_input_size,
            num_classes       = n_sensor_cls,
            image_feat_dim    = image_feat_dim,
            embed_dim         = args.embed_dim,
        ).to(device)

        model.load_pretrained_tcn(args.tcn_ckpt, device)

    else:
        # ═══════ LIVE IMAGE MODE (fallback) ═══════
        mode = "live"
        print(f"\n⚠ Pre-extracted features not found — using LIVE image processing (slower)")
        print("  Hint: Run `python extract_image_features.py` first for faster training.\n")

        print("Loading StateFarm images…")
        train_img_ds, val_img_ds, test_img_ds, n_vision_cls = load_statefarm()

        tr_idx, va_idx, te_idx = _split(len(s_feats))
        train_ds = LiveFusionDataset(s_feats[tr_idx], s_labels[tr_idx], train_img_ds)
        val_ds   = LiveFusionDataset(s_feats[va_idx], s_labels[va_idx], val_img_ds)
        test_ds  = LiveFusionDataset(s_feats[te_idx], s_labels[te_idx], test_img_ds)

        sensor_input_size = int(s_feats.shape[1])
        image_feat_dim = 0  # not used
        model = LiveFusionModel(
            sensor_input_size  = sensor_input_size,
            num_sensor_classes = n_sensor_cls,
            num_fusion_classes = n_sensor_cls,
            embed_dim          = args.embed_dim,
        ).to(device)

        model.load_pretrained_branches(args.tcn_ckpt, args.vision_ckpt, device)

    print(f"\nFusion splits: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    def _dl(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    train_loader = _dl(train_ds, True)
    val_loader   = _dl(val_ds,   False)
    test_loader  = _dl(test_ds,  False)

    # Class-weighted loss
    class_weights = compute_class_weights(s_labels, n_sensor_cls).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    print(f"\nFusion params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"sensor_input={sensor_input_size}  n_sensor={n_sensor_cls}  n_vision={n_vision_cls}")
    print(f"Mode={mode}  label_smoothing={args.label_smoothing}")
    print(f"Warmup={args.warmup_epochs}  patience={args.patience}\n")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    best_f1, best_epoch = 0.0, 0
    patience_counter = 0
    backbone_frozen = False

    def _freeze(freeze):
        for p in model.tcn.parameters(): p.requires_grad = not freeze
        if mode == "live" and hasattr(model, "vision_backbone"):
            for p in model.vision_backbone.parameters(): p.requires_grad = not freeze

    for epoch in range(args.epochs):
        should_freeze = epoch < args.freeze_epochs
        if should_freeze and not backbone_frozen:
            _freeze(True);  backbone_frozen = True;  print("  Backbones frozen.")
        elif not should_freeze and backbone_frozen:
            _freeze(False); backbone_frozen = False; print("  Backbones unfrozen.")

        tr_loss, tr_acc = train_fusion_epoch(
            model, train_loader, optimizer, criterion, device,
            args.accum_steps, mode,
        )
        va_loss, va_acc, va_f1 = validate_fusion_epoch(
            model, val_loader, criterion, device, mode,
        )
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        status = f"({mode})" + (" frozen" if backbone_frozen else "")
        print(f"Epoch {epoch+1:3d} {status} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val   loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f} | "
              f"lr={lr_now:.6f}")

        if va_f1 > best_f1:
            best_f1, best_epoch = va_f1, epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": va_acc, "val_f1": va_f1,
                "sensor_input_size": sensor_input_size,
                "n_sensor_cls": n_sensor_cls, "n_vision_cls": n_vision_cls,
                "embed_dim": args.embed_dim,
                "mode": mode,
                "image_feat_dim": image_feat_dim if mode == "feature" else 0,
            }, checkpoint_dir / "fusion_best.pt")
        else:
            patience_counter += 1

        clear_cache()

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest epoch {best_epoch} — val F1 {best_f1:.4f}")
    ckpt_path = checkpoint_dir / "fusion_best.pt"
    if not ckpt_path.exists():
        print("No checkpoint saved."); return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te_loss, te_acc, te_f1 = validate_fusion_epoch(model, test_loader, criterion, device, mode)
    print(f"\nFusion Test — loss={te_loss:.4f}  acc={te_acc:.4f}  f1={te_f1:.4f}")


if __name__ == "__main__":
    main()