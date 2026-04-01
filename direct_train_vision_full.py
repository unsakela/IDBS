"""
Direct Vision Training Script for IDBS  (Optimised)
=====================================================
Primary task : StateFarm 10-class distracted driver detection
Auxiliary    : Sensor MLP branch regularises shared embedding
               (Mendeley Driving + Mendeley Risky + GitHub Driving)

Two modes of operation:
  1. PRE-EXTRACTED features (fast, CPU-friendly) — default
     Loads 512-d ResNet18 feature vectors from image_features.pkl
     Uses a lightweight MLP classifier instead of a full CNN backbone

  2. LIVE image processing (slow, needs GPU ideally) — fallback
     Loads raw images, runs ResNet/EfficientNet backbone per batch
     Activated automatically if image_features.pkl is not found

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
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from utils.device import clear_cache, get_batch_size, get_device


# ──────────────────────────────────────────────────────────────────────────────
# Transforms (only used in live-image fallback mode)
# ──────────────────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
_VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 1:  PRE-EXTRACTED FEATURES (fast, CPU-friendly)
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# 1a.  Load pre-extracted image features
# ──────────────────────────────────────────────────────────────────────────────

def load_preextracted_image_data() -> Optional[Tuple[torch.Tensor, torch.Tensor, int, int]]:
    """
    Load pre-extracted image features from image_features.pkl.
    Returns (features, labels, num_classes, feat_dim) or None if not found.
    """
    feat_path = Path("data/processed/statefarm/image_features.pkl")
    if not feat_path.exists():
        return None

    print("Loading pre-extracted image features…")
    with open(feat_path, "rb") as fh:
        d = pickle.load(fh)

    features = torch.FloatTensor(np.array(d["features"], dtype=np.float32))
    labels   = torch.LongTensor(np.array(d["labels"], dtype=np.int64))
    feat_dim = int(features.shape[1])
    n_cls    = int(labels.max().item()) + 1

    print(f"  Image features: {features.shape}  ({n_cls} classes)")
    print(f"  Backbone used: {d.get('backbone', 'unknown')}")
    for cls in range(n_cls):
        count = (labels == cls).sum().item()
        print(f"    class {cls}: {count} samples")

    return features, labels, n_cls, feat_dim


def split_preextracted(
    features: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split pre-extracted features into train/val/test (80/10/10)."""
    N = len(features)
    indices = np.arange(N)
    labels_np = labels.numpy()

    train_idx, tmp_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=labels_np
    )
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, random_state=42, stratify=labels_np[tmp_idx]
    )

    print(f"  Splits: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    return (features[train_idx], labels[train_idx],
            features[val_idx],   labels[val_idx],
            features[test_idx],  labels[test_idx])


# ──────────────────────────────────────────────────────────────────────────────
# 1b.  Sensor data (auxiliary)
# ──────────────────────────────────────────────────────────────────────────────

def _load_pkl_sensor(path: Path, tag: str) -> Tuple[torch.Tensor, np.ndarray]:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    feats = torch.FloatTensor(np.array(d["features"], dtype=np.float32))
    if feats.ndim == 3:
        feats = feats.reshape(feats.shape[0], -1)
    labels = np.array(d["labels"], dtype=np.int64).ravel()
    return feats, np.array([f"{tag}_{l}" for l in labels])


def _load_csv_sensor(path: Path, tag: str) -> Tuple[torch.Tensor, np.ndarray]:
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


def load_sensor_data() -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load Mendeley + GitHub sensor data. Returns (features, labels, n_classes)."""
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
            f, sl = _load_pkl_sensor(path, tag) if fmt == "pkl" else _load_csv_sensor(path, tag)
            all_feats.append(f)
            all_strlbl.append(sl)
            print(f"  {tag}: {f.shape}")
        except Exception as e:
            print(f"  {tag}: failed — {e}")

    if not all_feats:
        print("  No sensor data found — sensor branch disabled.")
        return torch.empty(0, 1), torch.empty(0, dtype=torch.long), 0

    max_f = max(f.shape[1] for f in all_feats)
    aligned = []
    for f in all_feats:
        if f.shape[1] < max_f:
            f = torch.cat([f, torch.zeros(f.shape[0], max_f - f.shape[1])], dim=1)
        aligned.append(f)

    feats  = torch.nan_to_num(torch.vstack(aligned), nan=0.0, posinf=0.0, neginf=0.0)
    strlbl = np.concatenate(all_strlbl)
    le     = LabelEncoder()
    labels = torch.LongTensor(le.fit_transform(strlbl))
    n_cls  = int(labels.max().item()) + 1
    print(f"  Sensor combined: {feats.shape}  ({n_cls} classes)")
    return feats, labels, n_cls


# ──────────────────────────────────────────────────────────────────────────────
# 1c.  Feature-based multimodal dataset (for pre-extracted mode)
# ──────────────────────────────────────────────────────────────────────────────

class FeatureMultiModalDataset(Dataset):
    """(image_feat, sensor_feat, vision_label, sensor_label, has_sensor)"""

    def __init__(
        self,
        img_features:    torch.Tensor,
        img_labels:      torch.Tensor,
        sensor_feats:    Optional[torch.Tensor],
        sensor_labels:   Optional[torch.Tensor],
        sensor_feat_dim: int,
    ):
        self.img_feats  = img_features
        self.img_labels = img_labels
        self.s_feats    = sensor_feats
        self.s_labels   = sensor_labels
        self.feat_dim   = sensor_feat_dim
        self.has_sensor = sensor_feats is not None and len(sensor_feats) > 0
        self.n_sensor   = len(sensor_feats) if self.has_sensor else 0

    def __len__(self):
        return len(self.img_feats)

    def __getitem__(self, idx):
        img_feat  = self.img_feats[idx]
        vis_label = int(self.img_labels[idx].item())
        if self.has_sensor:
            s_idx      = np.random.randint(0, self.n_sensor)
            s_feat     = self.s_feats[s_idx]
            s_label    = int(self.s_labels[s_idx].item())
            has_sensor = 1.0
        else:
            s_feat     = torch.zeros(self.feat_dim)
            s_label    = 0
            has_sensor = 0.0
        return img_feat, s_feat, vis_label, s_label, torch.tensor(has_sensor)


# ──────────────────────────────────────────────────────────────────────────────
# 1d.  Feature-based model (MLP instead of CNN)
# ──────────────────────────────────────────────────────────────────────────────

class SensorMLP(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(0.2),
        )
    def forward(self, x): return self.net(x)


class FeatureVisionSensorModel(nn.Module):
    """Vision+Sensor model using pre-extracted image features (no CNN backbone)."""

    def __init__(
        self,
        num_vision_classes: int,
        num_sensor_classes: int,
        image_feat_dim:     int,
        sensor_feat_dim:    int,
        embed_dim:          int = 256,
    ):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(image_feat_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(0.1),
        )
        self.sensor_mlp = SensorMLP(max(sensor_feat_dim, 1), embed_dim)
        self.vision_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(embed_dim, num_vision_classes),
        )
        self.sensor_head = (
            nn.Linear(embed_dim, num_sensor_classes) if num_sensor_classes > 0 else None
        )

    def forward(self, img_feat, sensor_feat):
        v_emb = self.image_proj(img_feat)
        s_emb = self.sensor_mlp(sensor_feat)
        vis_logits = self.vision_head(torch.cat([v_emb, s_emb], dim=1))
        sens_logits = self.sensor_head(s_emb) if self.sensor_head else None
        return vis_logits, sens_logits


# ══════════════════════════════════════════════════════════════════════════════
#  MODE 2:  LIVE IMAGE PROCESSING (fallback — slow on CPU)
# ══════════════════════════════════════════════════════════════════════════════

class StateFarmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


class _TransformSubset(Dataset):
    def __init__(self, base, indices, transform):
        self.base    = base
        self.indices = list(indices)
        self.tf      = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.base.samples[self.indices[idx]]
        return self.tf(Image.open(path).convert("RGB")), label


class LiveMultiModalDataset(Dataset):
    """(image, sensor_feat, vision_label, sensor_label, has_sensor)"""

    def __init__(self, img_dataset, sensor_feats, sensor_labels, sensor_feat_dim):
        self.img_ds     = img_dataset
        self.s_feats    = sensor_feats
        self.s_labels   = sensor_labels
        self.feat_dim   = sensor_feat_dim
        self.has_sensor = sensor_feats is not None and len(sensor_feats) > 0
        self.n_sensor   = len(sensor_feats) if self.has_sensor else 0

    def __len__(self):
        return len(self.img_ds)

    def __getitem__(self, idx):
        image, vis_label = self.img_ds[idx]
        if self.has_sensor:
            s_idx      = np.random.randint(0, self.n_sensor)
            s_feat     = self.s_feats[s_idx]
            s_label    = int(self.s_labels[s_idx].item())
            has_sensor = 1.0
        else:
            s_feat     = torch.zeros(self.feat_dim)
            s_label    = 0
            has_sensor = 0.0
        return image, s_feat, vis_label, s_label, torch.tensor(has_sensor)


def load_efficientnet(num_classes: int) -> Tuple[nn.Module, int]:
    """Try multiple ResNet variants. Returns (backbone_no_head, feat_dim)."""
    try:
        import timm
    except ImportError:
        raise ImportError("pip install timm")
    for name in ["resnet18", "resnet34", "resnet50", "efficientnet_b0"]:
        try:
            m = timm.create_model(name, pretrained=True, num_classes=0)
            print(f"  Vision backbone: {name}  feature_dim={m.num_features}")
            return m, m.num_features
        except Exception as e:
            print(f"  {name} failed: {e}")
    raise RuntimeError("Could not load any ResNet/EfficientNet model via timm.")


class LiveVisionSensorModel(nn.Module):
    """Full CNN-based model (fallback for when pre-extracted features are unavailable)."""

    def __init__(self, num_vision_classes, num_sensor_classes, sensor_feat_dim, embed_dim=256):
        super().__init__()
        self.vision_backbone, vision_dim = load_efficientnet(num_vision_classes)
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(0.2),
        )
        self.sensor_mlp = SensorMLP(max(sensor_feat_dim, 1), embed_dim)
        self.vision_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(embed_dim, num_vision_classes),
        )
        self.sensor_head = (
            nn.Linear(embed_dim, num_sensor_classes) if num_sensor_classes > 0 else None
        )

    def forward(self, images, sensor_feat):
        v_emb = self.vision_proj(self.vision_backbone(images))
        s_emb = self.sensor_mlp(sensor_feat)
        vis_logits = self.vision_head(torch.cat([v_emb, s_emb], dim=1))
        sens_logits = self.sensor_head(s_emb) if self.sensor_head else None
        return vis_logits, sens_logits


def load_statefarm_data():
    """Load StateFarm image datasets (live mode). Returns (train, val, test, n_cls)."""
    flat = Path("data/processed/statefarm/train_paths.csv")
    if flat.exists():
        df = pd.read_csv(flat)
        n  = int(df["label"].nunique())
        train_df, tmp = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"].tolist())
        val_df, test_df = train_test_split(tmp, test_size=0.50, random_state=42, stratify=tmp["label"].tolist())
        return (StateFarmDataset(train_df, _TRAIN_TF),
                StateFarmDataset(val_df, _VAL_TF),
                StateFarmDataset(test_df, _VAL_TF), n)

    d = Path("data/processed/statefarm")
    tc, vc, xc = d/"train/train_paths.csv", d/"val/val_paths.csv", d/"test/test_paths.csv"
    if tc.exists() and vc.exists():
        tdf, vdf = pd.read_csv(tc), pd.read_csv(vc)
        xdf = pd.read_csv(xc) if xc.exists() else vdf.iloc[:0]
        n   = int(tdf["label"].nunique())
        if len(tdf) < len(vdf): tdf, vdf = vdf, tdf
        return (StateFarmDataset(tdf, _TRAIN_TF),
                StateFarmDataset(vdf, _VAL_TF),
                StateFarmDataset(xdf, _VAL_TF), n)

    for raw in [Path("data/raw/statefarm/imgs/train"), Path("data/raw/state_farm/imgs/train"),
                Path("data/raw/statefarm/train")]:
        if raw.exists():
            import torchvision.datasets as tvds
            full = tvds.ImageFolder(raw)
            ti, tmp = train_test_split(range(len(full)), test_size=0.20, random_state=42, stratify=full.targets)
            vi, xi = train_test_split(tmp, test_size=0.50, random_state=42, stratify=[full.targets[i] for i in tmp])
            n = len(full.classes)
            return (_TransformSubset(full, ti, _TRAIN_TF),
                    _TransformSubset(full, vi, _VAL_TF),
                    _TransformSubset(full, xi, _VAL_TF), n)

    raise FileNotFoundError("No StateFarm data found.")


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, capped at 10×."""
    counts = torch.zeros(num_classes)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    weights = weights.clamp(max=10.0)
    return weights


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


# ──────────────────────────────────────────────────────────────────────────────
# Train / validate  (works for both modes — feature-based and live)
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device,
                sensor_weight=0.3, accum_steps=4, mode="feature"):
    model.train()
    amp_ctx = get_amp_context(device)
    total_loss, preds, gts = 0.0, [], []
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        if mode == "feature":
            img_feat, s_feats, vis_labels, s_labels, has_sensor = batch
            img_feat   = img_feat.to(device)
        else:
            images, s_feats, vis_labels, s_labels, has_sensor = batch
            images = images.to(device)

        s_feats    = s_feats.to(device)
        vis_labels = vis_labels.to(device)
        s_labels   = s_labels.to(device)
        has_sensor = has_sensor.to(device)

        with amp_ctx:
            if mode == "feature":
                vis_logits, sens_logits = model(img_feat, s_feats)
            else:
                vis_logits, sens_logits = model(images, s_feats)
            loss = criterion(vis_logits, vis_labels)
            if sens_logits is not None and has_sensor.sum() > 0:
                mask = has_sensor.bool()
                loss = loss + sensor_weight * criterion(sens_logits[mask], s_labels[mask])
            loss = loss / accum_steps

        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        preds.extend(vis_logits.argmax(1).cpu().numpy())
        gts.extend(vis_labels.cpu().numpy())

    acc = accuracy_score(gts, preds) if gts else 0.0
    return total_loss / max(len(loader), 1), acc


def validate_epoch(model, loader, criterion, device, mode="feature"):
    model.eval()
    amp_ctx = get_amp_context(device)
    total_loss, preds, gts = 0.0, [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            if mode == "feature":
                img_feat, s_feats, vis_labels, s_labels, has_sensor = batch
                img_feat = img_feat.to(device)
            else:
                images, s_feats, vis_labels, s_labels, has_sensor = batch
                images = images.to(device)

            s_feats    = s_feats.to(device)
            vis_labels = vis_labels.to(device)

            with amp_ctx:
                if mode == "feature":
                    vis_logits, _ = model(img_feat, s_feats)
                else:
                    vis_logits, _ = model(images, s_feats)

                total_loss += criterion(vis_logits, vis_labels).item()
            preds.extend(vis_logits.argmax(1).cpu().numpy())
            gts.extend(vis_labels.cpu().numpy())

    acc = accuracy_score(gts, preds) if gts else 0.0
    f1  = f1_score(gts, preds, average="weighted", zero_division=0) if gts else 0.0
    return total_loss / max(len(loader), 1), acc, f1


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",         type=int,   default=4)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--freeze_epochs",  type=int,   default=5)
    parser.add_argument("--embed_dim",      type=int,   default=256)
    parser.add_argument("--sensor_weight",  type=float, default=0.3)
    parser.add_argument("--accum_steps",    type=int,   default=4)
    parser.add_argument("--warmup_epochs",  type=int,   default=3)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience",       type=int,   default=7)
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    args = parser.parse_args()

    device      = get_device()
    batch_size  = get_batch_size(args.batch_size)
    num_workers = 0
    pin_memory  = device.type != "cpu"

    print(f"\n{'='*60}")
    print(f"  IDBS Vision Training (Optimised)")
    print(f"  device={device}  batch_size={batch_size}  accum={args.accum_steps}")
    print(f"{'='*60}\n")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # ── Try pre-extracted features first ─────────────────────────────────
    preextracted = load_preextracted_image_data()

    if preextracted is not None:
        # ═══════ FEATURE MODE ═══════
        mode = "feature"
        img_features, img_labels, num_vision_cls, image_feat_dim = preextracted
        print(f"\n✓ Using PRE-EXTRACTED features mode (fast, CPU-friendly)")

        tr_feat, tr_lbl, va_feat, va_lbl, te_feat, te_lbl = split_preextracted(img_features, img_labels)

        print("\nLoading sensor datasets (auxiliary)…")
        s_feats, s_labels, num_sensor_cls = load_sensor_data()
        sensor_feat_dim = int(s_feats.shape[1]) if s_feats.shape[0] > 0 else 1
        has_sensor_data = s_feats.shape[0] > 0

        def _make_loader(img_f, img_l, shuffle):
            ds = FeatureMultiModalDataset(
                img_features    = img_f,
                img_labels      = img_l,
                sensor_feats    = s_feats  if has_sensor_data else None,
                sensor_labels   = s_labels if has_sensor_data else None,
                sensor_feat_dim = sensor_feat_dim,
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory)

        train_loader = _make_loader(tr_feat, tr_lbl, True)
        val_loader   = _make_loader(va_feat, va_lbl, False)
        test_loader  = _make_loader(te_feat, te_lbl, False)

        # Class weights from training labels
        class_weights = compute_class_weights(tr_lbl, num_vision_cls).to(device)

        model = FeatureVisionSensorModel(
            num_vision_classes = num_vision_cls,
            num_sensor_classes = num_sensor_cls,
            image_feat_dim     = image_feat_dim,
            sensor_feat_dim    = sensor_feat_dim,
            embed_dim          = args.embed_dim,
        ).to(device)

    else:
        # ═══════ LIVE IMAGE MODE (fallback) ═══════
        mode = "live"
        print("\n⚠ Pre-extracted features not found — using LIVE image processing (slower)")
        print("  Hint: Run `python extract_image_features.py` first for faster training.\n")

        print("Loading State Farm images…")
        train_img_ds, val_img_ds, test_img_ds, num_vision_cls = load_statefarm_data()

        print("\nLoading sensor datasets (auxiliary)…")
        s_feats, s_labels, num_sensor_cls = load_sensor_data()
        sensor_feat_dim = int(s_feats.shape[1]) if s_feats.shape[0] > 0 else 1
        has_sensor_data = s_feats.shape[0] > 0

        def _make_loader_live(img_ds, shuffle):
            mm = LiveMultiModalDataset(
                img_dataset     = img_ds,
                sensor_feats    = s_feats  if has_sensor_data else None,
                sensor_labels   = s_labels if has_sensor_data else None,
                sensor_feat_dim = sensor_feat_dim,
            )
            return DataLoader(mm, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=pin_memory)

        train_loader = _make_loader_live(train_img_ds, True)
        val_loader   = _make_loader_live(val_img_ds,   False)
        test_loader  = _make_loader_live(test_img_ds,  False)

        # Approximate class weights from first loader pass
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch[2].numpy())
        all_labels_t = torch.LongTensor(all_labels)
        class_weights = compute_class_weights(all_labels_t, num_vision_cls).to(device)

        model = LiveVisionSensorModel(
            num_vision_classes = num_vision_cls,
            num_sensor_classes = num_sensor_cls,
            sensor_feat_dim    = sensor_feat_dim,
            embed_dim          = args.embed_dim,
        ).to(device)

    # ── Common training setup ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)

    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vision classes={num_vision_cls}  Sensor classes={num_sensor_cls}")
    print(f"Mode={mode}  label_smoothing={args.label_smoothing}")
    print(f"Warmup={args.warmup_epochs}  patience={args.patience}\n")

    best_f1, best_epoch = 0.0, 0
    patience_counter = 0

    # Backbone freezing (only meaningful in live mode)
    backbone_frozen = False
    if mode == "live" and hasattr(model, "vision_backbone"):
        def _set_backbone_grad(requires_grad):
            for p in model.vision_backbone.parameters():
                p.requires_grad = requires_grad

    for epoch in range(args.epochs):
        # Progressive unfreezing (live mode only)
        if mode == "live" and hasattr(model, "vision_backbone"):
            should_freeze = epoch < args.freeze_epochs
            if should_freeze and not backbone_frozen:
                _set_backbone_grad(False); backbone_frozen = True
                print("  Backbone frozen.")
            elif not should_freeze and backbone_frozen:
                _set_backbone_grad(True); backbone_frozen = False
                print("  Backbone unfrozen.")

        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            args.sensor_weight, args.accum_steps, mode,
        )
        va_loss, va_acc, va_f1 = validate_epoch(model, val_loader, criterion, device, mode)
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
                "epoch": epoch+1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": va_f1, "val_acc": va_acc,
                "num_vision_classes": num_vision_cls,
                "num_sensor_classes": num_sensor_cls,
                "sensor_feat_dim": sensor_feat_dim,
                "embed_dim": args.embed_dim,
                "mode": mode,
            }, checkpoint_dir / "vision_best.pt")
        else:
            patience_counter += 1

        clear_cache()

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    print(f"\nBest epoch {best_epoch} — val F1 {best_f1:.4f}")
    ckpt_path = checkpoint_dir / "vision_best.pt"
    if not ckpt_path.exists():
        print("No checkpoint saved."); return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    te_loss, te_acc, te_f1 = validate_epoch(model, test_loader, criterion, device, mode)
    print(f"Test — loss={te_loss:.4f}  acc={te_acc:.4f}  f1={te_f1:.4f}")


if __name__ == "__main__":
    main()