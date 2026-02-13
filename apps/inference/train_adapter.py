"""
HomeVision CLIP Adapter Training
Trains a linear probe on top of frozen OpenCLIP embeddings.

Usage (local or Colab):
  python train_adapter.py \
    --labels labels.json \
    --images-dir ./images \
    --output-weights adapter.pt \
    --output-meta adapter_meta.json \
    --epochs 30 \
    --lr 0.001 \
    --batch-size 32

labels.json format:
[
  { "filename": "img1.jpg", "roomType": "Kitchen", "amenities": [...], "features": [...] },
  ...
]

Produces:
  adapter.pt          - trained linear layer weights
  adapter_meta.json   - { "labels": [...], "embed_dim": 512, "type": "multilabel" }
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LabeledImageDataset(Dataset):
    """Dataset that returns (image_path, multi-hot label vector)."""

    def __init__(
        self,
        entries: list[dict],
        images_dir: Path,
        all_labels: list[str],
        preprocess,
    ):
        self.entries = entries
        self.images_dir = images_dir
        self.all_labels = all_labels
        self.label_to_idx = {l: i for i, l in enumerate(all_labels)}
        self.preprocess = preprocess

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = self.images_dir / entry["filename"]
        img = Image.open(img_path).convert("RGB")
        img_t = self.preprocess(img)

        # Multi-hot: roomType + amenities + features
        target = torch.zeros(len(self.all_labels), dtype=torch.float32)
        for label_field in ["roomType"]:
            val = entry.get(label_field, "")
            if isinstance(val, str) and val in self.label_to_idx:
                target[self.label_to_idx[val]] = 1.0
        for label_field in ["amenities", "features"]:
            for val in entry.get(label_field, []):
                if val in self.label_to_idx:
                    target[self.label_to_idx[val]] = 1.0
        return img_t, target


def collect_all_labels(entries: list[dict]) -> list[str]:
    """Gather unique labels across roomType, amenities, features."""
    labels = set()
    for e in entries:
        rt = e.get("roomType", "")
        if rt:
            labels.add(rt)
        for field in ["amenities", "features"]:
            for v in e.get(field, []):
                labels.add(v)
    return sorted(labels)


def split_data(entries: list[dict], test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(entries))
    split = int(len(entries) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train = [entries[i] for i in train_idx]
    test = [entries[i] for i in test_idx]
    return train, test


def main():
    parser = argparse.ArgumentParser(description="Train CLIP linear adapter")
    parser.add_argument("--labels", required=True, help="Path to labels.json")
    parser.add_argument("--images-dir", required=True, help="Directory with images")
    parser.add_argument("--output-weights", default="adapter.pt", help="Output weights file")
    parser.add_argument("--output-meta", default="adapter_meta.json", help="Output metadata file")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    # Load labels
    with open(args.labels) as f:
        entries = json.load(f)

    images_dir = Path(args.images_dir)

    # Filter to entries whose images exist
    valid = [e for e in entries if (images_dir / e["filename"]).is_file()]
    missing = [e["filename"] for e in entries if not (images_dir / e["filename"]).is_file()]
    print(f"Found {len(valid)}/{len(entries)} images with labels")
    if missing:
        print(f"Missing in {images_dir}: {missing[:15]}{' ...' if len(missing) > 15 else ''}")
    if len(valid) < 5:
        print("Not enough data to train. Need at least 5 labeled images.")
        sys.exit(1)

    all_labels = collect_all_labels(valid)
    num_classes = len(all_labels)
    print(f"Labels: {num_classes} unique classes")

    # Load CLIP model (frozen)
    print("Loading OpenCLIP ViT-B-32...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    clip_model = clip_model.to(DEVICE).eval()
    embed_dim = clip_model.visual.output_dim if hasattr(clip_model.visual, "output_dim") else 512

    # Split data
    train_entries, test_entries = split_data(valid, test_ratio=args.test_ratio)
    print(f"Train: {len(train_entries)}, Test: {len(test_entries)}")

    # Precompute embeddings (much faster than encoding per epoch)
    print("Computing image embeddings...")

    def encode_images(entries_list):
        ds = LabeledImageDataset(entries_list, images_dir, all_labels, preprocess)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        all_embs = []
        all_targets = []
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(DEVICE)
                embs = clip_model.encode_image(imgs)
                embs = embs / embs.norm(dim=-1, keepdim=True)
                all_embs.append(embs.cpu())
                all_targets.append(targets)
        return torch.cat(all_embs, dim=0), torch.cat(all_targets, dim=0)

    train_embs, train_targets = encode_images(train_entries)
    test_embs, test_targets = encode_images(test_entries)
    print(f"Embeddings: train={train_embs.shape}, test={test_embs.shape}")

    # Train linear probe
    adapter = nn.Linear(embed_dim, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    train_embs_d = train_embs.to(DEVICE)
    train_targets_d = train_targets.to(DEVICE)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        adapter.train()
        # Mini-batch training on precomputed embeddings
        perm = torch.randperm(len(train_embs_d))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            logits = adapter(train_embs_d[idx])
            loss = criterion(logits, train_targets_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate on test set
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            adapter.eval()
            with torch.no_grad():
                test_logits = adapter(test_embs.to(DEVICE))
                test_preds = (torch.sigmoid(test_logits) >= 0.5).float().cpu()
                # Subset accuracy (exact match per sample)
                exact_match = (test_preds == test_targets).all(dim=1).float().mean().item()
                # Per-label accuracy
                label_acc = (test_preds == test_targets).float().mean(dim=0)
                mean_label_acc = label_acc.mean().item()

            print(
                f"  Epoch {epoch+1:3d}/{args.epochs}  "
                f"loss={avg_loss:.4f}  "
                f"exact_match={exact_match:.1%}  "
                f"mean_label_acc={mean_label_acc:.1%}"
            )

    # Final eval: report least accurate labels and most misclassified images
    adapter.eval()
    with torch.no_grad():
        test_logits = adapter(test_embs.to(DEVICE))
        test_preds = (torch.sigmoid(test_logits) >= 0.5).float().cpu()
        label_acc = (test_preds == test_targets).float().mean(dim=0)
        per_sample_correct = (test_preds == test_targets).float().sum(dim=1)
        per_sample_total = test_targets.shape[1]

    # Worst labels (lowest accuracy on test set)
    label_acc_np = label_acc.numpy()
    worst_label_idx = label_acc_np.argsort()
    n_show = min(15, len(all_labels))
    print(f"\n--- Least accurate labels (test set, worst {n_show}) ---")
    for i in range(n_show):
        idx = worst_label_idx[i]
        acc = label_acc_np[idx]
        name = all_labels[idx]
        support = int(test_targets[:, idx].sum().item())
        print(f"  {name}: {acc:.1%}  (support={support})")

    # Worst images (most label errors on test set)
    errors_per_image = (per_sample_total - per_sample_correct).numpy()
    worst_image_idx = errors_per_image.argsort()[::-1]
    n_show_img = min(10, len(test_entries))
    print(f"\n--- Most misclassified images (test set, worst {n_show_img}) ---")
    for i in range(n_show_img):
        idx = worst_image_idx[i]
        err = int(errors_per_image[idx])
        total = per_sample_total
        fname = test_entries[idx]["filename"]
        print(f"  {fname}: {err}/{total} labels wrong")

    # Save adapter
    torch.save(adapter.state_dict(), args.output_weights)
    meta = {
        "labels": all_labels,
        "embed_dim": embed_dim,
        "num_classes": num_classes,
        "type": "multilabel",
        "train_size": len(train_entries),
        "test_size": len(test_entries),
    }
    with open(args.output_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved weights to {args.output_weights}")
    print(f"Saved metadata to {args.output_meta}")
    print(f"\nTo use in inference, set env vars:")
    print(f"  ADAPTER_WEIGHTS={args.output_weights}")
    print(f"  ADAPTER_META={args.output_meta}")


if __name__ == "__main__":
    main()
