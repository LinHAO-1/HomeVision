"""
HomeVision Evaluation Script
Measures zero-shot inference accuracy against labeled ground truth.

Usage:
  python evaluate.py --labels labels.json --images-dir ./images

labels.json format (export from /api/v1/labels/export):
[
  { "filename": "img1.jpg", "roomType": "Kitchen", "amenities": [...], "features": [...] },
  ...
]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import requests

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://localhost:8000")


def load_labels(path: str) -> list[dict]:
    with open(path) as labels_file:
        return json.load(labels_file)


def run_inference(image_paths: list[str]) -> list[dict]:
    """Send images to inference and return results."""
    files = [("files", (os.path.basename(image_path), open(image_path, "rb"), "image/jpeg")) for image_path in image_paths]
    response = requests.post(f"{INFERENCE_URL}/analyze/batch", files=files)
    for _, (_, file_handle, _) in files:
        file_handle.close()
    response.raise_for_status()
    return response.json()


def compute_metrics(y_true: list[set[str]], y_pred: list[set[str]], all_labels: set[str]):
    """Compute per-label and macro precision, recall, F1."""
    per_label: dict[str, dict[str, float]] = {}

    for label in sorted(all_labels):
        tp = sum(1 for true_set, pred_set in zip(y_true, y_pred) if label in true_set and label in pred_set)
        fp = sum(1 for true_set, pred_set in zip(y_true, y_pred) if label not in true_set and label in pred_set)
        fn = sum(1 for true_set, pred_set in zip(y_true, y_pred) if label in true_set and label not in pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_label[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": tp + fn,
        }

    # Macro average (only labels with support > 0)
    labels_with_support = [label_name for label_name, metrics in per_label.items() if metrics["support"] > 0]
    macro_p = sum(per_label[label_name]["precision"] for label_name in labels_with_support) / max(len(labels_with_support), 1)
    macro_r = sum(per_label[label_name]["recall"] for label_name in labels_with_support) / max(len(labels_with_support), 1)
    macro_f1 = sum(per_label[label_name]["f1"] for label_name in labels_with_support) / max(len(labels_with_support), 1)

    return per_label, {
        "macro_precision": round(macro_p, 3),
        "macro_recall": round(macro_r, 3),
        "macro_f1": round(macro_f1, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate HomeVision inference accuracy")
    parser.add_argument("--labels", required=True, help="Path to labels.json")
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--output", default=None, help="Save results JSON to file")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for inference")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    images_dir = Path(args.images_dir)

    # Validate all images exist
    valid = []
    for entry in labels:
        img_path = images_dir / entry["filename"]
        if img_path.is_file():
            valid.append((entry, str(img_path)))
        else:
            print(f"  SKIP: {entry['filename']} not found in {images_dir}")

    if not valid:
        print("No valid image-label pairs found. Exiting.")
        sys.exit(1)

    print(f"Evaluating {len(valid)} images...")

    # Run inference in batches
    predictions: list[dict] = []
    for i in range(0, len(valid), args.batch_size):
        batch = valid[i : i + args.batch_size]
        paths = [path for _, path in batch]
        batch_predictions = run_inference(paths)
        predictions.extend(batch_predictions)
        print(f"  Batch {i // args.batch_size + 1}: {len(batch_predictions)} results")

    # --- Room Type accuracy ---
    room_true = []
    room_pred = []
    for (entry, _), pred in zip(valid, predictions):
        room_true.append(entry["roomType"])
        room_pred.append(pred["roomType"]["label"])

    room_correct = sum(1 for true_room, pred_room in zip(room_true, room_pred) if true_room == pred_room)
    room_accuracy = room_correct / len(room_true)
    print(f"\n=== Room Type ===")
    print(f"Accuracy: {room_correct}/{len(room_true)} = {room_accuracy:.1%}")

    # Per-room breakdown
    all_rooms = set(room_true + room_pred)
    room_true_sets = [{room} for room in room_true]
    room_pred_sets = [{room} for room in room_pred]
    room_per_label, room_macro = compute_metrics(room_true_sets, room_pred_sets, all_rooms)
    for label, metrics in room_per_label.items():
        if metrics["support"] > 0:
            print(f"  {label:20s}  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}  (n={metrics['support']})")
    print(f"  {'MACRO':20s}  P={room_macro['macro_precision']:.2f}  R={room_macro['macro_recall']:.2f}  F1={room_macro['macro_f1']:.2f}")

    # --- Amenities ---
    amenity_true = [set(entry.get("amenities", [])) for entry, _ in valid]
    amenity_pred = [set(amenity["label"] for amenity in pred.get("amenities", [])) for pred in predictions]
    all_amenities = set()
    for label_set in amenity_true + amenity_pred:
        all_amenities.update(label_set)
    amenity_per_label, amenity_macro = compute_metrics(amenity_true, amenity_pred, all_amenities)

    print(f"\n=== Amenities ===")
    for label, metrics in amenity_per_label.items():
        if metrics["support"] > 0:
            print(f"  {label:30s}  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}  (n={metrics['support']})")
    print(f"  {'MACRO':30s}  P={amenity_macro['macro_precision']:.2f}  R={amenity_macro['macro_recall']:.2f}  F1={amenity_macro['macro_f1']:.2f}")

    # --- Features ---
    feature_true = [set(entry.get("features", [])) for entry, _ in valid]
    feature_pred = [set(feature["label"] for feature in pred.get("features", [])) for pred in predictions]
    all_features = set()
    for label_set in feature_true + feature_pred:
        all_features.update(label_set)
    feature_per_label, feature_macro = compute_metrics(feature_true, feature_pred, all_features)

    print(f"\n=== Features ===")
    for label, metrics in feature_per_label.items():
        if metrics["support"] > 0:
            print(f"  {label:30s}  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}  (n={metrics['support']})")
    print(f"  {'MACRO':30s}  P={feature_macro['macro_precision']:.2f}  R={feature_macro['macro_recall']:.2f}  F1={feature_macro['macro_f1']:.2f}")

    # --- Save results ---
    results = {
        "num_images": len(valid),
        "room": {"accuracy": round(room_accuracy, 3), "per_label": room_per_label, "macro": room_macro},
        "amenities": {"per_label": amenity_per_label, "macro": amenity_macro},
        "features": {"per_label": feature_per_label, "macro": feature_macro},
    }

    if args.output:
        with open(args.output, "w") as output_file:
            json.dump(results, output_file, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
