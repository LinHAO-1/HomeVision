"""
HomeVision Inference - FastAPI + OpenCLIP + quality heuristics.
POST /analyze/batch: multipart files[] -> JSON array per file
  (room, amenities, features, quality).
"""
import io
import os
from typing import Any

import cv2
import numpy as np
import open_clip
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="HomeVision Inference", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenCLIP model (load once at startup) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model = model.to(DEVICE).eval()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ROOM_PROMPTS = {
    "Kitchen": "a photo of a kitchen",
    "Bathroom": "a photo of a bathroom",
    "Bedroom": "a photo of a bedroom",
    "Living Room": "a photo of a living room",
    "Dining Room": "a photo of a dining room",
    "Exterior": "a photo of the outside of a house",
}

AMENITY_PROMPTS = {
    "Stainless Steel Appliances": "stainless steel appliances",
    "Fireplace": "a fireplace",
    "Pool": "a swimming pool",
    "View": "a scenic view from a home",
    "Natural Light": "a bright room with natural light",
    "Updated Kitchen": "a modern updated kitchen",
}

# 50+ real-estate feature prompts organised by category
FEATURE_PROMPTS: dict[str, dict[str, str]] = {
    "Kitchen": {
        "Kitchen Island": "a kitchen island",
        "Granite Countertop": "granite countertop in a kitchen",
        "Marble Countertop": "marble countertop in a kitchen",
        "Quartz Countertop": "quartz countertop in a kitchen",
        "Stone Countertop": "stone countertop in a kitchen",
        "Stainless Steel Countertop": "stainless steel countertop in a kitchen",
        "Wood Countertop": "wood countertop in a kitchen",
        "Gas Stove": "a gas stove in a kitchen",
        "Electric Stove": "an electric stove in a kitchen",
        "Tile Backsplash": "tile backsplash in a kitchen",
        "Pantry": "a pantry in a kitchen",
        "Double Oven": "a double oven in a kitchen",
        "Breakfast Bar": "a breakfast bar in a kitchen",
        "Under-Cabinet Lighting": "under-cabinet lighting in a kitchen",
        "Dishwasher": "a dishwasher in a kitchen",
    },
    "Bathroom": {
        "Walk-In Shower": "a walk-in shower",
        "Bathtub": "a bathtub in a bathroom",
        "Double Vanity": "a double vanity in a bathroom",
        "Tile Floor": "tile floor in a bathroom",
        "Soaking Tub": "a soaking tub in a bathroom",
        "Glass Shower Door": "a glass shower door",
    },
    "Living Space": {
        "Crown Molding": "crown molding on ceiling",
        "Recessed Lighting": "recessed lighting in a room",
        "Hanging Lights": "hanging pendant lights or chandelier in a room",
        "Ceiling Fan": "a ceiling fan",
        "Built-In Shelving": "built-in shelving in a room",
        "Wainscoting": "wainscoting on walls",
        "Exposed Brick": "exposed brick wall",
        "Accent Wall": "an accent wall in a room",
    },
    "Bedroom": {
        "Walk-In Closet": "a walk-in closet",
        "En-Suite Bathroom": "an en-suite bathroom attached to a bedroom",
        "Bay Window": "a bay window in a room",
        "Window Seat": "a window seat",
    },
    "Flooring": {
        "Hardwood Floors": "hardwood floors in a room",
        "Carpet": "carpet flooring in a room",
        "Tile Flooring": "tile flooring in a room",
        "Laminate Flooring": "laminate flooring in a room",
        "Stone Flooring": "stone flooring in a room",
    },
    "Exterior": {
        "Patio": "a patio outside a house",
        "Deck": "a deck outside a house",
        "Garage": "a garage attached to a house",
        "Front Porch": "a front porch of a house",
        "Landscaping": "professional landscaping around a house",
        "Fenced Yard": "a fenced yard",
        "Driveway": "a driveway leading to a house",
        "Outdoor Kitchen": "an outdoor kitchen",
        "Pergola": "a pergola in a backyard",
        "Garden": "a garden at a house",
    },
    "General": {
        "Open Floor Plan": "an open floor plan in a home",
        "Vaulted Ceiling": "vaulted ceiling in a room",
        "Washer/Dryer": "a washer and dryer in a home",
        "Central AC Unit": "a central air conditioning unit",
        "Skylight": "a skylight in a room",
        "French Doors": "french doors in a home",
        "Sliding Glass Door": "a sliding glass door",
        "Staircase": "a staircase in a home",
        "Laundry Room": "a laundry room",
        "Home Office": "a home office",
        "Storage Space": "storage space or closet in a home",
    },
}

# ---------------------------------------------------------------------------
# Flatten & cache embeddings
# ---------------------------------------------------------------------------

ROOM_LABELS = list(ROOM_PROMPTS.keys())
ROOM_PROMPT_TEXTS = [ROOM_PROMPTS[k] for k in ROOM_LABELS]
AMENITY_LABELS = list(AMENITY_PROMPTS.keys())
AMENITY_PROMPT_TEXTS = [AMENITY_PROMPTS[k] for k in AMENITY_LABELS]

# Flatten features: list of (label, prompt, category)
FEATURE_FLAT: list[tuple[str, str, str]] = []
for cat, prompts in FEATURE_PROMPTS.items():
    for label, prompt in prompts.items():
        FEATURE_FLAT.append((label, prompt, cat))

FEATURE_LABELS = [f[0] for f in FEATURE_FLAT]
FEATURE_PROMPT_TEXTS = [f[1] for f in FEATURE_FLAT]
FEATURE_CATEGORIES = [f[2] for f in FEATURE_FLAT]

with torch.no_grad():
    room_tokens = open_clip.tokenize(ROOM_PROMPT_TEXTS).to(DEVICE)
    room_text_features = model.encode_text(room_tokens)
    room_text_features = room_text_features / room_text_features.norm(dim=-1, keepdim=True)

    amenity_tokens = open_clip.tokenize(AMENITY_PROMPT_TEXTS).to(DEVICE)
    amenity_text_features = model.encode_text(amenity_tokens)
    amenity_text_features = amenity_text_features / amenity_text_features.norm(dim=-1, keepdim=True)

    feature_tokens = open_clip.tokenize(FEATURE_PROMPT_TEXTS).to(DEVICE)
    feature_text_features = model.encode_text(feature_tokens)
    feature_text_features = feature_text_features / feature_text_features.norm(dim=-1, keepdim=True)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ROOM_THRESHOLD = 0.25
AMENITY_THRESHOLD = 0.30
FEATURE_THRESHOLD = 0.25  # Lowered so more features (e.g. kitchen island, countertops) are detected

# ---------------------------------------------------------------------------
# Optional: load fine-tuned adapter if available
# ---------------------------------------------------------------------------

adapter_model = None
adapter_labels: list[str] = []
try:
    import json
    adapter_path = os.environ.get("ADAPTER_WEIGHTS", "adapter.pt")
    adapter_meta_path = os.environ.get("ADAPTER_META", "adapter_meta.json")
    if os.path.isfile(adapter_path) and os.path.isfile(adapter_meta_path):
        with open(adapter_meta_path) as f:
            meta = json.load(f)
        adapter_labels = meta["labels"]
        num_classes = len(adapter_labels)
        embed_dim = model.visual.output_dim if hasattr(model.visual, "output_dim") else 512
        adapter_model = torch.nn.Linear(embed_dim, num_classes).to(DEVICE)
        adapter_model.load_state_dict(torch.load(adapter_path, map_location=DEVICE))
        adapter_model.eval()
        print(f"[adapter] Loaded fine-tuned adapter with {num_classes} classes")
except Exception as e:
    print(f"[adapter] Not loaded: {e}")

# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess for OpenCLIP."""
    return preprocess(pil_img).unsqueeze(0).to(DEVICE)


def compute_analysis(pil_img: Image.Image) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Return (roomType, amenities, features, adapterPredictions)."""
    with torch.no_grad():
        img_t = pil_to_tensor(pil_img)
        image_features = model.encode_image(img_t)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # --- Room type ---
        room_scores = (image_features @ room_text_features.T).cpu().numpy().squeeze()
        best_room_idx = int(room_scores.argmax())
        top_room_score = float(room_scores[best_room_idx])
        label = ROOM_LABELS[best_room_idx] if top_room_score >= ROOM_THRESHOLD else "Unknown"
        room_type = {
            "label": label,
            "score": round(top_room_score, 2),
            "topPrompt": ROOM_PROMPT_TEXTS[best_room_idx],
        }

        # --- Amenities ---
        amenity_scores = (image_features @ amenity_text_features.T).cpu().numpy().squeeze()
        amenities = []
        for i, sc in enumerate(amenity_scores):
            if sc >= AMENITY_THRESHOLD:
                amenities.append({
                    "label": AMENITY_LABELS[i],
                    "score": round(float(sc), 2),
                    "prompt": AMENITY_PROMPT_TEXTS[i],
                })

        # --- Features (50+) ---
        feat_scores = (image_features @ feature_text_features.T).cpu().numpy().squeeze()
        features = []
        for i, sc in enumerate(feat_scores):
            if sc >= FEATURE_THRESHOLD:
                features.append({
                    "label": FEATURE_LABELS[i],
                    "score": round(float(sc), 2),
                    "category": FEATURE_CATEGORIES[i],
                    "prompt": FEATURE_PROMPT_TEXTS[i],
                })
        features.sort(key=lambda x: x["score"], reverse=True)

        # --- Adapter predictions (if loaded) ---
        adapter_preds: list[dict] = []
        if adapter_model is not None:
            adapter_threshold = float(os.environ.get("ADAPTER_CONFIDENCE_THRESHOLD", "0.4"))
            logits = adapter_model(image_features.squeeze(0))
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, p in enumerate(probs):
                if p >= adapter_threshold:
                    adapter_preds.append({
                        "label": adapter_labels[i],
                        "confidence": round(float(p), 2),
                    })

    return room_type, amenities, features, adapter_preds


def quality_metrics(np_img: np.ndarray) -> dict[str, Any]:
    """blurVar, brightness, width, height, isBlurry, isDark, overallScore (0-1 continuous)."""
    h, w = np_img.shape[:2]
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY) if len(np_img.shape) == 3 else np_img
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    is_blurry = blur_var < 100
    is_dark = brightness < 50
    min_dim = min(w, h)
    # Continuous 0-1 score: sharpness (blur_var), brightness, and size
    sharpness = min(1.0, blur_var / 200.0)  # blur_var 200+ => 1
    brightness_score = min(1.0, max(0.0, (brightness - 30) / 120.0))  # 30 dark -> 0, 150+ bright -> 1
    size_score = min(1.0, min_dim / 800.0)  # 800px+ => 1
    overall = 0.5 * sharpness + 0.35 * brightness_score + 0.15 * size_score
    overall = max(0.0, min(1.0, overall))
    return {
        "blurVar": round(blur_var, 1),
        "brightness": round(brightness, 1),
        "width": int(w),
        "height": int(h),
        "isBlurry": is_blurry,
        "isDark": is_dark,
        "overallScore": round(overall, 2),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "adapter_loaded": adapter_model is not None}


@app.post("/analyze/batch")
async def analyze_batch(files: list[UploadFile] = File(...)):
    """Process 1..20 images; return JSON array one per file in same order."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one file required")
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed")

    results = []
    for f in files:
        content = await f.read()
        try:
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image {f.filename}: {e}")
        np_img = np.array(pil_img)

        room_type, amenities, features, adapter_preds = compute_analysis(pil_img)
        quality = quality_metrics(np_img)

        result: dict[str, Any] = {
            "filename": f.filename or "unknown",
            "roomType": room_type,
            "amenities": amenities,
            "features": features,
            "quality": quality,
        }
        if adapter_preds:
            result["adapterPredictions"] = adapter_preds

        results.append(result)
    return results
