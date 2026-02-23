# HomeVision

Real estate listing photo analysis. Upload photos of a property, get back room classifications, detected amenities and features, and photo quality scores — all powered by OpenCLIP zero-shot inference.

## What It Does

- **Room classification** — Kitchen, Bathroom, Bedroom, Living Room, Dining Room, Exterior (with confidence scores)
- **Amenity detection** — Stainless steel appliances, fireplace, pool, scenic view, natural light, updated kitchen, and more
- **Feature detection** — 50+ features across categories like flooring, countertops, fixtures, and outdoor elements
- **Photo quality scoring** — Sharpness (blur detection), brightness, resolution, and an overall quality score per photo
- **Batch processing** — Upload 1–20 photos at once; results come back per-photo with an aggregated summary
- **Optional fine-tuned adapter** — Train a linear adapter on labeled data for domain-specific predictions

## Tech Stack

| Layer      | Stack                                              |
|------------|----------------------------------------------------|
| Frontend   | Next.js 14, React 18, TypeScript                   |
| Backend    | NestJS 10, TypeORM, Postgres 16                    |
| Inference  | FastAPI, PyTorch, OpenCLIP (ViT-B-32), OpenCV      |
| Infra      | Docker Compose, multi-stage Dockerfiles             |

## Project Structure

```
HomeVision/
├── apps/
│   ├── api/              # NestJS backend
│   │   └── src/
│   │       ├── jobs/     # Job creation, polling, inference orchestration
│   │       └── labels/   # CRUD for training labels
│   ├── inference/        # FastAPI + OpenCLIP inference
│   │   ├── main.py       # /analyze/batch endpoint
│   │   ├── train_adapter.py
│   │   └── evaluate.py
│   └── web/              # Next.js frontend
│       └── src/
│           ├── app/      # Pages (home, labeling)
│           └── components/
├── docker-compose.yml          # Base services
├── docker-compose.dev.yml      # Dev overrides (hot-reload, volume mounts)
└── docker-compose.adapter.yml  # Optional: mount trained adapter files
```

## Architecture

- **Web** — Upload UI, async job polling, photo grid with results, click-to-expand detail modal
- **API** — Accepts uploads, creates async jobs in Postgres, forwards images to the inference service, stores results as JSON
- **Inference** — Loads OpenCLIP ViT-B-32 at startup, caches text embeddings for all prompts, runs zero-shot classification and quality analysis per image

Postgres stores job status and result JSON. Images are processed in-memory only — nothing is written to disk or cloud storage.

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
- Ports 3000, 3001, 5432, and 8000 available

### Run

```bash
git clone https://github.com/LinHAO-1/HomeVision.git && cd HomeVision
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

That starts all four services: Postgres, inference, API, and web — with hot-reload enabled for development. No additional setup or dependencies required.

Open **http://localhost:3000** in your browser.

### Compose files

The project uses layered Docker Compose files:

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Base services (Postgres, inference, API, web) |
| `docker-compose.dev.yml` | Dev overrides — hot-reload, source volume mounts |
| `docker-compose.adapter.yml` | **Optional** — mounts trained adapter files for enhanced predictions |

For development, use the first two. Add the third after training an adapter (see [Labeling & Training](#labeling--training)).

### Environment

Optionally create a `.env` file in the project root to use an external Postgres instance:

```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

If omitted, the stack uses the local Postgres container with default credentials.

### Stop

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

Add `-v` to also remove the Postgres data volume.

## Usage

1. Open http://localhost:3000
2. Click **Choose files** and select 1–20 property photos (JPEG, PNG, GIF, or WebP — max 5 MB each)
3. Click **Analyze**
4. Results appear once processing completes — usually a few seconds per photo

The results page shows:
- **Summary** — overall quality score and most common amenities across the batch
- **Photo grid** — each photo card displays room type, amenities, detected features, and quality indicators
- **Detail modal** — click any photo for a larger view with full breakdown

## API Reference

Swagger docs are available at **http://localhost:3001/api/docs** when the API is running.

### Create a job

```bash
curl -X POST http://localhost:3001/api/v1/jobs \
  -F "files=@kitchen.jpg" \
  -F "files=@bedroom.jpg"
```

Response:
```json
{ "jobId": "c0a80164-...", "status": "processing" }
```

### Poll for results

```bash
curl http://localhost:3001/api/v1/jobs/<jobId>
```

Response (when completed):
```json
{
  "id": "c0a80164-...",
  "status": "completed",
  "results": {
    "summary": {
      "topAmenities": [
        { "label": "Natural Light", "count": 3, "avgScore": 0.35 }
      ],
      "topFeatures": [
        { "label": "Hardwood Floors", "category": "Flooring", "count": 2, "avgScore": 0.31 }
      ],
      "overallQualityScore": 0.85
    },
    "photos": [
      {
        "filename": "kitchen.jpg",
        "roomType": { "label": "Kitchen", "score": 0.33, "topPrompt": "a photo of a kitchen" },
        "amenities": [
          { "label": "Stainless Steel Appliances", "score": 0.41, "prompt": "stainless steel appliances" }
        ],
        "features": [
          { "label": "Granite Countertops", "score": 0.29, "category": "Kitchen", "prompt": "..." }
        ],
        "quality": {
          "blurVar": 120.5,
          "brightness": 150.0,
          "width": 1920,
          "height": 1080,
          "isBlurry": false,
          "isDark": false,
          "overallScore": 0.90
        }
      }
    ]
  },
  "errorMessage": null
}
```

### Inference directly

```bash
curl -X POST http://localhost:8000/analyze/batch \
  -F "files=@photo.jpg"
```

Returns the same per-photo array without the job wrapper.

## Labeling & Training

HomeVision includes a labeling tool and adapter training pipeline for improving predictions on your own data. The adapter is completely optional. Without it, the app uses zero-shot OpenCLIP inference out of the box.

The full workflow has four steps, done in order:

### 1. Label photos

Open **http://localhost:3000/label** (labeling is already enabled in the dev compose). Upload a photo and the model shows its predictions. Correct the room type, amenities, and features as needed, then save. Each saved label is stored in your database.

### 2. Put training images on disk

The training script needs access to the actual image files on disk. Create an `images/` folder at the repo root and place the same photos you labeled there (e.g. `images/kitchen/photo1.jpg`). The filenames must match what you used when labeling.

The dev compose mounts `./images` to `/data/images` inside the inference container, so anything you put in `images/` is available to the training script.

### 3. Export labels and train

First, export your labels from the database into a JSON file that the training script can read:

```bash
curl -s http://localhost:3001/api/v1/labels/export -o apps/inference/labels.json
```

This pulls every label you saved and writes it to `apps/inference/labels.json`.

Then run training inside the inference container. No local Python setup needed:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec inference \
  python train_adapter.py --labels labels.json --images-dir /data/images/kitchen \
  --output-weights adapter.pt --output-meta adapter_meta.json --epochs 60
```

This trains a lightweight linear adapter on frozen CLIP embeddings. The output files (`adapter.pt` and `adapter_meta.json`) land in `apps/inference/` on your host since the dev compose mounts that directory to `/app` in the container.

### 4. Load the adapter

Restart the stack with the adapter override so inference picks up the trained weights:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.adapter.yml up --build
```

The adapter override mounts `apps/inference/adapter.pt` and `apps/inference/adapter_meta.json` into the inference container. No copying files around. Training output and inference input are the same location.

You can verify the adapter loaded by hitting the health endpoint:

```bash
curl http://localhost:8000/health
# {"status":"ok","adapter_loaded":true}
```

### Evaluate (optional)

After training, you can check how well the model performs against your labeled data:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec inference \
  python evaluate.py --labels labels.json --images-dir /data/images/kitchen
```

This reports per label accuracy and shows which images the model struggles with the most. Useful for deciding whether to add more labels or adjust your training data before retraining.

