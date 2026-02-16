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
HomeVisionV2/
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
├── docker-compose.yml
├── docker-compose.dev.yml
└── .env
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
git clone <repo-url> && cd HomeVisionV2
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

That starts all four services: Postgres, inference, API, and web — with hot-reload enabled for development.

Open **http://localhost:3000** in your browser.

### Environment

Create a `.env` file in the project root if you want to use an external Postgres instance:

```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

If omitted, the Docker Compose stack uses a local Postgres container with default credentials.

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

HomeVision includes a built-in labeling tool and adapter training pipeline for improving predictions on your own data.

### Labeling tool

Set `ENABLE_LABELING=true` (already set in the dev compose) and open **http://localhost:3000/label**. From there you can:

- Upload a photo and see the model's predictions
- Correct room type, amenities, and features
- Save labels to the database

### Export labels

```bash
curl http://localhost:3001/api/v1/labels/export > labels.json
```

### Train adapter

Place training images in the `images/` directory and run:

```bash
cd apps/inference
python train_adapter.py --labels ../../labels.json --images ../../images --epochs 20
```

This trains a lightweight linear adapter on frozen CLIP embeddings. The output (`adapter.pt` and `adapter_meta.json`) can be mounted into the inference container — see `docker-compose.yml` for the volume config.

### Evaluate

```bash
python evaluate.py --labels ../../labels.json --images ../../images --adapter adapter.pt --meta adapter_meta.json
```

Reports per-label precision, recall, F1, and macro averages.

