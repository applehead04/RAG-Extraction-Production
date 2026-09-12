# HKMA Pillar 3 RAG Extraction Service

Retrieval-augmented extraction of Basel Pillar 3 disclosure line items (e.g.
template OV1) from Hong Kong authorized institutions, exposed as a FastAPI
service backed by a persistent ChromaDB vector store.

## Architecture

```
                    OFFLINE (local machine)                      ONLINE (Docker / cloud)
┌─────────────────────────────────────────────────┐   ┌────────────────────────────────────┐
│ scripts/ingest.py                               │   │ app/main.py (FastAPI)              │
│   HKMA VPR scraping → Mistral OCR →             │   │   POST /query                      │
│   chunking → Gemini embeddings → ChromaDB       │──▶│     ├─ app/retrieval.py            │
│                                                 │   │     │    BM25 / Dense / Hybrid RRF │
│ scripts/build_bm25.py                           │──▶│     └─ app/extraction.py           │
│   per-bank BM25 indices (pickled)               │   │          Gemini structured output  │
│                                                 │   │   GET /health                      │
│ scripts/evaluate.py                             │   └────────────────────────────────────┘
│   RAGAS metrics + ground-truth accuracy         │        ▲ volumes: chroma_db/, bm25_index/
└─────────────────────────────────────────────────┘
```

## Features

- **Three retrieval strategies**: BM25 (sparse), dense vector search (ChromaDB,
  cosine/HNSW), and hybrid Reciprocal Rank Fusion — selectable per request.
- **Structured LLM extraction** with a Pydantic-enforced schema, including
  explicit disambiguation between *value is zero* and *not applicable*.
- **Offline/online split**: heavy dependencies (Selenium, OCR, RAGAS) never
  enter the Docker image; the API container stays small.
- **Idempotent ingestion**: re-running ingestion for a bank replaces its
  chunks atomically via metadata-scoped deletes.
- **Evaluation harness**: per-strategy accuracy (numeric and N/A items
  reported separately) plus four RAGAS metrics (context precision, context
  recall, faithfulness, response relevancy).

## Quickstart

### 1. Setup

```bash
cp .env.example .env          # fill in your API keys
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-offline.txt
```

### 2. Ingest (offline, one-off)

```bash
python scripts/ingest.py --ground-truth ground_truth.xlsx --links ov1_link.xlsx
python scripts/build_bm25.py
```

### 3. Serve the API

```bash
uvicorn app.main:app --reload
```

Or with Docker:

```bash
docker build -t pillar3-rag .
docker run -p 8000:8000 --env-file .env \
  -v "$(pwd)/chroma_db:/srv/chroma_db" \
  -v "$(pwd)/bm25_index:/srv/bm25_index" \
  pillar3-rag
```

### 4. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
        "bank_name": "EXAMPLE BANK (HONG KONG) LIMITED",
        "item": "Credit risk (excluding counterparty credit risk)",
        "template": "OV1",
        "year": "2024",
        "strategy": "hybrid_rrf"
      }'
```

Interactive docs are available at `http://localhost:8000/docs`.

### 5. Deploy to Cloud Run

The container reads `$PORT` if the platform injects one, so no manual
`--port` flag is needed on Cloud Run.

```bash
# Build for linux/amd64 — required if you're on Apple Silicon, since Cloud
# Run's infrastructure is amd64 and a locally-built arm64 image will be
# rejected. --provenance=false --sbom=false avoid a multi-manifest image
# index that has caused Cloud Run to fail resolving the actual image layers.
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -t <region>-docker.pkg.dev/<project>/<repo>/pillar3-rag:latest --push .

# chroma_db/ and bm25_index/ aren't baked into the image (see Architecture
# above) — upload them to a bucket and mount it as a volume at deploy time.
gcloud storage cp -r chroma_db bm25_index gs://<bucket>/

gcloud run deploy pillar3-rag \
  --image=<region>-docker.pkg.dev/<project>/<repo>/pillar3-rag:latest \
  --region=<region> --allow-unauthenticated --execution-environment=gen2 \
  --add-volume=name=data,type=cloud-storage,bucket=<bucket> \
  --add-volume-mount=volume=data,mount-path=/data \
  --set-env-vars="GOOGLE_API_KEY=...,CHROMA_DIR=/data/chroma_db,BM25_DIR=/data/bm25_index,CHROMA_COLLECTION=pillar3_disclosures"
```

**If you're changing volume/mount configuration on an existing service and
hit a confusing state** (e.g. `ModuleNotFoundError: No module named 'app'`
despite the image being fine, or a "volumes ... not found" error):
`gcloud run deploy` merges new flags into the existing service spec rather
than replacing it outright, so stale volume config from an earlier deploy
can silently persist. Delete the service and redeploy clean rather than
trying to patch it further:

```bash
gcloud run services delete pillar3-rag --region=<region>
# then re-run the gcloud run deploy command above
```

### 6. Evaluate (offline)

```bash
python scripts/evaluate.py --ground-truth ground_truth.xlsx --workers 8
```

Outputs `Extraction_Results_<strategy>.xlsx` per strategy with per-query
correctness flags and RAGAS scores.

## Error Handling

| Status | Meaning |
|--------|---------|
| `404`  | Bank not ingested, or no relevant chunks retrieved |
| `409`  | BM25 index missing (sparse/hybrid requested before `build_bm25.py`) |
| `502`  | LLM extraction failed after retries |

## Project Structure

| Path | Role | Ships in Docker image |
|------|------|:---:|
| `app/config.py` | Env, logging, cached client factories | ✅ |
| `app/schemas.py` | LLM output schema + API contracts | ✅ |
| `app/retrieval.py` | BM25 / dense / hybrid RRF search | ✅ |
| `app/extraction.py` | Structured LLM extraction + cache | ✅ |
| `app/main.py` | FastAPI endpoints | ✅ |
| `scripts/ingest.py` | Scrape → OCR → embed → ChromaDB | ❌ |
| `scripts/build_bm25.py` | Per-bank BM25 index builder | ❌ |
| `scripts/evaluate.py` | RAGAS + ground-truth evaluation | ❌ |