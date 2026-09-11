"""FastAPI service exposing the RAG extraction pipeline.

Endpoints
---------
GET  /health : liveness probe + vector-store stats
POST /query  : retrieve -> extract one line item for one bank
"""
from fastapi import FastAPI, HTTPException

from app.config import COLLECTION_NAME, get_chroma_collection, get_logger
from app.extraction import extract_metrics_from_top_chunks
from app.retrieval import BM25IndexMissingError, bank_exists, build_query, search
from app.schemas import FinancialExtraction, QueryRequest, QueryResponse, SourceChunk

logger = get_logger(__name__)

app = FastAPI(
    title="HKMA Pillar 3 RAG Extraction API",
    description=(
        "Retrieval-augmented extraction of Basel Pillar 3 disclosure line items "
        "from Hong Kong authorized institutions."
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    collection = get_chroma_collection()
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "total_chunks": collection.count(),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not bank_exists(req.bank_name):
        raise HTTPException(
            status_code=404,
            detail=f"Bank '{req.bank_name}' has not been ingested. Run scripts/ingest.py first.",
        )

    q = build_query(req.template, req.item, req.year)

    try:
        chunks, metas = search(req.bank_name, q, strategy=req.strategy, top_k=req.top_k)
    except BM25IndexMissingError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks retrieved.")

    extracted = extract_metrics_from_top_chunks(
        chunks,
        bank_name=req.bank_name,
        query=q,
        year=req.year,
        template=req.template,
        target_item=req.item,
    )
    if extracted is None:
        raise HTTPException(status_code=502, detail="LLM extraction failed.")

    return QueryResponse(
        bank_name=req.bank_name,
        strategy=req.strategy,
        query=q,
        extraction=FinancialExtraction(**extracted),
        sources=[
            SourceChunk(source=m.get("source", "unknown"), preview=c[:300])
            for c, m in zip(chunks, metas)
        ],
    )