"""Retrieval layer: BM25 sparse search, ChromaDB dense search, and hybrid RRF fusion.

Design notes
------------
- Dense search is delegated entirely to ChromaDB (cosine similarity is computed
  inside the HNSW index), filtered per bank via metadata.
- BM25 indices are built offline (scripts/build_bm25.py) and pickled per bank
  as a BM25Bundle. Chunk IDs act as the bridge between the BM25 corpus and the
  Chroma collection, replacing the array-index coupling of the notebook version.
- Query embeddings are cached with an LRU cache since evaluation runs issue
  identical queries across strategies.
"""
import os
import pickle
import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from app.config import (
    BM25_DIR,
    get_chroma_collection,
    get_embedding_model,
    get_logger,
    standardize_bank_name,
)

logger = get_logger(__name__)

RRF_K = 60          # standard reciprocal-rank-fusion constant
RRF_POOL_SIZE = 50  # candidate pool per retriever before fusion

for _pkg in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(_pkg, quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


class BM25IndexMissingError(Exception):
    """Raised when a sparse/hybrid query targets a bank with no BM25 index on disk."""


def preprocess(text: str) -> list[str]:
    """Lowercase, tokenise, remove stopwords, and stem (BM25 preprocessing)."""
    tokens = word_tokenize(text.lower())
    return [_stemmer.stem(w) for w in tokens if w.isalpha() and w not in _stop_words]


def build_query(template: str, item: str, year: str) -> str:
    """Single source of truth for the retrieval/extraction query template."""
    return (
        f"Extract {template} data specifically for: '{item}' in year {year}. "
        f"Return the numerical value and explicitly state the exact unit of "
        f"measurement as found in the source document. "
        f"If the item is not applicable or marked N/A in the document, "
        f"indicate that clearly by setting is_applicable to false."
    )


# ---------------------------------------------------------------------------
# BM25 bundle (built offline, loaded lazily, cached in memory)
# ---------------------------------------------------------------------------
@dataclass
class BM25Bundle:
    chunk_ids: list[str]  # aligned 1:1 with the BM25 corpus rows
    index: BM25Okapi


_bm25_cache: dict[str, BM25Bundle] = {}
_bm25_lock = threading.Lock()


def bm25_path(bank_name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in standardize_bank_name(bank_name))
    return os.path.join(BM25_DIR, f"{safe}.pkl")


def load_bm25_bundle(bank_name: str) -> BM25Bundle:
    key = standardize_bank_name(bank_name)
    with _bm25_lock:
        if key in _bm25_cache:
            return _bm25_cache[key]

    path = bm25_path(bank_name)
    if not os.path.exists(path):
        raise BM25IndexMissingError(
            f"No BM25 index for '{key}'. Run scripts/build_bm25.py first."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    with _bm25_lock:
        _bm25_cache[key] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Dense search helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def embed_query_cached(query: str) -> tuple[float, ...]:
    """LRU-cached query embedding (tuple so it is hashable/immutable)."""
    return tuple(get_embedding_model().embed_query(query))


def bank_exists(bank_name: str) -> bool:
    res = get_chroma_collection().get(
        where={"bank": standardize_bank_name(bank_name)}, limit=1
    )
    return bool(res["ids"])


def _dense_ranked_ids(bank_name: str, query: str, n: int) -> list[str]:
    res = get_chroma_collection().query(
        query_embeddings=[list(embed_query_cached(query))],
        n_results=n,
        where={"bank": standardize_bank_name(bank_name)},
    )
    return res["ids"][0]


def _sparse_ranked_ids(bank_name: str, query: str, n: int) -> list[str]:
    bundle = load_bm25_bundle(bank_name)
    scores = bundle.index.get_scores(preprocess(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [bundle.chunk_ids[i] for i in order]


def _fetch_by_ids(ids: list[str]) -> tuple[list[str], list[dict]]:
    """Fetch documents from Chroma preserving the given ranking order."""
    if not ids:
        return [], []
    res = get_chroma_collection().get(ids=ids)
    by_id = {
        cid: (doc, meta)
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
    }
    docs = [by_id[cid][0] for cid in ids if cid in by_id]
    metas = [by_id[cid][1] for cid in ids if cid in by_id]
    return docs, metas


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def search(
    bank_name: str,
    query: str,
    strategy: str = "hybrid_rrf",
    top_k: int = 3,
) -> tuple[list[str], list[dict]]:
    """Retrieve the top-k chunks for a bank under the given strategy."""
    if strategy == "dense":
        top_ids = _dense_ranked_ids(bank_name, query, top_k)
    elif strategy == "sparse":
        top_ids = _sparse_ranked_ids(bank_name, query, top_k)
    elif strategy == "hybrid_rrf":
        dense_ids = _dense_ranked_ids(bank_name, query, RRF_POOL_SIZE)
        sparse_ids = _sparse_ranked_ids(bank_name, query, RRF_POOL_SIZE)
        rrf_scores: dict[str, float] = defaultdict(float)
        for rank, cid in enumerate(dense_ids):
            rrf_scores[cid] += 1.0 / (RRF_K + rank + 1)
        for rank, cid in enumerate(sparse_ids):
            rrf_scores[cid] += 1.0 / (RRF_K + rank + 1)
        top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    return _fetch_by_ids(top_ids)