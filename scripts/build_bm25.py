"""Build per-bank BM25 indices from the persisted ChromaDB collection.

Chunk IDs stored alongside the BM25 corpus act as the bridge back to Chroma,
enabling hybrid RRF fusion without array-index coupling.

Usage:
    python scripts/build_bm25.py            # all banks in the collection
    python scripts/build_bm25.py --bank "EXAMPLE BANK"
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import BM25_DIR, get_chroma_collection, get_logger, standardize_bank_name  # noqa: E402
from app.retrieval import BM25Bundle, bm25_path, preprocess  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

logger = get_logger("build_bm25")


def build_for_bank(bank_std: str) -> None:
    collection = get_chroma_collection()
    res = collection.get(where={"bank": bank_std})
    ids, docs = res["ids"], res["documents"]
    if not ids:
        logger.warning("No chunks found for %s — skipping", bank_std)
        return

    corpus = [preprocess(d) for d in docs]
    bundle = BM25Bundle(chunk_ids=ids, index=BM25Okapi(corpus))

    os.makedirs(BM25_DIR, exist_ok=True)
    with open(bm25_path(bank_std), "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Built BM25 index for %s (%d chunks)", bank_std, len(ids))


def main():
    parser = argparse.ArgumentParser(description="Build per-bank BM25 indices.")
    parser.add_argument("--bank", default=None, help="Build for a single bank only.")
    args = parser.parse_args()

    if args.bank:
        build_for_bank(standardize_bank_name(args.bank))
        return

    res = get_chroma_collection().get(include=["metadatas"])
    banks = sorted({m["bank"] for m in res["metadatas"] if m and "bank" in m})
    logger.info("Building BM25 indices for %d banks", len(banks))
    for bank in banks:
        build_for_bank(bank)


if __name__ == "__main__":
    main()