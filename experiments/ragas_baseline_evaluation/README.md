# RAGAS baseline evaluation — production, resolved

**This resolves the RAGAS import failure documented in
[`../chunk_size_ablation/README.md`](../chunk_size_ablation/README.md).** That
document diagnosed a three-layer dependency conflict in `RAG-Production/.venv`
(`ragas 0.4.3` → `instructor` → `mistralai` → `langchain_community`, each pair
resolved to mutually incompatible versions because `requirements-offline.txt`
pins `ragas>=0.2` with no upper bound). Rather than force a risky downgrade of
`mistralai` in the venv that `scripts/ingest.py`'s working OCR pipeline
depends on, the fix was to find and use an environment where the same
`ragas==0.4.3` already works against a compatible, older dependency set.

## What actually happened

A separate, long-lived Python environment on the same machine (created months
earlier, before `instructor`/`mistralai`/`langchain-community` had drifted to
their current releases, and kept as a general-purpose environment rather than
recreated per project) still has `instructor==1.14.5` + `mistralai==2.1.2` +
`langchain-community==0.4.1`, which import cleanly against `ragas==0.4.3`.
That environment also already had every other package `scripts/evaluate.py`
and `app/*.py` need (`chromadb`, `rank-bm25`, `langchain-google-genai`,
`langchain-core`, `langchain-text-splitters`, `nltk`) — confirmed by directly
importing `app.config`, `app.retrieval`, `app.extraction`, and the `ragas`
wrapper/metric classes, and confirming it connects to the real production
ChromaDB collection (`pillar3_disclosures`, 70 chunks).

`scripts/evaluate.py` was then run from that interpreter, completely
unmodified, against the live production collection — `RAG-Production/.venv`
itself was never touched further, so the working OCR pipeline was never put
at risk.

```
python3 scripts/evaluate.py --ground-truth ground_truth.xlsx \
    --strategies sparse dense hybrid_rrf --workers 8
```

## Results — production (chunk_size=10000, chunk_overlap=2000), all 3 strategies, 200 queries

| Strategy | Exact-match | Context Precision | Context Recall | Faithfulness | Response Relevancy |
|---|---:|---:|---:|---:|---:|
| sparse | 47.5% | 0.767 | 0.914 | 0.897 | 0.743 |
| dense | 56.0% | **0.880** | **0.955** | 0.896 | 0.745 |
| hybrid_rrf | 54.5% | 0.872 | 0.945 | **0.901** | **0.746** |

Exact-match figures are consistent with the RAGAS-free comparison run the
same evening (`../chunk_size_ablation/accuracy_full_OLD_10000_2000.csv`:
sparse ~49%, dense 56.0%, hybrid_rrf 55.5%) — two independent runs agreeing
is a reasonable sanity check that neither was a fluke.

Dense leads on context precision/recall; hybrid_rrf edges ahead on
faithfulness and response relevancy, though the gaps between dense and
hybrid_rrf are small. Sparse is weakest across every RAGAS metric, consistent
with keyword matching alone retrieving less semantically relevant context
than the embedding-based approaches.

## What this does *not* fix

`RAG-Production/.venv` — the environment a fresh clone of this repo would
actually get from `pip install -r requirements-offline.txt` — still cannot
import `ragas`. This run used a pre-existing, un-reproducible environment
specific to this machine, not a portable fix. The durable fix is to pin exact,
mutually-compatible versions of `ragas`, `instructor`, `mistralai`, and
`langchain-community` in `requirements-offline.txt` (starting from the
versions confirmed working here) so a fresh install reproduces a working
environment — not yet done.

## Files

- `Extraction_Results_sparse.xlsx` / `_dense.xlsx` / `_hybrid_rrf.xlsx` — full
  per-query results including all 4 RAGAS metrics.
- `run_log_trimmed.txt` — per-query log lines (accuracy + timing), httpx/AFC
  noise stripped out.
