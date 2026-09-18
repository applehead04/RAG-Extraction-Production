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

## Update — durable fix landed

The evaluation above was run using a separate, pre-existing local environment
(see "What actually happened" below) because at the time, a fresh
`pip install -r requirements-offline.txt` still couldn't import `ragas` in
`RAG-Production/.venv`. Digging into *why* that one old environment worked
turned up the real root cause: it wasn't the package **versions** at all — its
`mistralai` package had a hand-patched `__init__.py` (`from mistralai.client
import Mistral`) that doesn't exist in any real PyPI release of `mistralai`.
`instructor`'s mistral provider (pulled in transitively by `ragas`) expects
exactly that top-level `mistralai.Mistral` export, which current `mistralai`
releases don't provide — only `mistralai.client.Mistral`, which is what
`scripts/ingest.py` already uses.

The durable fix, now in the repo:

- `requirements-offline.txt` pins `ragas==0.4.3`, `instructor==1.14.5`, and
  `langchain-community==0.4.1` — a combination verified (via a completely
  fresh venv, installed only from this repo's requirement files, nothing
  copied from any personal machine state) to import cleanly. `mistralai`
  itself is **not pinned down** — it stays on whatever recent release
  `scripts/ingest.py` already resolves and relies on.
- `scripts/evaluate.py` now includes a small compatibility shim before its
  `ragas` imports: if `mistralai.Mistral` isn't exposed at the top level, it
  patches it in from `mistralai.client.Mistral` at runtime. This is the same
  fix the hand-patched environment had, just written as three lines of
  version-controlled code instead of a manual, undocumented edit to
  `site-packages` on one machine.

Verified end-to-end: `pip install -r requirements-offline.txt` into a brand
new venv, then `python scripts/evaluate.py --help` runs cleanly (deprecation
warnings only), and `scripts/ingest.py` still imports and resolves
`mistralai` completely normally — the OCR pipeline was never touched.

## Files

- `Extraction_Results_sparse.xlsx` / `_dense.xlsx` / `_hybrid_rrf.xlsx` — full
  per-query results including all 4 RAGAS metrics.
- `run_log_trimmed.txt` — per-query log lines (accuracy + timing), httpx/AFC
  noise stripped out.
