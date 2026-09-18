# Chunk size ablation — does `chunk_size`/`chunk_overlap` affect extraction accuracy?

**Motivation**: production ingestion (`scripts/ingest.py`) uses `chunk_size=10000, chunk_overlap=2000`
(`RecursiveCharacterTextSplitter`). A ChromaDB metadata audit showed the resulting corpus is very coarse
— 20 banks, 70 chunks total, median 3 chunks/bank, 65% of banks at ≤3 chunks (i.e. `top_k=3` already
returns nearly the entire document for most banks, making cross-strategy retrieval comparisons
degenerate for the majority of the dataset). This raised a natural follow-up: does the coarse chunk size
itself introduce an extraction-accuracy problem — specifically, does bundling many differently-unit-ed
line items (e.g. a `%` ratio next to `HK$M` monetary figures) into one large chunk confuse the LLM's
per-item unit extraction?

**Method**: `app/config.py` was extended with `CHUNK_SIZE`/`CHUNK_OVERLAP` env-configurable constants
(previously hardcoded in `scripts/ingest.py`), defaulting to the existing production values. A second,
fully isolated ChromaDB collection (`pillar3_disclosures_chunk3000`, `BM25_DIR=./bm25_index_chunk3000`)
was ingested from scratch with `chunk_size=3000, chunk_overlap=1000` using the real `scripts/ingest.py`
+ `scripts/build_bm25.py` — the production collection (`pillar3_disclosures`) was never written to.
All 200 ground-truth rows × all 3 retrieval strategies (`sparse`, `dense`, `hybrid_rrf`) were then run
through the real `app.retrieval.search()` + `app.extraction.extract_metrics_from_top_chunks()` against
both collections, and compared against `ground_truth.xlsx` on value and unit correctness.

**RAGAS was not used for this specific experiment** — at the time this ablation was run,
`scripts/evaluate.py` failed to import in `RAG-Production/.venv` (`ragas 0.4.3` → `instructor 1.3.2` →
`mistralai.async_client`, which no longer exists in the installed `mistralai==2.10.0`; upgrading
`instructor` only pushed the failure one layer deeper, to `ragas.llms.base` importing
`langchain_community.chat_models.vertexai`, which has been relocated out of `langchain_community`
upstream). `requirements-offline.txt` pins `ragas>=0.2` with no upper bound, so `pip install`-ing it
fresh today resolves a different, incompatible set of transitive versions than whatever was installed
when this repo's `ragas` support was first written. Value/unit exact-match against `ground_truth.xlsx`
was used instead for this ablation.

**This was later resolved** — see
[`../ragas_baseline_evaluation/`](../ragas_baseline_evaluation/) for a real RAGAS evaluation (all 4
metrics, all 3 strategies, full 200-query set) run against production by using a different, older
environment on the same machine where `ragas` still imports cleanly, rather than risking a `mistralai`
downgrade inside `RAG-Production/.venv`. The durable fix — pinning exact compatible versions in
`requirements-offline.txt` so a fresh `pip install` reproduces a working environment — is noted there
as still outstanding.

## Files

- `accuracy_full_OLD_10000_2000.csv` — full run (3 strategies × 200 rows) against the real production
  chunking/collection.
- `accuracy_full_NEW_3000_1000.csv` — full run (3 strategies × 200 rows) against the `chunk_size=3000,
  chunk_overlap=1000` test collection.
- `analysis/Chunk_Size_Ablation_Analysis.xlsx` — Table A (EM accuracy), Table B (Fisher's exact
  significance + Cramér's V per strategy/metric), Table C (error taxonomy), Table D (per-bank
  breakdown), and the combined raw data, all in one workbook.
- `analysis/fig1_fully_correct_by_strategy.png` — grouped bar, fully-correct % by strategy × chunking.
- `analysis/fig2_heatmap.png` — fully-correct % heatmap (strategy × chunking).
- `analysis/fig3_error_taxonomy.png` — stacked bar of error categories (correct / unit-wrong-only /
  value-wrong-only / both-wrong) by chunking.
- `analysis/fig4_value_vs_unit_tradeoff.png` — scatter of value EM vs unit EM per strategy × chunking,
  visualising the trade-off directly.

(An earlier single-bank pilot and an intermediate rate-limited partial run were used to develop this
methodology but are not included here since they're superseded by the full runs above.)

## Result summary

| Strategy | Chunking | Value EM | Unit EM | Fully correct |
|---|---|---:|---:|---:|
| sparse | 10000/2000 (prod) | 85.1% | 52.6% | 49.0% |
| sparse | 3000/1000 | 66.7% | 57.8% | 48.4% |
| dense | 10000/2000 (prod) | 90.5% | 58.0% | 56.0% |
| dense | 3000/1000 | 75.5% | 64.0% | 55.5% |
| hybrid_rrf | 10000/2000 (prod) | 91.0% | 57.0% | 55.5% |
| hybrid_rrf | 3000/1000 | 85.0% | 71.0% | **65.0%** |

**Finding**: smaller chunks consistently trade value-recall for unit-precision across all three retrieval
strategies (value EM drops, unit EM rises). `sparse`/`dense` alone roughly break even on the combined
"fully correct" metric. `hybrid_rrf` is the only strategy where the trade nets positive — RRF fusion's
two independent retrieval signals appear to offset the value-recall loss from finer chunking while still
keeping the unit-precision gain, making `hybrid_rrf` + smaller chunking the best combination tested
(65.0% fully correct, the highest cell in the whole grid). A per-bank breakdown (Table D) shows the gain
concentrated in banks whose disclosures were long enough to produce many chunks even under the old
10000/2000 setting (e.g. +50pp for ICBC (Asia), +30pp for ZA Bank and DBS) — consistent with the
retrieval-degeneracy issue that originally motivated this experiment.

**Statistical significance** (Fisher's exact test, OLD vs NEW, per strategy — see Table B): the
`value_ok` drop is significant for `sparse` and `dense` (p < 0.001). The `unit_ok` gain for `hybrid_rrf`
is significant (p = 0.0048). The headline `hybrid_rrf` "fully correct" improvement (55.5% → 65.0%) is
directionally consistent with the rest of the evidence but **is not itself significant at n=200**
(p = 0.0657) — reported as a caveat, not smoothed over.

**Caveats**: single embedding/LLM config held constant (`gemini-embedding-001` /
`gemini-2.5-flash-lite`); no RAGAS-level context precision/recall/faithfulness scoring (see above);
~7-14 rows per run skipped on transient retrieval errors, not analyzed further; this is one ablation
point (3000/1000) against the production default (10000/2000), not a sweep across multiple sizes.
