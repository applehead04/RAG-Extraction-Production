"""Offline evaluation harness (run locally, NOT shipped in the Docker image).

For each ground-truth row and each retrieval strategy:
  retrieve (app.retrieval) -> extract (app.extraction) -> compare against
  ground truth (with explicit N/A branching) -> score with RAGAS.

Usage:
    python scripts/evaluate.py --ground-truth ground_truth.xlsx --workers 8
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import GOOGLE_API_KEY, get_logger  # noqa: E402
from app.extraction import extract_metrics_from_top_chunks  # noqa: E402
from app.retrieval import build_query, search  # noqa: E402

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)

logger = get_logger("evaluate")

# ===========================================================================
# RAGAS setup: dedicated evaluator instances + output sanitisation
# ===========================================================================
_ragas_base_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", temperature=0.0,
    google_api_key=GOOGLE_API_KEY, max_retries=3,
)
_ragas_base_emb = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY,
)
ragas_llm = LangchainLLMWrapper(_ragas_base_llm)
ragas_emb = LangchainEmbeddingsWrapper(_ragas_base_emb)


def _sanitize_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\\'", "'").strip()
    text = re.sub(r"^```(?:json)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _fix_result(result):
    """Strip markdown fences / escape artefacts from evaluator LLM outputs."""
    if isinstance(result, str):
        return _sanitize_text(result)
    if hasattr(result, "generations"):
        for gen_list in result.generations:
            if isinstance(gen_list, list):
                for gen in gen_list:
                    if hasattr(gen, "text"):
                        gen.text = _sanitize_text(gen.text)
                    if hasattr(gen, "message") and hasattr(gen.message, "content"):
                        gen.message.content = _sanitize_text(gen.message.content)
            elif hasattr(gen_list, "text"):
                gen_list.text = _sanitize_text(gen_list.text)
    if hasattr(result, "text"):
        result.text = _sanitize_text(result.text)
    return result


_orig_agenerate = ragas_llm.agenerate_text


async def _patched_agenerate(*args, **kwargs):
    return _fix_result(await _orig_agenerate(*args, **kwargs))


ragas_llm.agenerate_text = _patched_agenerate

if hasattr(ragas_llm, "generate_text"):
    _orig_generate = ragas_llm.generate_text

    def _patched_generate(*args, **kwargs):
        return _fix_result(_orig_generate(*args, **kwargs))

    ragas_llm.generate_text = _patched_generate

context_precision = LLMContextPrecisionWithoutReference(llm=ragas_llm)
context_recall = LLMContextRecall(llm=ragas_llm)
faithfulness = Faithfulness(llm=ragas_llm)
response_relevancy = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb)

ALL_METRICS = [context_precision, context_recall, faithfulness, response_relevancy]
METRIC_KEY_MAP = {
    context_precision.name: "context_precision",
    context_recall.name: "context_recall",
    faithfulness.name: "faithfulness",
    response_relevancy.name: "response_relevancy",
}

# Shared event loop for async metric scoring
_RAGAS_LOOP = asyncio.new_event_loop()
threading.Thread(
    target=lambda: (asyncio.set_event_loop(_RAGAS_LOOP), _RAGAS_LOOP.run_forever()),
    daemon=True, name="ragas-loop",
).start()

_RAGAS_ASYNC_SEM = asyncio.Semaphore(6)   # concurrent metric calls
_RAGAS_THROTTLE = threading.Semaphore(3)  # concurrent sample evaluations

_ragas_eval_cache: dict[str, dict] = {}
_ragas_cache_lock = threading.Lock()


def _sanitize_for_ragas(text):
    if not isinstance(text, str):
        return str(text)
    text = text.replace("HK$'000", "HK$000").replace("hk$'000", "hk$000")
    text = text.replace("HK$\\'000", "HK$000")
    return text.replace("\u2018", "'").replace("\u2019", "'")


def _flatten_response_to_text(json_str):
    """Fallback: convert the structured JSON response into plain prose."""
    try:
        parsed = json.loads(json_str)
    except Exception:
        return json_str
    parts = []
    for k, tmpl in [("bank_name", "The bank is {}."), ("template", "The template is {}."),
                    ("year", "The year is {}.")]:
        if k in parsed:
            parts.append(tmpl.format(parsed[k]))
    if "unit" in parsed:
        parts.append(f"The unit of measurement is {str(parsed['unit']).replace(chr(39), '')}.")
    for d in parsed.get("data") or []:
        num, name = d.get("item_number", "?"), d.get("item", "?")
        if not d.get("is_applicable", True):
            parts.append(f"Line item {num} ({name}) is not applicable.")
        else:
            parts.append(f"Line item {num} ({name}) has a value of {d.get('value_1', '?')}.")
    return " ".join(parts) if parts else json_str


async def _score_one_metric(metric, sample, max_retries=5):
    last_err, current_sample = None, sample
    async with _RAGAS_ASYNC_SEM:
        for attempt in range(max_retries):
            try:
                score = await metric.single_turn_ascore(current_sample)
                if isinstance(score, float) and np.isnan(score):
                    return (metric.name, None)
                return (metric.name, score)
            except Exception as e:
                last_err = e
                # First failure: retry once with a flattened prose response.
                if attempt == 0 and current_sample.response:
                    flat = _flatten_response_to_text(current_sample.response)
                    if flat != current_sample.response:
                        current_sample = SingleTurnSample(
                            user_input=current_sample.user_input,
                            response=_sanitize_for_ragas(flat),
                            reference=current_sample.reference,
                            retrieved_contexts=current_sample.retrieved_contexts,
                        )
                        continue
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(2 ** attempt, 16))
    logger.warning("[%s] failed after %d attempts: %s", metric.name, max_retries, str(last_err)[:150])
    return (metric.name, None)


def evaluate_with_ragas(query, response_text, reference_text, retrieved_contexts):
    response_text = _sanitize_for_ragas(response_text)
    reference_text = _sanitize_for_ragas(reference_text)
    retrieved_contexts = [_sanitize_for_ragas(c) for c in retrieved_contexts]

    ckey = hashlib.sha256(json.dumps(
        {"q": query, "r": response_text, "ref": reference_text, "ctx": retrieved_contexts},
        sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()
    with _ragas_cache_lock:
        if ckey in _ragas_eval_cache:
            return _ragas_eval_cache[ckey]

    sample = SingleTurnSample(
        user_input=query, response=response_text,
        reference=reference_text, retrieved_contexts=retrieved_contexts,
    )

    async def _eval_all():
        results_list = await asyncio.gather(
            *[_score_one_metric(m, sample) for m in ALL_METRICS], return_exceptions=True
        )
        return {name: score for item in results_list
                if not isinstance(item, Exception) for name, score in [item]}

    with _RAGAS_THROTTLE:
        try:
            scores = asyncio.run_coroutine_threadsafe(_eval_all(), _RAGAS_LOOP).result(timeout=300)
        except Exception as e:
            logger.warning("RAGAS submit error: %s", e)
            scores = {}

    mapped = {METRIC_KEY_MAP[m.name]: scores.get(m.name) for m in ALL_METRICS}
    with _ragas_cache_lock:
        _ragas_eval_cache[ckey] = mapped
    return mapped


# ===========================================================================
# Ground-truth comparison utilities (with explicit N/A branching)
# ===========================================================================
_NA_STRINGS = {
    "not applicable", "not application",
    "n/a", "na", "n.a.", "n.a", "n. a.", "n. a",
    "nil", "-", "—", "–", "−", "*", "#", "",
}


def _canonicalize_unit(unit_str):
    """Character-level canonicalisation only (no semantic unit normalisation)."""
    if not isinstance(unit_str, str):
        return str(unit_str).strip().lower()
    s = unit_str.strip()
    for ch in ("\u2018", "\u2019", "\u02BC", "\u02B9", "\u0060", "\u00B4"):
        s = s.replace(ch, "'")
    for ch in ("\u2013", "\u2014", "\u2212"):
        s = s.replace(ch, "-")
    return " ".join(s.split()).lower()


def _is_ground_truth_na(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned in _NA_STRINGS or re.sub(r"\s+", "", cleaned) in _NA_STRINGS
    return False


# ===========================================================================
# Per-query pipeline: retrieve -> extract -> compare -> RAGAS
# ===========================================================================
def process_single_query(row, strategy):
    bank_name, year = row["bank_name"], row["year"]
    template, item = row["template"], row["item"]
    expected_unit = row.get("unit", "")
    ground_truth = row.get("ground_truth_value", 0.0)
    gt_is_na = _is_ground_truth_na(ground_truth)

    query = build_query(template, item, year)
    t0 = time.time()

    try:
        top_chunks, _ = search(bank_name, query, strategy=strategy, top_k=3)
    except Exception as e:
        logger.warning("[skip] %s — retrieval failed: %s", bank_name, e)
        return None
    if not top_chunks:
        logger.warning("[skip] %s — no chunks retrieved", bank_name)
        return None

    extracted_data = extract_metrics_from_top_chunks(
        top_chunks, bank_name, query, year=year, template=template, target_item=item,
    )

    extracted_value, extracted_unit, extracted_is_na = None, "", False
    is_value_correct = is_unit_correct = is_fully_correct = False
    response_text = "No data extracted."

    if extracted_data and extracted_data.get("data"):
        first = extracted_data["data"][0]
        extracted_value = first.get("value_1")
        extracted_unit = extracted_data.get("unit", "")
        extracted_is_na = not first.get("is_applicable", True)

        # Safety net for omitted is_applicable flags.
        if not extracted_is_na:
            na_signals = {"n/a", "not applicable", "not application",
                          "nil", "n.a.", "n.a", "—", "–", "-"}
            item_text = first.get("item", "").strip().lower()
            unit_text = str(extracted_unit).strip().lower()
            if (item_text in na_signals or unit_text in na_signals
                    or item_text.endswith("(not applicable)") or item_text.endswith("(n/a)")):
                extracted_is_na = True

        response_text = json.dumps(extracted_data)

        if gt_is_na:
            is_value_correct = extracted_is_na
            is_unit_correct = is_fully_correct = is_value_correct
        elif not extracted_is_na:
            try:
                is_value_correct = (extracted_value is not None
                                    and float(extracted_value) == float(ground_truth))
            except (ValueError, TypeError):
                pass
            is_unit_correct = _canonicalize_unit(extracted_unit) == _canonicalize_unit(expected_unit)
            is_fully_correct = is_value_correct and is_unit_correct

    reference_text = (
        f"The item '{item}' is not applicable for this entity."
        if gt_is_na else f"The value is {ground_truth} {expected_unit}."
    )
    ragas_metrics = evaluate_with_ragas(query, response_text, reference_text, top_chunks)

    logger.info("[%s] %s | %s | correct=%s | %.1fs",
                strategy, bank_name, item, is_fully_correct, time.time() - t0)

    return {
        "bank_name": bank_name, "year": year, "template": template, "item": item,
        "query": query,
        "ground_truth_value": "N/A" if gt_is_na else ground_truth,
        "ground_truth_unit": "N/A" if gt_is_na else expected_unit,
        "extracted_value": "N/A" if extracted_is_na else extracted_value,
        "extracted_unit": "N/A" if extracted_is_na else extracted_unit,
        "gt_is_na": gt_is_na, "extracted_is_na": extracted_is_na,
        "is_value_correct": is_value_correct, "is_unit_correct": is_unit_correct,
        "is_fully_correct": is_fully_correct, "strategy": strategy,
        "retrieved_chunks_preview": " ||| ".join(c[:500] for c in top_chunks),
        **ragas_metrics,
    }


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies against ground truth.")
    parser.add_argument("--ground-truth", default="ground_truth.xlsx")
    parser.add_argument("--strategies", nargs="+", default=["sparse", "dense", "hybrid_rrf"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    gt_df = pd.read_excel(args.ground_truth, keep_default_na=False)
    logger.info("Loaded %d queries from %s", len(gt_df), args.ground_truth)

    for strategy in args.strategies:
        logger.info("=" * 60)
        logger.info("Running experiment: %s (workers=%d)", strategy.upper(), args.workers)
        rows = [row for _, row in gt_df.iterrows()]
        collected = {}

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_single_query, row, strategy): i
                for i, row in enumerate(rows)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        collected[idx] = result
                except Exception as e:
                    logger.error("Query %d raised: %s", idx, e)

        data = [collected[i] for i in sorted(collected)]
        if not data:
            logger.warning("No results for %s", strategy)
            continue

        correct = sum(1 for r in data if r["is_fully_correct"])
        na_total = sum(1 for r in data if r["gt_is_na"])
        na_correct = sum(1 for r in data if r["gt_is_na"] and r["is_fully_correct"])
        logger.info(">>> %s accuracy: %.2f%% (%d/%d) | numeric %d/%d | N/A %d/%d",
                    strategy.upper(), 100 * correct / len(data), correct, len(data),
                    correct - na_correct, len(data) - na_total, na_correct, na_total)

        out = f"Extraction_Results_{strategy}.xlsx"
        pd.DataFrame(data).to_excel(out, index=False)
        logger.info("Saved %s", out)


if __name__ == "__main__":
    main()