"""LLM extraction layer: structured-output extraction from retrieved chunks.

Includes a thread-safe in-memory cache keyed on (query + chunk content) so
identical retrievals (e.g. strategies converging on the same chunks) skip
redundant LLM calls, plus a best-match reordering safety net for cases where
the model returns multiple line items despite being asked for one.
"""
import hashlib
import threading
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from app.config import get_llm, get_logger
from app.schemas import FinancialExtraction

logger = get_logger(__name__)

_extraction_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()

SYSTEM_PROMPT = (
    "You are an expert financial data extraction AI.\n"
    "IMPORTANT RULES:\n"
    "1. Extract ONLY the single line item that the user explicitly asks for.\n"
    "2. Return exactly ONE entry in the 'data' list — the requested item only.\n"
    "3. Do NOT return other rows from the same table.\n"
    "4. Identify the unit of measurement exactly as it appears in the source.\n"
    "5. Follow the JSON schema strictly.\n"
    "6. If a line item is marked in special value, such as 'N/A', 'Not applicable', "
    "'Not Application', 'N.A.', 'N.A', 'Nil', '—', '–', '-', '*', '#', or is blank / "
    "explicitly stated as not applicable in the document, "
    "set is_applicable to false and value_1 to 0.0.\n"
    "7. If a line item has a genuine numerical value of 0 or 0.0, "
    "set is_applicable to true and value_1 to 0.0.\n"
    "8. DISTINGUISH between 'value is zero' (is_applicable=true, value_1=0.0) "
    "and 'not applicable' (is_applicable=false, value_1=0.0)."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user",
     "Entity: {bank_name}\nYear: {year}\nTemplate: {template}\n"
     "Query: {query}\n\nContext:\n{context}"),
])


def _cache_key(query: str, chunks: list[str]) -> str:
    raw = query + "\n|||CHUNK|||\n".join(chunks)
    return hashlib.sha256(raw.encode()).hexdigest()


def _best_match_index(data_list: list[dict], target_item: str) -> int:
    """Find the entry in data_list whose item description best matches target_item."""
    target = target_item.lower().strip()
    best_idx, best_score = 0, -1

    for i, d in enumerate(data_list):
        candidate = d.get("item", "").lower().strip()
        if candidate == target:
            return i
        if target in candidate or candidate in target:
            return i
        overlap = len(set(target.split()) & set(candidate.split()))
        if overlap > best_score:
            best_score, best_idx = overlap, i

    return best_idx


def extract_metrics_from_top_chunks(
    chunks: list[str],
    bank_name: str,
    query: str,
    year: str,
    template: str,
    target_item: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[dict]:
    """Run structured extraction over the retrieved chunks. Returns a dict or None."""
    key = _cache_key(query, chunks)
    if use_cache:
        with _cache_lock:
            cached = _extraction_cache.get(key)
        if cached is not None:
            logger.info("Extraction cache hit for %s", bank_name)
            return cached

    chain = _PROMPT | get_llm().with_structured_output(FinancialExtraction)

    try:
        result = chain.invoke({
            "bank_name": bank_name,
            "year": year,
            "template": template,
            "query": query,
            "context": "\n\n".join(chunks),
        })
        if result is None:
            return None

        data = result.model_dump()

        # Safety net: if the LLM returned multiple items, promote the best match.
        if target_item and data.get("data") and len(data["data"]) > 1:
            best = _best_match_index(data["data"], target_item)
            if best != 0:
                logger.info(
                    "Reordered extraction output: picked '%s' for target '%s'",
                    data["data"][best].get("item", ""), target_item,
                )
                data["data"].insert(0, data["data"].pop(best))

        if use_cache:
            with _cache_lock:
                _extraction_cache[key] = data
        return data

    except Exception:
        logger.exception("Extraction failed for %s", bank_name)
        return None