"""Offline ingestion pipeline (run locally, NOT shipped in the Docker image).

Pipeline: HKMA VPR scraping -> Mistral OCR -> chunking -> Gemini embeddings
-> ChromaDB persistence (batched, with per-bank metadata).

Usage:
    python scripts/ingest.py --ground-truth ground_truth.xlsx --links ov1_link.xlsx
"""
import argparse
import os
import sys
import tempfile
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai.client import Mistral
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MISTRAL_API_KEY,
    get_chroma_collection,
    get_embedding_model,
    get_logger,
    standardize_bank_name,
)

logger = get_logger("ingest")

EMBED_BATCH_SIZE = 100
BASE_URL = "https://vpr.hkma.gov.hk"

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# ---------------------------------------------------------------------------
# Step 1: Scrape disclosure PDF links from the HKMA VPR register
# ---------------------------------------------------------------------------
def create_driver():
    options = Options()
    for arg in (
        "--headless=new", "--no-sandbox", "--disable-dev-shm-usage",
        "--disable-gpu", "--dns-prefetch-disable",
        "--disable-features=VizDisplayCompositor",
    ):
        options.add_argument(arg)
    return webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()), options=options
    )


def get_bank_links():
    register_url = f"{BASE_URL}/eng/regulatory-resources/registers/register-of-ais-and-lros/"
    driver = create_driver()
    try:
        driver.get(register_url)
        time.sleep(3)
        clicks = 0
        while clicks < 50:
            try:
                load_more = driver.find_element(
                    By.XPATH,
                    "//a[contains(text(),'Load More')] | //button[contains(text(),'Load More')]",
                )
                if not load_more.is_displayed():
                    break
                driver.execute_script("arguments[0].click();", load_more)
                clicks += 1
                time.sleep(2)
            except Exception:
                break
        logger.info("Clicked 'Load More' %d time(s)", clicks)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = soup.select("#alphabet-result-list ul.list a")
        return [
            (l.get("title"), l.get("href"))
            for l in links if l.get("title") and l.get("href")
        ]
    finally:
        driver.quit()


def process_bank_requests(bank_data):
    bank_name, href = bank_data
    try:
        with requests.Session() as session:
            response = session.get(BASE_URL + href, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            pdf_tags = soup.select('a[href*="fd_int_0325"][href$=".pdf"]')
            pdf_urls = []
            for tag in pdf_tags:
                pdf_href = tag.get("href")
                if not pdf_href:
                    continue
                full = pdf_href if pdf_href.startswith("http") else BASE_URL + pdf_href
                if full not in pdf_urls:
                    pdf_urls.append(full)
            record = {"bank_name": bank_name, "has_disclosure": "Y" if pdf_urls else "N"}
            for i in range(3):
                record[f"link_{i + 1}"] = pdf_urls[i] if i < len(pdf_urls) else None
            return record
    except Exception:
        return {"bank_name": bank_name, "has_disclosure": "N",
                "link_1": None, "link_2": None, "link_3": None}


def get_disclosure_links_df(n_jobs=8) -> pd.DataFrame:
    logger.info("Fetching bank links from HKMA VPR register...")
    bank_data = get_bank_links()
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_bank_requests)(b) for b in bank_data
    )
    return pd.DataFrame([r for r in results if r is not None])


# ---------------------------------------------------------------------------
# Step 2: OCR via Mistral
# ---------------------------------------------------------------------------

def _mistral_ocr_with_retry(bank_name, tmp_path, max_retries=5):
    """Call Mistral OCR with exponential backoff on rate-limit (429) errors."""
    for attempt in range(max_retries):
        try:
            with open(tmp_path, "rb") as f:
                file_upload = mistral_client.files.upload(
                    file={"file_name": f"{bank_name}.pdf", "content": f}, purpose="ocr"
                )
            signed_url = mistral_client.files.get_signed_url(file_id=file_upload.id)
            return mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": signed_url.url},
                include_image_base64=False,
            )
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = min(2 ** attempt * 5, 60)  # 5s, 10s, 20s, 40s, 60s
                logger.warning("Rate limited for %s — retrying in %ds (attempt %d/%d)",
                               bank_name, wait, attempt + 1, max_retries)
                time.sleep(wait)
            else:
                raise
    return None

def extract_markdown_with_mistral(pdf_url, bank_name, pdf_label=None):
    tmp_path = None
    try:
        logger.info("Downloading: %s", pdf_label or pdf_url)
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        ocr_response = _mistral_ocr_with_retry(bank_name, tmp_path)
        if ocr_response is None:
            return None

        pages = [p.markdown.strip() for p in ocr_response.pages if p.markdown.strip()]
        # Politeness delay between OCR calls to stay under the rate limit.
        time.sleep(2)
        return {"source": pdf_label or pdf_url, "markdown": "\n\n".join(pages)}
    except Exception:
        logger.exception("OCR failed for %s [%s]", bank_name, pdf_label)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Step 3: Chunk, embed, and persist into ChromaDB
# ---------------------------------------------------------------------------
def ingest_bank(bank_name: str, pdf_urls: list[str]) -> int:
    """OCR all PDFs for one bank, chunk, embed, and upsert into Chroma.

    Existing chunks for the bank are deleted first, so re-ingestion is idempotent.
    Returns the number of chunks ingested.
    """
    std_bank = standardize_bank_name(bank_name)
    collection = get_chroma_collection()
    embedder = get_embedding_model()

    docs = []
    for idx, url in enumerate(pdf_urls):
        if url and str(url).lower() != "nan":
            doc = extract_markdown_with_mistral(url, bank_name, f"{std_bank}_pdf{idx + 1}")
            if doc:
                docs.append(doc)
    if not docs:
        logger.warning("No documents parsed for %s", bank_name)
        return 0

    ids, texts, metadatas = [], [], []
    for doc in docs:
        for i, chunk in enumerate(text_splitter.split_text(doc["markdown"])):
            ids.append(f"{std_bank}::{doc['source']}::chunk{i}")
            texts.append(chunk)
            metadatas.append({"bank": std_bank, "source": doc["source"]})

    # Idempotent re-ingestion: remove any previous version of this bank.
    collection.delete(where={"bank": std_bank})

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        end = start + EMBED_BATCH_SIZE
        batch_embeddings = embedder.embed_documents(texts[start:end])
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=batch_embeddings,
            metadatas=metadatas[start:end],
        )
        logger.info("  %s: added chunks %d-%d", std_bank, start, min(end, len(texts)))

    logger.info("Ingested %d chunks for %s", len(texts), std_bank)
    return len(texts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest HKMA Pillar 3 disclosures into ChromaDB.")
    parser.add_argument("--ground-truth", default="ground_truth.xlsx")
    parser.add_argument("--links", default="ov1_link.xlsx")
    args = parser.parse_args()

    gt_df = pd.read_excel(args.ground_truth, keep_default_na=False)
    gt_df["bank_std"] = gt_df["bank_name"].map(standardize_bank_name)
    banks_needed = set(gt_df["bank_std"].unique())
    logger.info("Ground truth requires %d unique banks", len(banks_needed))

    try:
        link_df = pd.read_excel(args.links)
    except FileNotFoundError:
        logger.info("%s not found — scraping HKMA register...", args.links)
        link_df = get_disclosure_links_df(n_jobs=8)
        link_df.to_excel(args.links, index=False)

    link_df["bank_std"] = link_df["bank_name"].map(standardize_bank_name)
    link_df = link_df[link_df["has_disclosure"] == "Y"]
    targets = link_df[link_df["bank_std"].isin(banks_needed)]
    logger.info("Found %d matching banks with disclosures", len(targets))

    total = 0
    for _, row in targets.iterrows():
        pdf_urls = [row.get(f"link_{i + 1}") for i in range(3) if pd.notna(row.get(f"link_{i + 1}"))]
        if pdf_urls:
            total += ingest_bank(row["bank_name"], pdf_urls)

    logger.info("Done. Total chunks in collection: %d (added/updated %d)",
                get_chroma_collection().count(), total)


if __name__ == "__main__":
    main()