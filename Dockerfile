FROM python:3.12-slim

WORKDIR /srv

# Install runtime dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data at build time so the container starts instantly.
RUN python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab', 'stopwords')]"

# Copy application code only — scripts/ is intentionally excluded.
COPY app/ ./app/

# The persisted vector store and BM25 indices are provided as volumes:
#   docker run -v ./chroma_db:/srv/chroma_db -v ./bm25_index:/srv/bm25_index ...
ENV CHROMA_DIR=/srv/chroma_db \
    BM25_DIR=/srv/bm25_index

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]