"""
Build the TF-IDF retrieval index from the scraped catalog JSON.
Run after save_catalog.py (or fetch_all_catalog.py):

    python scripts/build_index.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CATALOG_PATH = ROOT / "data" / "catalog.json"
INDEX_PATH = ROOT / "data" / "tfidf_index.pkl"


def main() -> None:
    if not CATALOG_PATH.exists():
        print(f"Catalog not found at {CATALOG_PATH}")
        print("Run scripts/save_catalog.py first.")
        return

    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog: list[dict] = json.load(f)

    print(f"Loaded {len(catalog)} assessments.")

    # Import retriever and trigger index build + save
    from app.retrieval import CatalogRetriever, _build_document
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [_build_document(item) for item in catalog]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        analyzer="word",
        token_pattern=r"(?u)\b\w[\w.#+\-]*\b",
    )
    matrix = vectorizer.fit_transform(docs)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)

    print(f"Saved TF-IDF index ({matrix.shape}) to {INDEX_PATH}")

    # Also verify BM25
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built for {len(tokenized)} documents.")
    except ImportError:
        print("rank-bm25 not installed — BM25 will be skipped at runtime.")


if __name__ == "__main__":
    main()
