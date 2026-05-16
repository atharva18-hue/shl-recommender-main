"""
Hybrid retrieval over the SHL assessment catalog.

Strategy:
  1. TF-IDF cosine similarity on enriched text (name + types + description).
  2. BM25 keyword match re-ranking.
  3. Exact / partial name lookup for comparison queries.

No external model downloads needed — all computation is in scikit-learn + rank-bm25.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"
CATALOG_PATH = DATA_DIR / "catalog.json"
INDEX_PATH = DATA_DIR / "tfidf_index.pkl"

TEST_TYPE_LABELS: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# Synonyms: expand user terms into catalog vocabulary
SYNONYM_MAP: dict[str, str] = {
    "personality": "personality behavior OPQ",
    "cognitive": "ability aptitude reasoning verbal numerical",
    "aptitude": "ability aptitude reasoning",
    "coding": "programming software developer automata simulation",
    "programming": "programming software developer knowledge skills",
    "situational judgment": "biodata situational judgement scenarios",
    "sjt": "biodata situational judgement scenarios",
    "leadership": "management scenarios executive OPQ leadership competency",
    "360": "360 development multi-rater feedback MFS",
    "simulation": "simulation automata",
    "verbal": "verbal ability reasoning reading comprehension",
    "numerical": "numerical ability reasoning",
    "inductive": "inductive reasoning ability",
    "deductive": "deductive reasoning ability",
    "java": "java programming knowledge skills",
    "python": "python programming knowledge skills",
    "data science": "data science python R statistics machine learning",
    "machine learning": "data science python statistics AI",
    "devops": "docker kubernetes jenkins devops cloud",
    "frontend": "javascript react angular html css frontend",
    "backend": "java python node spring backend server",
    "fullstack": "javascript java python react angular node fullstack",
    "sales": "sales OPQ personality transformation MQ",
    "customer service": "customer service simulation phone behavioral contact center",
    "contact center": "customer service phone simulation contact center SVAR spoken",
    "contact centre": "customer service phone simulation contact center SVAR spoken",
    "call center": "customer service phone simulation contact center SVAR spoken",
    "call centre": "customer service phone simulation contact center SVAR spoken",
    "hr": "human resources OPQ personality competency",
    "finance": "financial accounting banking economics numerical",
    "project management": "project management competency ability",
    "remote": "remote testing",
    "graduate": "graduate scenarios entry level cognitive ability",
    "executive": "executive scenarios leadership enterprise OPQ UCF",
    "cxo": "executive leadership OPQ personality director C-suite",
    "manager": "management scenarios managerial leadership OPQ",
    "mid": "mid-professional",
    "senior": "professional individual contributor advanced verify",
    "entry": "entry level graduate",
    "rust": "programming software developer live coding systems",
    "plant operator": "safety dependability industrial manufacturing",
    "healthcare": "HIPAA medical terminology health dependability",
    "admin": "MS Office word excel administrative",
    "restructuring": "skills assessment development GSA OPQ",
    "reskilling": "skills assessment development GSA",
    "talent audit": "skills global assessment OPQ personality",
    "safety": "safety dependability instrument industrial DSI",
}


def _expand_query(query: str) -> str:
    """Expand known synonyms / domain terms in the query."""
    q = query.lower()
    expansions = []
    for term, expansion in SYNONYM_MAP.items():
        if term in q:
            expansions.append(expansion)
    if expansions:
        return query + " " + " ".join(expansions)
    return query


def _build_document(item: dict) -> str:
    """Create a rich text document for an assessment (used by TF-IDF)."""
    parts = [item.get("name", "")]

    codes = item.get("test_types", [])
    labels = [TEST_TYPE_LABELS.get(c, c) for c in codes]
    if labels:
        parts.append(" ".join(labels))
        parts.append(" ".join(codes))

    desc = item.get("description", "")
    if desc:
        parts.append(desc)

    job_levels = item.get("job_levels", "")
    if job_levels:
        parts.append(job_levels)

    if item.get("remote_testing"):
        parts.append("remote testing")
    if item.get("adaptive_irt"):
        parts.append("adaptive IRT")

    return " ".join(filter(None, parts))


class CatalogRetriever:
    """Loads the catalog at startup and serves semantic + keyword search."""

    def __init__(self) -> None:
        self._catalog: list[dict] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None          # sparse (n_docs, vocab)
        self._bm25 = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return

        if not CATALOG_PATH.exists():
            print("WARNING: catalog.json not found — retrieval disabled.")
            self._loaded = True
            return

        with open(CATALOG_PATH, encoding="utf-8") as f:
            self._catalog = json.load(f)

        # Try loading a pre-built index
        if INDEX_PATH.exists():
            print("Loading pre-built TF-IDF index...")
            with open(INDEX_PATH, "rb") as f:
                state = pickle.load(f)
            self._vectorizer = state["vectorizer"]
            self._tfidf_matrix = state["matrix"]
        else:
            print("Building TF-IDF index from catalog...")
            self._build_tfidf()
            self._save_index()

        # Build BM25 index (in-memory, fast)
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [_build_document(item).lower().split() for item in self._catalog]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            self._bm25 = None

        print(f"Retriever ready: {len(self._catalog)} assessments.")
        self._loaded = True

    def _build_tfidf(self) -> None:
        docs = [_build_document(item) for item in self._catalog]
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            analyzer="word",
            token_pattern=r"(?u)\b\w[\w.#+\-]*\b",
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(docs)

    def _save_index(self) -> None:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(
                {"vectorizer": self._vectorizer, "matrix": self._tfidf_matrix}, f
            )
        print(f"Saved TF-IDF index to {INDEX_PATH}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 10,
        filter_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Hybrid search: TF-IDF cosine + BM25 re-ranking.

        Parameters
        ----------
        query : str  Natural-language description of what the user needs.
        k : int      Max results to return.
        filter_types : optional list of test-type codes (A/B/C/D/E/K/P/S).
        """
        if not self._loaded:
            self.load()
        if not self._catalog:
            return []

        expanded = _expand_query(query)

        # --- TF-IDF scores ---
        q_vec = self._vectorizer.transform([expanded])
        tfidf_scores = cosine_similarity(q_vec, self._tfidf_matrix)[0]

        # --- BM25 scores (normalised) ---
        bm25_scores = np.zeros(len(self._catalog))
        if self._bm25 is not None:
            raw_bm25 = np.array(
                self._bm25.get_scores(expanded.lower().split()), dtype=np.float32
            )
            max_b = raw_bm25.max()
            if max_b > 0:
                bm25_scores = raw_bm25 / max_b

        # Hybrid: weighted sum (TF-IDF dominant, BM25 as signal)
        combined = 0.6 * tfidf_scores + 0.4 * bm25_scores

        # Sort by combined score
        ranked_indices = np.argsort(-combined)

        results: list[dict] = []
        for idx in ranked_indices:
            item = self._catalog[int(idx)].copy()
            item["_score"] = float(combined[idx])

            if filter_types:
                item_types = item.get("test_types", [])
                if not any(t in filter_types for t in item_types):
                    continue

            if item["_score"] > 0:
                results.append(item)

            if len(results) >= k:
                break

        # If nothing scored, fall back to top items
        if not results:
            results = [self._catalog[int(i)].copy() for i in ranked_indices[:k]]

        return results

    def get_by_name(self, name: str) -> Optional[dict]:
        """Find an assessment by exact or partial name match."""
        name_lower = name.strip().lower()
        # Exact match first
        for item in self._catalog:
            if item["name"].lower() == name_lower:
                return item
        # Partial match
        for item in self._catalog:
            if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
                return item
        return None

    def format_for_prompt(self, items: list[dict], max_desc_len: int = 120) -> str:
        """Compact, human-readable catalog snippet for injection into LLM prompts."""
        lines: list[str] = []
        for item in items:
            codes = item.get("test_types", [])
            labels = [TEST_TYPE_LABELS.get(c, c) for c in codes]
            type_str = " | ".join(labels) if labels else "Unknown"
            dur = item.get("duration_minutes")
            dur_str = f", ~{dur} min" if dur else ""
            remote = " [Remote]" if item.get("remote_testing") else ""
            adaptive = " [Adaptive]" if item.get("adaptive_irt") else ""
            desc = item.get("description", "")
            desc_short = (desc[:max_desc_len] + "…") if len(desc) > max_desc_len else desc
            lvl = item.get("job_levels", "")
            lvl_str = f" | {lvl}" if lvl else ""
            lines.append(
                f"• {item['name']} [{' '.join(codes)}]{remote}{adaptive}{dur_str}\n"
                f"  URL: {item['url']}\n"
                f"  Type: {type_str}{lvl_str}\n"
                f"  {desc_short}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def catalog(self) -> list[dict]:
        return self._catalog

    def all_names_and_urls(self) -> list[tuple[str, str]]:
        return [(item["name"], item["url"]) for item in self._catalog]


# Singleton — loaded once at server startup
retriever = CatalogRetriever()
