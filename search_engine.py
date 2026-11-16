"""Trzy metody wyszukiwania: słowa kluczowe, TF-IDF i semantyczne.

Każda metoda zwraca listę SearchResult (dokument, wynik, fragment tekstu).
allowed_ids ogranicza kandydatów, np. do dokumentów zawierających daną encję.
"""
import html
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from document_store import Document, DocumentStore
from nlp_pl import lemmatize_query

# model wielojęzyczny - dobrze radzi sobie z polskim (all-MiniLM-L6-v2 zna tylko angielski)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SNIPPET_CHARS = 320


@dataclass
class SearchResult:
    doc_id: str
    doc_name: str
    score: float
    snippet: str      # HTML (wyróżnione słowa w <mark>)
    details: str = ""


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    # normalizacja do długości 1 - wtedy iloczyn skalarny = podobieństwo cosinusowe
    return get_embedding_model().encode(texts, batch_size=32, normalize_embeddings=True)


def parse_query(query: str) -> list[str]:
    """Frazy w cudzysłowie traktuję jako całość, resztę dzielę na słowa."""
    phrases = [p.strip().lower() for p in re.findall(r'"([^"]+)"', query) if p.strip()]
    rest = re.sub(r'"[^"]*"', " ", query)
    words = [w.lower() for w in re.findall(r"\w+", rest) if len(w) > 1]
    return phrases + words


def term_pattern(term: str) -> re.Pattern:
    # dokładne słowo/fraza, bez względu na wielkość liter; spacje w frazie = dowolne białe znaki
    body = r"\s+".join(re.escape(part) for part in term.split())
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def make_snippet(text: str, patterns: list[re.Pattern] | None = None) -> str:
    """Fragment do pokazania: przycięty i z wyróżnionymi trafieniami."""
    text = " ".join(text.split())
    start = 0
    if patterns:
        first = [m.start() for p in patterns if (m := p.search(text))]
        if first:
            start = max(0, min(first) - 100)
    piece = text[start:start + SNIPPET_CHARS]
    out = html.escape(piece)
    for p in patterns or []:
        out = p.sub(lambda m: f"<mark>{m.group()}</mark>", out)
    return ("..." if start > 0 else "") + out + ("..." if start + SNIPPET_CHARS < len(text) else "")


class SearchEngine:
    def __init__(self, store: DocumentStore):
        self.store = store
        self._tfidf_ids = None
        self._vectorizer = None
        self._matrix = None

    def _candidates(self, allowed_ids) -> list[Document]:
        docs = self.store.get_all()
        if allowed_ids is not None:
            docs = [d for d in docs if d.id in allowed_ids]
        return docs

    # --- 1. słowa kluczowe ---------------------------------------------------
    def keyword_search(self, query: str, top_k: int = 5, allowed_ids=None) -> list[SearchResult]:
        """Dokumenty zawierające słowa/frazy z zapytania (dokładne dopasowanie, bez odmiany).
        Ranking: najpierw liczba różnych dopasowanych słów, potem łączna liczba wystąpień."""
        terms = parse_query(query)
        if not terms:
            return []
        patterns = [term_pattern(t) for t in terms]
        scored = []
        for doc in self._candidates(allowed_ids):
            counts = [len(p.findall(doc.content)) for p in patterns]
            matched = sum(c > 0 for c in counts)
            if matched == 0:
                continue
            # fragment z największą liczbą trafień
            best = max(range(len(doc.chunks)),
                       key=lambda i: sum(len(p.findall(doc.chunk_text(i))) for p in patterns))
            scored.append(((matched, sum(counts)), SearchResult(
                doc.id, doc.name, float(sum(counts)),
                make_snippet(doc.chunk_text(best), patterns),
                f"słowa: {matched}/{len(terms)}, wystąpienia: {sum(counts)}")))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    # --- 2. TF-IDF -----------------------------------------------------------
    def _ensure_tfidf(self):
        ids = sorted(self.store.docs)
        if ids == self._tfidf_ids:
            return
        docs = [self.store.get(i) for i in ids]
        # dokumenty są już zlematyzowane i bez stopwords (nlp_pl.analyze_text)
        self._vectorizer = TfidfVectorizer(tokenizer=str.split, token_pattern=None,
                                           lowercase=False, sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform([d.lemmas for d in docs])
        self._tfidf_ids = ids

    def tfidf_search(self, query: str, top_k: int = 5, allowed_ids=None) -> list[SearchResult]:
        if not self.store.docs:
            return []
        self._ensure_tfidf()
        q_lemmas = lemmatize_query(query)
        if not q_lemmas:
            return []
        q_vec = self._vectorizer.transform([" ".join(q_lemmas)])
        scores = cosine_similarity(q_vec, self._matrix).ravel()
        allowed = set(allowed_ids) if allowed_ids is not None else None
        results = []
        for idx in np.argsort(-scores):
            if scores[idx] <= 0 or len(results) == top_k:
                break
            doc = self.store.get(self._tfidf_ids[idx])
            if allowed is not None and doc.id not in allowed:
                continue
            # snippet: fragment, w którym jest najwięcej lematów z zapytania
            q_set = set(q_lemmas)
            best = max(range(len(doc.chunks)),
                       key=lambda i: sum(l in q_set for l in doc.chunk_lemmas[i].split()))
            results.append(SearchResult(doc.id, doc.name, round(float(scores[idx]), 4),
                                        make_snippet(doc.chunk_text(best)),
                                        "lematy zapytania: " + ", ".join(q_lemmas)))
        return results

    # --- 3. semantyczne ------------------------------------------------------
    def semantic_search(self, query: str, top_k: int = 5, allowed_ids=None) -> list[SearchResult]:
        """Każdy dokument jest podzielony na fragmenty; wynik dokumentu to
        największe podobieństwo cosinusowe zapytania do któregoś fragmentu."""
        docs = self._candidates(allowed_ids)
        if not docs or not query.strip():
            return []
        q = embed([query])[0]
        scored = []
        for doc in docs:
            sims = self.store.embeddings(doc.id) @ q
            best = int(np.argmax(sims))
            scored.append(SearchResult(doc.id, doc.name, round(float(sims[best]), 4),
                                       make_snippet(doc.chunk_text(best)),
                                       f"najlepszy fragment nr {best + 1} z {len(doc.chunks)}"))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def search_all(self, query: str, top_k: int = 5, allowed_ids=None) -> dict:
        return {
            "Słowa kluczowe": self.keyword_search(query, top_k, allowed_ids),
            "TF-IDF": self.tfidf_search(query, top_k, allowed_ids),
            "Semantyczne": self.semantic_search(query, top_k, allowed_ids),
        }

    def top_terms(self, n: int = 20) -> list[tuple[str, int]]:
        """Najczęstsze lematy w całym korpusie (po usunięciu stopwords)."""
        counts = Counter(l for d in self.store.get_all() for l in d.lemmas.split())
        return counts.most_common(n)
