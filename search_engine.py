"""3 metody wyszukiwania dokumentów: słowa kluczowe, TF-IDF, semantyczne."""
import re
from dataclasses import dataclass
from pathlib import Path

from document_store import DocumentStore, Document


@dataclass
class SearchResult:
    doc_id: str
    doc_name: str
    score: float
    snippet: str  # ~200-char fragment z trafieniem


def _make_snippet(content: str, query: str, snippet_len: int = 200) -> str:
    """Wytnij fragment tekstu wokół pierwszego trafienia."""
    lower_content = content.lower()
    lower_query = query.lower()
    words = lower_query.split()
    pos = -1
    for word in words:
        pos = lower_content.find(word)
        if pos >= 0:
            break
    if pos < 0:
        return content[:snippet_len].strip() + "..."
    start = max(0, pos - 80)
    end = min(len(content), start + snippet_len)
    snippet = content[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet


class SearchEngine:
    def __init__(self, document_store: DocumentStore, chroma_dir: str = "storage/chroma"):
        self._store = document_store
        self._chroma_dir = Path(chroma_dir)
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._tfidf_doc_ids: list[str] = []
        self._chroma_client = None
        self._chroma_collection = None
        self._sentence_model = None
        self._indexed_doc_ids: list[str] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def keyword_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        docs = self._store.get_all()
        if not docs:
            return []
        query_words = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 1]
        if not query_words:
            return []

        results: list[SearchResult] = []
        for doc in docs:
            lower = doc.content.lower()
            hits = sum(lower.count(w) for w in query_words)
            if hits > 0:
                # Normalizuj przez długość dokumentu (na 1000 znaków)
                score = hits / max(len(doc.content) / 1000, 1)
                snippet = _make_snippet(doc.content, query)
                results.append(SearchResult(doc.id, doc.name, round(score, 4), snippet))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def tfidf_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        docs = self._store.get_all()
        if not docs:
            return []

        # Sprawdź czy potrzeba reindeksowania
        current_ids = sorted(d.id for d in docs)
        if self._tfidf_matrix is None or sorted(self._tfidf_doc_ids) != current_ids:
            self._build_tfidf(docs)

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        query_vec = self._tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        results: list[SearchResult] = []
        for idx, score in enumerate(scores):
            if score > 0:
                doc_id = self._tfidf_doc_ids[idx]
                doc = self._store.get_by_id(doc_id)
                if doc:
                    snippet = _make_snippet(doc.content, query)
                    results.append(SearchResult(doc_id, doc.name, round(float(score), 4), snippet))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def semantic_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        docs = self._store.get_all()
        if not docs:
            return []

        self._ensure_sentence_model()
        self._ensure_chroma(docs)

        query_embedding = self._sentence_model.encode([query])[0].tolist()

        n_results = min(top_k, len(docs))
        try:
            chroma_results = self._chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["distances", "metadatas"],
            )
        except Exception:
            return []

        results: list[SearchResult] = []
        ids = chroma_results.get("ids", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]

        for doc_id, distance in zip(ids, distances):
            doc = self._store.get_by_id(doc_id)
            if doc:
                # Przekształć odległość cosinusową na podobieństwo [0,1]
                score = max(0.0, 1.0 - distance / 2.0)
                snippet = _make_snippet(doc.content, query)
                results.append(SearchResult(doc_id, doc.name, round(score, 4), snippet))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def reindex(self) -> None:
        """Przebuduj wszystkie indeksy po zmianie dokumentów."""
        docs = self._store.get_all()
        if not docs:
            self._tfidf_matrix = None
            self._tfidf_doc_ids = []
            self._chroma_collection = None
            return

        self._build_tfidf(docs)
        self._rebuild_chroma(docs)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _build_tfidf(self, docs: list[Document]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        corpus = [d.content for d in docs]
        self._tfidf_doc_ids = [d.id for d in docs]
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

    def _ensure_sentence_model(self) -> None:
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _ensure_chroma(self, docs: list[Document]) -> None:
        current_ids = sorted(d.id for d in docs)
        if self._chroma_collection is not None and sorted(self._indexed_doc_ids) == current_ids:
            return
        self._rebuild_chroma(docs)

    def _rebuild_chroma(self, docs: list[Document]) -> None:
        import chromadb

        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=str(self._chroma_dir))

        # Usuń i odtwórz kolekcję
        try:
            self._chroma_client.delete_collection("documents")
        except Exception:
            pass

        self._chroma_collection = self._chroma_client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

        self._ensure_sentence_model()
        texts = [d.content for d in docs]
        embeddings = self._sentence_model.encode(texts).tolist()

        self._chroma_collection.add(
            ids=[d.id for d in docs],
            embeddings=embeddings,
            metadatas=[{"name": d.name} for d in docs],
        )
        self._indexed_doc_ids = [d.id for d in docs]
