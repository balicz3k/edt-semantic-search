"""Przetwarzanie tekstu po polsku: lematyzacja, stopwords, NER, podział na fragmenty.

Jedno przejście spaCy (pl_core_news_md) na dokument daje naraz lematy do TF-IDF
i encje do NER - wynik zapisuję w bazie, żeby nie liczyć tego przy każdym zapytaniu.
"""
import re
from functools import lru_cache

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from spacy.lang.pl.stop_words import STOP_WORDS as PL_STOP  # noqa: F401 (używane też w search_engine)

SPACY_MODEL = "pl_core_news_md"
MAX_SPACY_CHARS = 100_000   # dłuższy tekst dzielę na części (limit pamięci spaCy)
CHUNK_CHARS = 600           # docelowa długość fragmentu do embeddingów i snippetów

# etykiety modelu pl_core_news (korpus NKJP) -> nazwy w GUI
LABELS = {
    "persName": "Osoba",
    "orgName": "Organizacja",
    "placeName": "Miejscowość / państwo",
    "geogName": "Obiekt geograficzny",
    "date": "Data",
    "time": "Czas",
}
COLORS = {
    "persName": "#ffd166", "orgName": "#95d5b2", "placeName": "#8ecae6",
    "geogName": "#bde0fe", "date": "#ffafcc", "time": "#e5c3ff",
}
# grupy do panelu najczęstszych encji
LOCATION_LABELS = ("placeName", "geogName")

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9])")


@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as e:
        raise RuntimeError(
            f"Brak modelu spaCy. Uruchom: python -m spacy download {SPACY_MODEL}") from e


def detect_language(text: str) -> str:
    """Prosty detektor: który zbiór stopwords pokrywa więcej słów tekstu."""
    words = re.findall(r"\w+", text[:20_000].lower())
    if not words:
        return "?"
    pl = sum(w in PL_STOP for w in words)
    en = sum(w in EN_STOP for w in words)
    if max(pl, en) < 0.05 * len(words):
        return "?"
    return "pl" if pl >= en else "en"


def make_chunks(text: str) -> list[tuple[int, int]]:
    """Dzieli tekst na fragmenty ok. CHUNK_CHARS znaków: najpierw po akapitach,
    zbyt długie akapity dalej po zdaniach. Zwraca pary (początek, koniec)."""
    chunks = []
    for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):  # akapity
        start, para = m.start(), m.group()
        if len(para.strip()) < 40:   # nagłówki sekcji doklejam do następnego fragmentu
            continue
        if len(para) <= CHUNK_CHARS * 1.5:
            chunks.append((start, m.end()))
            continue
        # granice zdań wewnątrz akapitu; skleję zdania do ok. CHUNK_CHARS znaków
        bounds = [start + s.end() for s in SENTENCE_END.finditer(para)] + [m.end()]
        cur = start
        for b in bounds:
            if b - cur >= CHUNK_CHARS:
                chunks.append((cur, b))
                cur = b
        if m.end() - cur > 40:
            chunks.append((cur, m.end()))
        elif cur < m.end():
            chunks[-1] = (chunks[-1][0], m.end())  # krótka końcówka do poprzedniego
    return chunks or [(0, len(text))]


def is_content_token(tok) -> bool:
    return tok.is_alpha and not tok.is_stop and len(tok) > 1


def analyze_text(text: str) -> dict:
    """Pełne przetworzenie dokumentu: lematy (całość i per fragment) oraz encje."""
    nlp = get_nlp()
    chunks = make_chunks(text)
    chunk_lemmas = [[] for _ in chunks]
    lemmas, entities = [], []
    ci = 0
    for offset in range(0, len(text), MAX_SPACY_CHARS):
        doc = nlp(text[offset:offset + MAX_SPACY_CHARS])
        for tok in doc:
            if not is_content_token(tok):
                continue
            lemma = tok.lemma_.lower()
            lemmas.append(lemma)
            pos = offset + tok.idx
            while ci < len(chunks) - 1 and pos >= chunks[ci][1]:
                ci += 1
            if chunks[ci][0] <= pos < chunks[ci][1]:
                chunk_lemmas[ci].append(lemma)
        for ent in doc.ents:
            if ent.label_ not in LABELS:
                continue
            entities.append({
                "text": ent.text,
                "lemma": " ".join(t.lemma_ for t in ent).strip(),
                "label": ent.label_,
                "start": offset + ent.start_char,
                "end": offset + ent.end_char,
            })
    return {
        "lemmas": " ".join(lemmas),
        "chunks": chunks,
        "chunk_lemmas": [" ".join(c) for c in chunk_lemmas],
        "entities": entities,
    }


def lemmatize_query(query: str) -> list[str]:
    """Zapytanie przechodzi przez ten sam preprocessing co dokumenty."""
    return [t.lemma_.lower() for t in get_nlp()(query) if is_content_token(t)]


def normalize_entity(text: str) -> str:
    """Lemat nazwy do wyszukiwania encji: 'Krakowie' -> 'kraków'."""
    return " ".join(t.lemma_ for t in get_nlp()(text)).lower().strip()
