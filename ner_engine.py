"""Operacje na encjach. Same encje liczy nlp_pl.analyze_text raz, przy dodawaniu
dokumentu, i są zapisane w bazie (Document.entities) - tu tylko z nich korzystam."""
import html

import pandas as pd

from document_store import Document, DocumentStore
from nlp_pl import COLORS, LABELS, normalize_entity

CONTEXT_CHARS = 120


def label_name(label: str) -> str:
    return LABELS.get(label, label)


def highlight_html(doc: Document, max_chars: int | None = None) -> str:
    """Treść dokumentu jako HTML z encjami w <mark> (kolor = typ encji)."""
    text = doc.content if max_chars is None else doc.content[:max_chars]
    parts, last = [], 0
    for e in sorted(doc.entities, key=lambda e: e["start"]):
        if e["start"] < last or e["end"] > len(text):
            continue
        parts.append(html.escape(text[last:e["start"]]))
        parts.append(
            f'<mark style="background:{COLORS.get(e["label"], "#ddd")};padding:0 3px;'
            f'border-radius:3px" title="{label_name(e["label"])}">'
            f'{html.escape(text[e["start"]:e["end"]])}'
            f'<sub style="font-size:0.65em"> {label_name(e["label"])}</sub></mark>')
        last = e["end"]
    parts.append(html.escape(text[last:]))
    return "".join(parts)


def context(doc: Document, ent: dict) -> str:
    s = max(0, ent["start"] - CONTEXT_CHARS)
    e = min(len(doc.content), ent["end"] + CONTEXT_CHARS)
    before = html.escape(doc.content[s:ent["start"]])
    after = html.escape(doc.content[ent["end"]:e])
    mid = html.escape(doc.content[ent["start"]:ent["end"]])
    return " ".join(f"...{before}<b>{mid}</b>{after}...".split())


def entities_table(store: DocumentStore) -> pd.DataFrame:
    """Wszystkie wystąpienia encji w bazie jako tabela (jeden wiersz = jedno wystąpienie)."""
    rows = []
    for doc in store.get_all():
        for i, e in enumerate(doc.entities):
            rows.append({"doc_id": doc.id, "dokument": doc.name, "encja": e["text"],
                         "lemat": e["lemma"].lower(), "forma_podst": e["lemma"], "label": e["label"],
                         "typ": label_name(e["label"]), "idx": i})
    return pd.DataFrame(rows, columns=["doc_id", "dokument", "encja", "lemat", "forma_podst",
                                       "label", "typ", "idx"])


def top_entities(table: pd.DataFrame, labels, n: int = 10) -> pd.DataFrame:
    """Najczęstsze encje danego typu (grupowane po lemacie, żeby 'Krakowa' i
    'Kraków' liczyły się razem). Liczę wystąpienia i liczbę dokumentów."""
    part = table[table["label"].isin(labels)]
    if part.empty:
        return pd.DataFrame(columns=["encja", "wystąpienia", "dokumenty"])
    grouped = part.groupby("lemat").agg(
        # do wyświetlenia biorę najczęstszą formę podstawową (lemat z wielkimi literami)
        encja=("forma_podst", lambda s: s.mode().iloc[0]),
        wystąpienia=("encja", "size"), dokumenty=("doc_id", "nunique"))
    return grouped.sort_values(["wystąpienia", "dokumenty"], ascending=False).head(n) \
        .reset_index(drop=True)


def match_entities(table: pd.DataFrame, query: str = "", label: str | None = None) -> pd.DataFrame:
    """Wystąpienia pasujące do zapytania (po lemacie albo tekście) i/lub typu."""
    part = table
    if label:
        part = part[part["label"] == label]
    if query.strip():
        q_lemma = normalize_entity(query)
        q_text = query.strip().lower()
        mask = part["lemat"].str.contains(q_lemma, regex=False) | \
            part["encja"].str.lower().str.contains(q_text, regex=False)
        part = part[mask]
    return part


def docs_with_entity(table: pd.DataFrame, query: str = "", label: str | None = None) -> set[str]:
    """Zbiór id dokumentów do filtra w wyszukiwarce."""
    return set(match_entities(table, query, label)["doc_id"])
