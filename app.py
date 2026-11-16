"""System wyszukiwania semantycznego i ekstrakcji wiedzy - GUI w Streamlit.

Uruchomienie: streamlit run app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import ner_engine as ner
from document_store import DocumentStore
from nlp_pl import LABELS, LOCATION_LABELS
from ranking_compare import pairwise_stats, positions_table
from search_engine import SearchEngine, embed

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_docs"
METHODS = ["Słowa kluczowe", "TF-IDF", "Semantyczne"]
METHOD_INFO = {
    "Słowa kluczowe": "dokładne słowa/frazy w cudzysłowie, ranking: liczba dopasowanych słów, potem wystąpień",
    "TF-IDF": "lematy + stopwords PL, podobieństwo cosinusowe wektorów TF-IDF",
    "Semantyczne": "embeddingi fragmentów (multilingual-e5-small), wynik = max cosinus",
}

st.set_page_config(page_title="Wyszukiwanie semantyczne i NER", layout="wide")


@st.cache_resource
def get_store():
    return DocumentStore(ROOT / "storage")


@st.cache_resource
def get_engine(_store):
    return SearchEngine(_store)


store = get_store()
engine = get_engine(store)
if store.load_error:
    st.error(store.load_error)


def entity_table():
    # tabela encji zależy od zawartości bazy - odświeżam ją po każdej zmianie
    key = tuple(sorted(store.docs))
    if st.session_state.get("ent_key") != key:
        st.session_state["ent_table"] = ner.entities_table(store)
        st.session_state["ent_key"] = key
    return st.session_state["ent_table"]


def add_files(files: list[tuple[str, bytes]]):
    bar = st.progress(0.0, text="Przetwarzanie dokumentów...")
    for i, (name, data) in enumerate(files, 1):
        if store.has_name(name):
            st.info(f"Pominięto {name} - dokument o tej nazwie już jest w bazie.")
            continue
        try:
            store.add(name, data, embed)
        except ValueError as e:
            st.warning(f"{name}: {e}")
        bar.progress(i / len(files), text=f"{i}/{len(files)}: {name}")
    bar.empty()


def show_document(doc, max_chars=None):
    meta = {
        "nazwa": doc.name, "typ pliku": doc.file_type, "rozmiar [KB]": round(doc.size_bytes / 1024, 1),
        "dodano": doc.added_at.replace("T", " "), "język": doc.language,
        "znaki": doc.char_count, "słowa": doc.word_count, "fragmenty": len(doc.chunks),
        "encje": doc.entity_count,
    }
    st.dataframe(pd.DataFrame([meta]), hide_index=True)
    if doc.language != "pl":
        st.warning("Dokument nie wygląda na polski - lematyzacja, stopwords i NER są "
                   "ustawione dla języka polskiego, więc wyniki mogą być słabe.")
    counts = pd.Series([LABELS[e["label"]] for e in doc.entities]).value_counts()
    st.caption("Encje w dokumencie: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    with st.container(height=450):
        st.markdown(f'<div style="line-height:1.7;white-space:pre-wrap">'
                    f'{ner.highlight_html(doc, max_chars)}</div>', unsafe_allow_html=True)
    path = store.file_path(doc.id)
    st.download_button("Pobierz oryginalny plik", path.read_bytes(), file_name=doc.name,
                       key=f"dl_{doc.id}_{max_chars}")


@st.dialog("Podgląd dokumentu", width="large")
def open_dialog(doc_id):
    show_document(store.get(doc_id))


# --- widok 1 --------------------------------------------------------------------
def view_documents():
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Dodaj dokumenty")
        uploaded = st.file_uploader("Pliki TXT lub PDF", type=["txt", "pdf"],
                                    accept_multiple_files=True)
        if uploaded and st.button("Dodaj do bazy", type="primary"):
            add_files([(f.name, f.getvalue()) for f in uploaded])
            st.rerun()
        samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix in (".txt", ".pdf"))
        st.caption(f"Przykładowy korpus: {len(samples)} artykułów z polskiej Wikipedii "
                   "(lista w sample_docs/SOURCES.md).")
        if samples and st.button("Załaduj przykładowe dokumenty"):
            add_files([(p.name, p.read_bytes()) for p in samples])
            st.rerun()
        st.caption("Przy dodawaniu dokument jest od razu czytany, lematyzowany, dzielony "
                   "na fragmenty, a encje i embeddingi zapisywane w bazie.")

    with right:
        docs = store.get_all()
        st.subheader(f"Dokumenty w bazie: {len(docs)}")
        if not docs:
            st.info("Baza jest pusta - dodaj pliki albo załaduj przykładowe dokumenty.")
            return
        table = pd.DataFrame([{
            "nazwa": d.name, "typ": d.file_type, "słowa": d.word_count,
            "język": d.language, "encje": d.entity_count, "dodano": d.added_at[:16].replace("T", " "),
        } for d in docs])
        st.dataframe(table, hide_index=True, height=260)
        names = {d.name: d.id for d in docs}
        chosen = st.selectbox("Szczegóły dokumentu", list(names))
        doc = store.get(names[chosen])
        show_document(doc)
        if st.button(f"Usuń dokument {doc.name}"):
            store.remove(doc.id)
            st.rerun()


# --- widok 2 --------------------------------------------------------------------
def entity_filter_ui(key: str):
    """Zwraca zbiór dozwolonych id dokumentów albo None (bez filtra)."""
    c1, c2 = st.columns(2)
    label_name = c1.selectbox("Ogranicz do dokumentów z encją typu",
                              ["(bez filtra)"] + list(LABELS.values()), key=f"{key}_type")
    ent_query = c2.text_input("...lub z konkretną encją (np. Kraków, Kopernik)", key=f"{key}_ent")
    label = {v: k for k, v in LABELS.items()}.get(label_name)
    if label is None and not ent_query.strip():
        return None
    allowed = ner.docs_with_entity(entity_table(), ent_query, label)
    st.caption(f"Filtr encji: wyszukiwanie tylko w {len(allowed)} z {len(store.docs)} dokumentów.")
    return allowed


def view_search():
    if not store.docs:
        st.info("Baza jest pusta - najpierw dodaj dokumenty.")
        return
    query = st.text_input("Zapytanie", placeholder='np. odkrycia astronomiczne, "unia lubelska", '
                                                   "zwycięstwo nad zakonem krzyżackim")
    k = st.slider("Liczba wyników na metodę (top-k)", 3, 10, 5)
    allowed = entity_filter_ui("search")
    if not query.strip():
        st.caption("Przykłady: 'skoki narciarskie', 'Krzyżacy', 'promieniotwórczość', "
                   "'stolica Polski', 'język programowania', 'gwiazdy i planety'.")
        return
    with st.spinner("Szukam..."):
        results = engine.search_all(query, k, allowed)

    cols = st.columns(3)
    for col, method in zip(cols, METHODS):
        with col:
            st.markdown(f"**{method}**")
            st.caption(METHOD_INFO[method])
            if not results[method]:
                st.info("Brak wyników.")
            for rank, r in enumerate(results[method], 1):
                with st.container(border=True):
                    st.markdown(f"**{rank}. {r.doc_name}**  \nwynik: `{r.score}` - {r.details}")
                    st.markdown(f"<small>{r.snippet}</small>", unsafe_allow_html=True)
                    if st.button("Otwórz", key=f"open_{method}_{rank}"):
                        open_dialog(r.doc_id)

    st.subheader("Porównanie rankingów")
    rankings = {m: [r.doc_id for r in results[m]] for m in METHODS}
    names = {d.id: d.name for d in store.get_all()}
    table = positions_table(rankings, names, k)
    if table.empty:
        return
    st.caption(f"Pozycja dokumentu w każdej metodzie (puste = poza top-{k}). Rozrzut to różnica "
               f"między najlepszą a najgorszą pozycją (brak w top-k liczony jako {k + 1}); "
               "wiersze z rozrzutem co najmniej 2 są zaznaczone.")
    styled = table.style.apply(
        lambda row: ["background-color: #ffe5b4" if row["rozrzut"] >= 2 else ""] * len(row),
        axis=1).format({m: "{:.0f}" for m in METHODS}, na_rep="-")
    st.dataframe(styled, hide_index=True)
    stats = pairwise_stats(rankings, k)
    st.caption("Korelacje rang liczone na sumie dokumentów z top-k wszystkich metod "
               "(1 = ta sama kolejność, 0 = brak związku, -1 = odwrotna). Jaccard@k = część "
               "wspólna zbiorów top-k.")
    st.dataframe(stats.style.format(precision=3, na_rep="nieokreślona"), hide_index=True)


# --- widok 3 --------------------------------------------------------------------
def view_entities():
    if not store.docs:
        st.info("Baza jest pusta - najpierw dodaj dokumenty.")
        return
    table = entity_table()
    st.subheader("Wyszukiwanie encji")
    c1, c2 = st.columns(2)
    q = c1.text_input("Encja (dowolna forma, np. 'Krakowie', 'Kopernika')")
    label_name = c2.selectbox("Typ encji", ["(wszystkie)"] + list(LABELS.values()))
    label = {v: k for k, v in LABELS.items()}.get(label_name)
    if q.strip() or label:
        found = ner.match_entities(table, q, label)
        st.write(f"Wystąpienia: {len(found)}, dokumenty: {found['doc_id'].nunique()}")
        summary = found.groupby(["encja", "typ"]).size().rename("wystąpienia") \
            .sort_values(ascending=False).reset_index()
        st.dataframe(summary.head(50), hide_index=True, height=220)
        for doc_id, part in found.groupby("doc_id", sort=False):
            doc = store.get(doc_id)
            with st.expander(f"{doc.name} - {len(part)} wystąpień"):
                for idx in part["idx"].head(8):
                    ent = doc.entities[idx]
                    st.markdown(f"[{ner.label_name(ent['label'])}] {ner.context(doc, ent)}",
                                unsafe_allow_html=True)
                if st.button("Otwórz dokument", key=f"ent_open_{doc_id}"):
                    open_dialog(doc_id)

    st.subheader("Najczęstsze encje w bazie")
    cols = st.columns(3)
    groups = (("Osoby", ["persName"]), ("Organizacje", ["orgName"]),
              ("Lokalizacje", list(LOCATION_LABELS)))
    for col, (title, labels) in zip(cols, groups):
        col.markdown(f"**{title}**")
        col.dataframe(ner.top_entities(table, labels, 10), hide_index=True)


# --- widok 4: statystyki -------------------------------------------------------
def view_stats():
    docs = store.get_all()
    if not docs:
        st.info("Baza jest pusta.")
        return
    table = entity_table()
    words = pd.Series({d.name: d.word_count for d in docs})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dokumenty", len(docs))
    c2.metric("Słowa łącznie", f"{words.sum():,}".replace(",", " "))
    c3.metric("Średnia długość [słowa]", int(words.mean()))
    c4.metric("Encje (wystąpienia)", len(table))

    left, right = st.columns(2)
    with left:
        st.markdown("**Rozkład długości dokumentów**")
        hist = alt.Chart(pd.DataFrame({"słowa": words.values})).mark_bar().encode(
            x=alt.X("słowa:Q", bin=alt.Bin(step=500), title="liczba słów"),
            y=alt.Y("count()", title="liczba dokumentów"))
        st.altair_chart(hist, use_container_width=True)
        st.markdown("**Rozkład typów encji**")
        types = table["typ"].value_counts().rename_axis("typ").reset_index(name="liczba")
        st.altair_chart(alt.Chart(types).mark_bar().encode(
            x=alt.X("liczba:Q"), y=alt.Y("typ:N", sort="-x", title=None)), use_container_width=True)
    with right:
        st.markdown("**Najczęstsze terminy (lematy bez stopwords)**")
        terms = pd.DataFrame(engine.top_terms(25), columns=["termin", "liczba"])
        st.altair_chart(alt.Chart(terms).mark_bar().encode(
            x=alt.X("liczba:Q"), y=alt.Y("termin:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False))
        ).properties(height=600),
            use_container_width=True)


st.title("Wyszukiwanie semantyczne i ekstrakcja wiedzy")
tabs = st.tabs(["Baza dokumentów", "Wyszukiwarka", "Encje NER", "Statystyki"])
with tabs[0]:
    view_documents()
with tabs[1]:
    view_search()
with tabs[2]:
    view_entities()
with tabs[3]:
    view_stats()
