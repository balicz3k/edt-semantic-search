"""Streamlit app — System wyszukiwania semantycznego i ekstrakcji wiedzy (NER)."""
import streamlit as st
from pathlib import Path
import tempfile
import os

st.set_page_config(
    page_title="Semantic Search & NER",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
# CSS                                                                  #
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    .search-result-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        background: #fafafa;
    }
    .score-badge {
        display: inline-block;
        background: #0066cc;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .entity-chip {
        display: inline-block;
        border-radius: 4px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Inicjalizacja                                                        #
# ------------------------------------------------------------------ #

@st.cache_resource
def get_document_store():
    from document_store import DocumentStore
    return DocumentStore()

@st.cache_resource
def get_search_engine(_store):
    from search_engine import SearchEngine
    return SearchEngine(_store)

@st.cache_resource
def get_ner_engine():
    from ner_engine import NEREngine
    engine = NEREngine()
    return engine

store = get_document_store()
search_engine = get_search_engine(store)
ner_engine = get_ner_engine()

# ------------------------------------------------------------------ #
# Tabs                                                                 #
# ------------------------------------------------------------------ #
tab1, tab2, tab3 = st.tabs(["📚 Baza dokumentów", "🔍 Wyszukiwarka", "🏷️ Encje NER"])


# ================================================================== #
# TAB 1 — Baza dokumentów                                            #
# ================================================================== #
with tab1:
    st.title("📚 Baza dokumentów")
    st.caption("Dodawaj pliki TXT i PDF, przeglądaj i usuwaj dokumenty.")

    col_upload, col_list = st.columns([1, 2])

    with col_upload:
        st.subheader("Dodaj dokument")
        uploaded_file = st.file_uploader(
            "Wybierz plik TXT lub PDF",
            type=["txt", "pdf"],
            key="uploader",
        )
        if uploaded_file is not None:
            if st.button("➕ Dodaj do bazy", use_container_width=True):
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    doc = store.add_document(tmp_path)
                    # Zmień nazwę na oryginalną
                    doc.name = uploaded_file.name
                    store._save()
                    search_engine.reindex()
                    st.success(f"✅ Dodano: **{uploaded_file.name}** ({doc.char_count:,} znaków)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Błąd: {e}")
                finally:
                    os.unlink(tmp_path)

        st.divider()
        st.subheader("Załaduj przykładowe dokumenty")
        sample_dir = Path("sample_docs")
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.pdf"))
            if sample_files:
                if st.button("📂 Załaduj wszystkie przykłady", use_container_width=True):
                    added = 0
                    already = 0
                    errors = 0
                    existing_names = {d.name for d in store.get_all()}
                    for sf in sample_files:
                        if sf.name in existing_names:
                            already += 1
                            continue
                        try:
                            store.add_document(sf)
                            added += 1
                        except Exception:
                            errors += 1
                    if added:
                        search_engine.reindex()
                        st.success(f"Dodano {added} dokument(ów). Pominiętych (już w bazie): {already}.")
                        st.rerun()
                    else:
                        st.info(f"Wszystkie przykładowe dokumenty są już w bazie ({already}).")

    with col_list:
        docs = store.get_all()
        st.subheader(f"Dokumenty w bazie ({len(docs)})")

        if not docs:
            st.info("Baza jest pusta. Dodaj dokumenty za pomocą formularza po lewej.")
        else:
            for doc in sorted(docs, key=lambda d: d.added_at, reverse=True):
                with st.expander(f"📄 {doc.name}  —  {doc.char_count:,} znaków"):
                    st.caption(f"ID: `{doc.id}` | Dodano: {doc.added_at[:19].replace('T', ' ')}")
                    st.text_area(
                        "Podgląd treści (pierwsze 500 znaków)",
                        value=doc.content[:500],
                        height=120,
                        disabled=True,
                        key=f"preview_{doc.id}",
                    )
                    if st.button(f"🗑️ Usuń {doc.name}", key=f"del_{doc.id}"):
                        store.remove_document(doc.id)
                        search_engine.reindex()
                        st.success(f"Usunięto: {doc.name}")
                        st.rerun()


# ================================================================== #
# TAB 2 — Wyszukiwarka                                               #
# ================================================================== #
with tab2:
    st.title("🔍 Wyszukiwarka dokumentów")
    st.caption("Porównaj trzy metody wyszukiwania dla tego samego zapytania.")

    docs_count = len(store.get_all())
    if docs_count == 0:
        st.warning("Baza dokumentów jest pusta. Przejdź do zakładki **📚 Baza dokumentów** i dodaj pliki.")
    else:
        query = st.text_input(
            "Wpisz zapytanie:",
            placeholder="np. artificial intelligence, World War II, climate change...",
            key="search_query",
        )
        top_k = st.slider("Liczba wyników", min_value=1, max_value=min(10, docs_count), value=min(5, docs_count))

        if query and query.strip():
            with st.spinner("Szukam..."):
                kw_results = search_engine.keyword_search(query.strip(), top_k=top_k)
                tfidf_results = search_engine.tfidf_search(query.strip(), top_k=top_k)
                sem_results = search_engine.semantic_search(query.strip(), top_k=top_k)

            col1, col2, col3 = st.columns(3)

            def render_results(results, label):
                if not results:
                    st.info("Brak wyników.")
                    return
                for rank, r in enumerate(results, 1):
                    with st.container():
                        st.markdown(
                            f'<div class="search-result-card">'
                            f'<b>#{rank} {r.doc_name}</b> '
                            f'<span class="score-badge">{r.score:.4f}</span><br>'
                            f'<small>{r.snippet}</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            with col1:
                st.subheader("🔑 Słowa kluczowe")
                st.caption("Zliczanie wystąpień słów (znormalizowane)")
                render_results(kw_results, "keyword")

            with col2:
                st.subheader("📊 TF-IDF")
                st.caption("Podobieństwo kosinusowe wektorów TF-IDF")
                render_results(tfidf_results, "tfidf")

            with col3:
                st.subheader("🧠 Semantyczne")
                st.caption("Sentence-BERT (all-MiniLM-L6-v2) + ChromaDB")
                render_results(sem_results, "semantic")

            # Porównanie kolejności
            if kw_results or tfidf_results or sem_results:
                st.divider()
                st.subheader("📈 Porównanie rankingów")
                all_doc_names = sorted(
                    set(r.doc_name for r in kw_results + tfidf_results + sem_results)
                )
                kw_rank = {r.doc_name: i + 1 for i, r in enumerate(kw_results)}
                tfidf_rank = {r.doc_name: i + 1 for i, r in enumerate(tfidf_results)}
                sem_rank = {r.doc_name: i + 1 for i, r in enumerate(sem_results)}

                import pandas as pd
                comparison_data = []
                for name in all_doc_names:
                    comparison_data.append({
                        "Dokument": name,
                        "Słowa kluczowe": kw_rank.get(name, "—"),
                        "TF-IDF": tfidf_rank.get(name, "—"),
                        "Semantyczne": sem_rank.get(name, "—"),
                    })
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)


# ================================================================== #
# TAB 3 — Encje NER                                                  #
# ================================================================== #
with tab3:
    st.title("🏷️ Ekstrakcja encji nazwanych (NER)")
    st.caption("Wykrywanie osób, organizacji, lokalizacji, dat i innych encji za pomocą modelu spacy.")

    docs = store.get_all()
    if not docs:
        st.warning("Baza dokumentów jest pusta. Przejdź do zakładki **📚 Baza dokumentów** i dodaj pliki.")
    else:
        # Załaduj model NER
        if not ner_engine.is_loaded():
            with st.spinner("Ładowanie modelu NER (spacy)..."):
                try:
                    ner_engine.load()
                    st.success("Model NER załadowany.")
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

        # Wybór dokumentu do analizy
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Ustawienia")
            doc_options = {d.name: d for d in docs}
            selected_doc_name = st.selectbox("Wybierz dokument", list(doc_options.keys()))
            selected_doc = doc_options[selected_doc_name]

            # Filtrowanie po typie encji
            from ner_engine import LABEL_NAMES
            all_labels = ["Wszystkie"] + list(LABEL_NAMES.keys())
            selected_label = st.selectbox("Filtruj po typie encji", all_labels)

            # Wyszukiwanie encji
            entity_search = st.text_input("Szukaj encji (tekst):", placeholder="np. Google, 2023, London")

            analyze_btn = st.button("🔎 Analizuj dokument", use_container_width=True, type="primary")

        with col_right:
            if analyze_btn or ("ner_entities" in st.session_state and st.session_state.get("ner_doc_id") == selected_doc.id):
                if analyze_btn:
                    with st.spinner("Analizuję encje..."):
                        entities = ner_engine.extract_entities(
                            selected_doc.content, selected_doc.id, selected_doc.name
                        )
                    st.session_state["ner_entities"] = entities
                    st.session_state["ner_doc_id"] = selected_doc.id
                else:
                    entities = st.session_state.get("ner_entities", [])

                entities = st.session_state.get("ner_entities", [])

                # Filtrowanie
                filtered = entities
                if selected_label != "Wszystkie":
                    filtered = [e for e in filtered if e.label == selected_label]
                if entity_search.strip():
                    filtered = [e for e in filtered if entity_search.lower() in e.text.lower()]

                st.subheader(f"Znalezione encje: {len(filtered)} (z {len(entities)} łącznie)")

                # Podgląd dokumentu z podświetleniem
                with st.expander("📖 Podgląd dokumentu z podświetlonymi encjami", expanded=False):
                    highlighted = ner_engine.highlight_text(selected_doc.content[:5000], entities=entities)
                    st.markdown(
                        f'<div style="line-height:1.8;white-space:pre-wrap;font-family:sans-serif;">'
                        f'{highlighted}</div>',
                        unsafe_allow_html=True,
                    )
                    if len(selected_doc.content) > 5000:
                        st.caption(f"(Pokazano pierwsze 5000 znaków z {len(selected_doc.content):,})")

                # Legenda
                st.subheader("Legenda typów encji")
                from ner_engine import ENTITY_COLORS
                legend_cols = st.columns(4)
                label_counts = {}
                for e in entities:
                    label_counts[e.label] = label_counts.get(e.label, 0) + 1

                for i, (label, count) in enumerate(sorted(label_counts.items(), key=lambda x: -x[1])):
                    color = ENTITY_COLORS.get(label, "#e8e8e8")
                    display_name = LABEL_NAMES.get(label, label)
                    legend_cols[i % 4].markdown(
                        f'<span class="entity-chip" style="background:{color};">'
                        f'{display_name} ({label}): <b>{count}</b></span>',
                        unsafe_allow_html=True,
                    )

                st.divider()

                # Tabela encji
                if filtered:
                    import pandas as pd
                    table_data = [
                        {
                            "Encja": e.text,
                            "Typ": e.label,
                            "Nazwa typu": LABEL_NAMES.get(e.label, e.label),
                            "Kontekst": e.context[:100] + "..." if len(e.context) > 100 else e.context,
                        }
                        for e in filtered
                    ]
                    st.dataframe(
                        pd.DataFrame(table_data),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Brak encji spełniających kryteria filtrowania.")

        # Wyszukiwanie encji w całej bazie
        st.divider()
        st.subheader("🌐 Wyszukaj encję w całej bazie dokumentów")
        global_entity_search = st.text_input("Wpisz encję do wyszukania:", key="global_ner_search")
        if global_entity_search.strip():
            with st.spinner("Przeszukuję wszystkie dokumenty..."):
                found_docs: dict[str, list] = {}
                for doc in docs:
                    ents = ner_engine.extract_entities(doc.content, doc.id, doc.name)
                    matches = [e for e in ents if global_entity_search.lower() in e.text.lower()]
                    if matches:
                        found_docs[doc.name] = matches

            if found_docs:
                st.success(f"Encja **'{global_entity_search}'** znaleziona w {len(found_docs)} dokument(ach):")
                for doc_name, ents in found_docs.items():
                    with st.expander(f"📄 {doc_name} — {len(ents)} trafień"):
                        for e in ents[:10]:
                            color = ENTITY_COLORS.get(e.label, "#e8e8e8")
                            st.markdown(
                                f'<span class="entity-chip" style="background:{color};">'
                                f'<b>{e.text}</b> [{e.label}]</span> — <i>{e.context[:120]}</i>',
                                unsafe_allow_html=True,
                            )
            else:
                st.info(f"Encja **'{global_entity_search}'** nie znaleziona w żadnym dokumencie.")
