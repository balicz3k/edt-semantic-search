# System wyszukiwania semantycznego i ekstrakcji wiedzy (NER)

Projekt 2 — Eksploracja Danych Tekstowych (EDT), semestr 2

## Opis

Aplikacja webowa zbudowana w Streamlit umożliwiająca:
- Zarządzanie bazą dokumentów tekstowych (TXT, PDF)
- Wyszukiwanie dokumentów trzema metodami: słowa kluczowe, TF-IDF, wyszukiwanie semantyczne
- Ekstrakcję i analizę encji nazwanych (NER) za pomocą modelu spacy

## Struktura projektu

```
edt-semantic-search/
  app.py               # Główna aplikacja Streamlit (3 zakładki)
  document_store.py    # Zarządzanie dokumentami i metadanymi (JSON)
  search_engine.py     # 3 metody wyszukiwania
  ner_engine.py        # NER z użyciem spacy
  requirements.txt     # Zależności Python
  sample_docs/         # Przykładowe dokumenty TXT do testów
    sample_tech.txt    # Artykuł o AI i technologii
    sample_news.txt    # Artykuł o polityce i gospodarce
    sample_history.txt # Artykuł historyczny (II Wojna Światowa)
```

## Stack technologiczny

| Komponent | Technologia |
|---|---|
| GUI | Streamlit |
| Wyszukiwanie semantyczne | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| TF-IDF | scikit-learn |
| NER | spacy (`en_core_web_sm`) |
| Czytanie PDF | PyMuPDF (`fitz`) |

## Funkcje

### Zakładka 1: Baza dokumentów
- Dodawanie plików TXT i PDF przez uploader
- Szybkie ładowanie przykładowych dokumentów
- Podgląd treści i metadanych dokumentu
- Usuwanie dokumentów z automatyczną reineksacją

### Zakładka 2: Wyszukiwarka
Trzy metody porównane obok siebie w kolumnach:

1. **Słowa kluczowe** — zlicza wystąpienia słów z zapytania, normalizuje przez długość dokumentu
2. **TF-IDF** — podobieństwo kosinusowe wektorów TF-IDF (sklearn), uwzględnia ważność słów
3. **Semantyczne** — koduje zapytanie modelem Sentence-BERT, przeszukuje ChromaDB (HNSW)

Tabela porównania rankingów pokazuje różnice w kolejności wyników między metodami.

### Zakładka 3: Encje NER
- Wykrywanie encji: osoby (PERSON), organizacje (ORG), miejsca (GPE/LOC), daty (DATE), i inne
- Podgląd dokumentu z podświetlonymi encjami (kolorowe znaczniki z typami)
- Legenda typów z liczbami wystąpień
- Tabela encji z filtrowaniem po typie i wyszukiwaniem tekstu
- Wyszukiwanie encji w całej bazie dokumentów

## Instalacja i uruchomienie

### Wymagania
- Python 3.11+ (zalecany) lub 3.14
- ~2 GB wolnej pamięci RAM (modele ML)

### Instalacja

```bash
# Sklonuj repozytorium
git clone https://github.com/balicz3k/edt-semantic-search.git
cd edt-semantic-search

# Utwórz środowisko wirtualne
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# lub: .venv\Scripts\activate  # Windows

# Zainstaluj zależności
pip install -r requirements.txt

# Pobierz model spacy
python -m spacy download en_core_web_sm
```

### Uruchomienie

```bash
source .venv/bin/activate
streamlit run app.py
```

Aplikacja będzie dostępna pod adresem `http://localhost:8501`

### Szybki test

```bash
python -c "
from document_store import DocumentStore
from search_engine import SearchEngine
ds = DocumentStore()
ds.add_document('sample_docs/sample_tech.txt')
se = SearchEngine(ds)
se.reindex()
results = se.semantic_search('artificial intelligence')
print([r.doc_name for r in results])
"
```

## Autor

Projekt realizowany na potrzeby przedmiotu Eksploracja Danych Tekstowych (EDT),
Politechnika — studia magisterskie, semestr 2.
