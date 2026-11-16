# EDT Projekt 2 - System wyszukiwania semantycznego i ekstrakcji wiedzy

Projekt z przedmiotu Eksploracja Danych Tekstowych (sem. 2, 2026/2027).
Aplikacja (Streamlit) przechowuje dokumenty TXT i PDF, wyszukuje w nich trzema metodami
(słowa kluczowe, TF-IDF, semantyczne), porównuje rankingi tych metod i wydobywa encje
nazwane (NER). Działa dla języka polskiego.

## Wymagania projektu i gdzie są spełnione

| Wymaganie | Gdzie |
|---|---|
| dodawanie TXT i PDF, odczyt i przetwarzanie treści | widok Baza dokumentów, `document_store.py` |
| przechowywanie dokumentów i metadanych, przeglądanie, usuwanie | `storage/`, widok Baza dokumentów |
| automatyczne przygotowanie do wyszukiwania i NER | przy dodaniu: lematy, fragmenty, encje, embeddingi (`nlp_pl.py`) |
| 3 metody wyszukiwania, wynik, fragment tekstu, otwarcie dokumentu | widok Wyszukiwarka, `search_engine.py` |
| wyniki trzech metod obok siebie i różnice w kolejności | tabela pozycji + Spearman, Kendall, Jaccard@k (`ranking_compare.py`) |
| NER: osoby, organizacje, lokalizacje, daty, czas | `pl_core_news_md`, widok Encje NER |
| encje z typami i fragmentami, wyróżnienie w dokumencie | widok Encje NER, podgląd dokumentu |
| wyszukiwanie encji i dokumentów z encją | widok Encje NER (dowolna forma: "Krakowie" -> "Kraków") |
| ograniczenie wyszukiwania do dokumentów z encją / typem encji | filtr w Wyszukiwarce |
| panel najczęstszych encji, statystyki (dodatkowe) | widok Encje NER, widok Statystyki |

## Korpus

40 artykułów z polskiej Wikipedii z 7 tematów (historia, nauka, sport, technologia,
geografia, kultura, organizacje), w tym 5 zapisanych jako PDF, żeby pokazać obsługę PDF.
Lista z adresami, datą pobrania i licencją (CC BY-SA 4.0): `sample_docs/SOURCES.md`.
Pobiera je `scripts/download_corpus.py` przez API MediaWiki (Wikimedia zaleca API zamiast
parsowania HTML; przerwa 3 s między zapytaniami, przy HTTP 429 czekam tyle, ile podaje
nagłówek `Retry-After`; ogólny User-Agent bez danych osobowych). Sekcje "Przypisy",
"Bibliografia" itp. są pomijane, a artykuły dłuższe niż 30 000 znaków przycięte na
granicy akapitu. Wcześniejsze 4 przykładowe teksty zostały usunięte.

## Instalacja i uruchomienie

Sprawdzone na Pythonie 3.12 (Kubuntu, czysty venv).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download pl_core_news_md
python scripts/build_index.py     # wczytuje sample_docs/ do bazy (ok. 1 min na CPU)
streamlit run app.py
```

Zamiast `build_index.py` można w aplikacji kliknąć "Załaduj przykładowe dokumenty".
Przy pierwszym uruchomieniu `sentence-transformers` pobiera model
`intfloat/multilingual-e5-small` (ok. 470 MB) do `~/.cache/huggingface`.
Korpus jest już w repozytorium; `python scripts/download_corpus.py` pobiera go od nowa.

Testy i notatnik:

```bash
pytest -q
jupyter notebook notebooks/01_porownanie_metod.ipynb
```

## Struktura

```
app.py                  GUI: Baza dokumentów, Wyszukiwarka, Encje NER, Statystyki
document_store.py       dodawanie/usuwanie, odczyt TXT/PDF, metadane, zapis w storage/
nlp_pl.py               spaCy PL: lematy, stopwords, encje, podział na fragmenty, język
search_engine.py        słowa kluczowe, TF-IDF, semantyczne, snippety, filtr dokumentów
ranking_compare.py      tabela pozycji, Spearman, Kendall tau, Jaccard@k
ner_engine.py           wyróżnianie encji, wyszukiwanie encji, najczęstsze encje
scripts/                download_corpus.py, build_index.py
notebooks/01_porownanie_metod.ipynb   13 zapytań testowych, MRR, korelacje rang
storage/                baza tworzona przez aplikację (poza gitem)
docs/NAUKA.md           opis "po ludzku" + pytania na prezentację
docs/prezentacja.pdf    slajdy
```

## Przetwarzanie dokumentu przy dodaniu

1. odczyt tekstu: TXT (UTF-8, z zapasowym cp1250 / ISO-8859-2) albo PDF (PyMuPDF,
   sklejanie linii w akapity, usuwanie przeniesień wyrazów),
2. metadane: nazwa, typ, rozmiar, data dodania, liczba znaków i słów, język
   (prosty detektor: które stopwords - polskie czy angielskie - pokrywają więcej słów;
   dla nie-polskich dokumentów GUI pokazuje ostrzeżenie),
3. jedno przejście `pl_core_news_md`: lematy bez stopwords (do TF-IDF) i encje (cache NER),
4. podział na fragmenty po ok. 600 znaków (zdania i akapity sklejane, za długie cięte),
5. embeddingi fragmentów (`multilingual-e5-small`) zapisane w `storage/embeddings/`.

Baza to `storage/documents.json` zapisywany atomowo (plik tymczasowy + zamiana).
Uszkodzony plik nie jest kasowany - aplikacja robi jego kopię i pokazuje błąd.

## Metody wyszukiwania

| metoda | jak liczy wynik | fragment tekstu |
|---|---|---|
| słowa kluczowe | dokument zawiera słowa / frazy w cudzysłowie (dokładna forma, bez stopwords); ranking: liczba dopasowanych słów, potem liczba wystąpień | fragment z największą liczbą trafień, trafienia wyróżnione |
| TF-IDF | podobieństwo cosinusowe wektorów TF-IDF z lematów (zapytanie lematyzowane tak samo) | fragment z największą liczbą lematów zapytania |
| semantyczne | największe podobieństwo cosinusowe embeddingu zapytania do embeddingu fragmentu | najlepiej pasujący fragment |

Model semantyczny wybrałem na 12 zapytaniach testowych: `multilingual-e5-small`
miał MRR@5 0,96, `paraphrase-multilingual-MiniLM-L12-v2` 0,82.

**Porównanie rankingów**: tabela dokumentów z pozycją w każdej metodzie i rozrzutem
pozycji (wiersze z rozrzutem >= 2 zaznaczone), a dla każdej pary metod korelacja rang
Spearmana i tau Kendalla oraz Jaccard@k. Korelacje liczę na sumie dokumentów z top-k
wszystkich metod; dokument spoza top-k danej metody dostaje pozycję k+1.

## Wyniki (notatnik)

13 zapytań (dosłowne, parafrazy, inna forma gramatyczna, wieloznaczne) z ręcznie
oznaczonymi trafnymi dokumentami: wszystkie metody zwykle stawiają trafny dokument na
1. miejscu (MRR: słowa kluczowe 1,00, semantyczne 0,96, TF-IDF 0,92), ale dalsza
kolejność bardzo się różni - średni Jaccard@5 od 0,28 (słowa kluczowe - semantyczne)
do 0,52 (słowa kluczowe - TF-IDF). TF-IDF przegrał na "Krzyżacy", bo lematyzator spaCy
zwrócił błędny lemat. Szczegóły i wnioski: `notebooks/01_porownanie_metod.ipynb`.

## Ograniczenia

- Lematyzacja spaCy myli się przy rzadkich nazwach własnych (np. "Krzyżacy" -> "krzyżaca",
  "Lema" i "Lem" jako dwie różne encje).
- Model NER (korpus NKJP) oznacza przymiotniki od nazw miejsc ("polski") jako `placeName`;
  w panelu najczęstszych encji pokazuję tylko nazwy pisane wielką literą.
- Wyszukiwanie semantyczne zawsze zwraca k wyników - nie ma progu "nic nie pasuje".
- Korpus jest mały (40 dokumentów) i tematycznie zróżnicowany, więc łatwy;
  na większym, jednorodnym zbiorze różnice między metodami byłyby większe.
