# Jak to działa - notatki do nauki i obrony

## 1. Co się dzieje po dodaniu dokumentu

plik -> tekst (TXT albo PDF przez PyMuPDF) -> metadane -> spaCy `pl_core_news_md`
(lematy + encje) -> podział na fragmenty ~600 znaków -> embedding każdego fragmentu ->
zapis w `storage/`. Wszystko liczone raz; przy wyszukiwaniu korzystam z gotowych wyników.

## 2. Słowa kluczowe

Zapytanie dzielę na frazy w cudzysłowie (`"unia lubelska"`) i pojedyncze słowa
(bez stopwords typu "nad", "w" - pasowałyby do każdego tekstu). Szukam dokładnych
wystąpień (wyrażenie regularne z granicami słowa, bez wielkości liter). Ranking:
najpierw ile różnych słów zapytania jest w dokumencie, potem łączna liczba wystąpień.
Wada: nie zna odmiany - "Kraków" nie znajdzie "w Krakowie" - ani synonimów.

## 3. TF-IDF

- Dokument = lista lematów bez stopwords (spaCy: "Krakowie" -> "kraków", "odkryła" -> "odkryć").
  Dla polskiego to ważne (pułapka 2 z prezentacji: kot, kota, kotu... to jedno słowo).
- Wektor TF-IDF: `tfidf(t, d) = (1 + ln tf(t, d)) * idf(t)`,
  `idf(t) = ln((1 + N) / (1 + df(t))) + 1`, potem normalizacja do długości 1.
  Słowo częste w dokumencie, a rzadkie w korpusie, dostaje dużą wagę.
- Zapytanie przechodzi przez ten sam preprocessing, wynik dokumentu to
  podobieństwo cosinusowe `cos(q, d) = q·d / (|q| |d|)`.
- Wada: potrzebne są wspólne słowa (lematy); przy parafrazie bez wspólnych słów wynik = 0.

## 4. Wyszukiwanie semantyczne

- Model `intfloat/multilingual-e5-small` zamienia tekst na wektor 384 liczb tak, że teksty
  o podobnym znaczeniu mają podobne wektory (nawet bez wspólnych słów i w różnych językach).
- Model był trenowany z prefiksami: fragment dokumentu koduję jako `"passage: ..."`,
  zapytanie jako `"query: ..."`.
- Cały artykuł jest za długi dla modelu (limit 512 tokenów, a sens się "rozmywa"),
  dlatego dzielę dokument na fragmenty ~600 znaków. Wynik dokumentu = największe
  podobieństwo cosinusowe zapytania do któregoś jego fragmentu; ten fragment to snippet.
- Pierwsza wersja dzieliła tekst po liniach i krótkie nagłówki (np. "Laboratorium ...")
  pasowały do przypadkowych zapytań - stąd sklejanie do min. 600 znaków.
- Wektory są znormalizowane, więc cosinus to zwykły iloczyn skalarny (`macierz @ wektor`).

## 5. Porównanie rankingów

- Tabela: każdy dokument z top-k którejkolwiek metody i jego pozycja w każdej metodzie;
  rozrzut = najgorsza - najlepsza pozycja (brak w top-k liczę jako k+1).
- **Spearman**: korelacja Pearsona policzona na pozycjach (rangach); 1 - ta sama kolejność,
  -1 - odwrotna, 0 - brak związku.
- **Kendall tau**: (pary zgodne - pary niezgodne) / wszystkie pary - patrzy, ile par
  dokumentów obie metody ustawiają w tej samej kolejności.
- **Jaccard@k** = |A ∩ B| / |A ∪ B| dla zbiorów top-k - czy metody w ogóle zwracają
  te same dokumenty, niezależnie od kolejności.
- Problem: metody zwracają różne dokumenty, a korelacja wymaga pozycji dla tych samych.
  Rozwiązanie: suma zbiorów top-k i pozycja k+1 dla brakujących (opisane w kodzie).

## 6. NER

- Model `pl_core_news_md` (trenowany na NKJP) ma 6 typów: `persName` (osoba),
  `orgName` (organizacja), `placeName` (miejscowość/państwo), `geogName` (obiekt
  geograficzny: rzeka, góry), `date`, `time`.
- Dla każdej encji zapisuję tekst, lemat (forma podstawowa: "Mikołaja Kopernika" ->
  "Mikołaj Kopernik"), typ i pozycję w tekście. Wyszukiwanie encji porównuje lematy,
  więc działa dla dowolnej formy ("Krakowie" znajduje "Kraków").
- Wyróżnianie: wstawiam `<mark>` w miejscach encji (pozycje z cache - bez ponownego NER).
- Filtr w wyszukiwarce: zbiór id dokumentów zawierających encję / typ encji ogranicza
  kandydatów we wszystkich trzech metodach.
- Wielkie litery są tu ważne (pułapka 3 z prezentacji) - dlatego do NER daję oryginalny
  tekst, a małe litery stosuję tylko do lematów w TF-IDF.

## 7. Pytania na prezentację (z odpowiedziami)

1. **Czym różnią się trzy metody?** Słowa kluczowe - dokładne słowa; TF-IDF - wspólne
   lematy ważone rzadkością; semantyczne - podobieństwo znaczenia wektorów z modelu.
2. **Kiedy która wygrywa?** Słowa kluczowe przy dokładnej nazwie/frazie, TF-IDF przy
   słowach w różnych formach i rzadkich terminach, semantyczne przy opisie innymi słowami.
3. **Po co lematyzacja w TF-IDF?** Polski ma bogatą fleksję; bez niej "Kraków",
   "Krakowa", "Krakowie" to trzy różne cechy.
4. **Dlaczego stopwords nie są problemem jak w sentymencie?** Tu nie oceniamy wydźwięku,
   więc "nie" nie zmienia tematu dokumentu; stopwords tylko zaszumiają TF-IDF.
5. **Skąd wiesz, że rankingi się różnią?** Tabela pozycji i miary: Spearman, Kendall,
   Jaccard@k - w notatniku średni Jaccard@5 to 0,28-0,52.
6. **Jak liczysz korelację, gdy metody zwracają inne dokumenty?** Na sumie top-k,
   brakującym daję pozycję k+1.
7. **Dlaczego dzielisz dokument na fragmenty?** Model ma limit długości i jeden wektor
   na cały artykuł uśrednia wiele tematów; fragment daje też sensowny snippet.
8. **Dlaczego e5-small, a nie MiniLM?** Jest trenowany do wyszukiwania; na moich
   12 zapytaniach MRR@5 0,96 zamiast 0,82, przy podobnym rozmiarze.
9. **Czym jest cosinus?** Kąt między wektorami: 1 - ten sam kierunek, 0 - prostopadłe;
   nie zależy od długości tekstu.
10. **Jak działa filtr encji?** Z tabeli encji wybieram dokumenty, w których jest dana
    encja (po lemacie) albo typ; wyszukiwanie liczy wyniki tylko dla nich.
11. **Jakie błędy robi NER?** Przymiotniki "polski" jako miejsce, odmiana nazwisk
    ("Lema" i "Lem"), czasem zły typ. Model `md` to kompromis między szybkością a jakością.
12. **Co się dzieje z PDF?** PyMuPDF wyciąga tekst ze stron; sklejam linie w akapity
    i usuwam przeniesienia wyrazów.
13. **Co gdy plik bazy się uszkodzi?** Aplikacja nie kasuje bazy: kopiuje uszkodzony plik
    obok i pokazuje błąd. Zapis jest atomowy, więc przerwany zapis nie psuje pliku.
14. **Co to MRR?** Średnia z 1/pozycja pierwszego trafnego dokumentu (0, gdy brak w top-k).
15. **Skąd korpus i czy wolno go użyć?** Wikipedia, licencja CC BY-SA 4.0 - źródła
    i data pobrania w `sample_docs/SOURCES.md`.
