"""Pobiera korpus przykładowych dokumentów: artykuły z polskiej Wikipedii.

Korzystam z API MediaWiki (prop=extracts, explaintext=1) - Wikimedia zaleca
API zamiast parsowania HTML. Między zapytaniami robię przerwę 3 s, a po odpowiedzi 429
czekam tyle, ile podaje nagłówek Retry-After.
Tematy są różne (historia, nauka, sport, technologia, geografia, kultura,
organizacje), żeby trzy metody wyszukiwania dawały różne rankingi.

Wynik: sample_docs/*.txt, kilka artykułów jako *.pdf (żeby pokazać obsługę PDF)
oraz sample_docs/SOURCES.md z listą źródeł i licencją.
Uruchomienie: python scripts/download_corpus.py
"""
import re
import time
from datetime import date
from pathlib import Path

import httpx
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

API = "https://pl.wikipedia.org/w/api.php"
USER_AGENT = "edt-semantic-search/1.0 (projekt studencki, python-httpx)"
OUT_DIR = Path(__file__).resolve().parent.parent / "sample_docs"
MAX_CHARS = 30_000  # dłuższe artykuły przycinam na granicy akapitu (szybszy NER)

ARTICLES = {
    "historia": ["Bitwa pod Grunwaldem", "Powstanie warszawskie", "Konstytucja 3 maja",
                 "Unia lubelska", "Chrzest Polski", "Solidarność (związek zawodowy)"],
    "nauka": ["Maria Skłodowska-Curie", "Mikołaj Kopernik", "Ogólna teoria względności",
              "Fotosynteza", "Kwas deoksyrybonukleinowy", "Czarna dziura"],
    "sport": ["Robert Lewandowski", "Iga Świątek", "Adam Małysz",
              "Letnie Igrzyska Olimpijskie 2024", "Tour de Pologne", "Mistrzostwa Świata w Piłce Nożnej 2022"],
    "technologia": ["Sztuczna inteligencja", "Internet", "Python",
                    "Linux", "Uczenie maszynowe", "Smartfon"],
    "geografia": ["Kraków", "Wisła", "Tatry", "Morze Bałtyckie", "Warszawa", "Kraina Wielkich Jezior Mazurskich"],
    "kultura": ["Fryderyk Chopin", "Adam Mickiewicz", "Pan Tadeusz", "Wisława Szymborska",
                "Andrzej Wajda", "Stanisław Lem"],
    "organizacje": ["Unia Europejska", "Organizacja Traktatu Północnoatlantyckiego",
                    "Narodowy Bank Polski", "Politechnika Krakowska"],
}
# te artykuły zapisuję jako PDF zamiast TXT
AS_PDF = {"Mikołaj Kopernik", "Tatry", "Iga Świątek", "Unia Europejska", "Pan Tadeusz"}

# sekcje końcowe, które nie wnoszą treści (przypisy, linki)
END_SECTIONS = ("Zobacz też", "Uwagi", "Przypisy", "Bibliografia", "Linki zewnętrzne",
                "Literatura", "Przypis")


def slug(title: str) -> str:
    s = re.sub(r"[^\w]+", "_", title, flags=re.UNICODE).strip("_")
    return s[:60]


def clean_extract(text: str) -> str:
    # odcinam sekcje końcowe
    for name in END_SECTIONS:
        m = re.search(rf"\n==+ {re.escape(name)} ==+", text)
        if m:
            text = text[:m.start()]
    # nagłówki "== Historia ==" zamieniam na zwykłe linie
    text = re.sub(r"\n(=+) *(.+?) *\1\n", r"\n\n\2\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > MAX_CHARS:
        cut = text.rfind("\n\n", 0, MAX_CHARS)
        text = text[:cut if cut > 0 else MAX_CHARS]
    return text


def fetch_article(client: httpx.Client, title: str) -> dict:
    params = {"action": "query", "prop": "extracts|info", "inprop": "url",
              "explaintext": 1, "redirects": 1, "format": "json", "titles": title}
    for attempt in range(5):
        resp = client.get(API, params=params)
        if resp.status_code != 429:
            break
        # limit zapytań - czekam tyle, ile każe serwer (nagłówek Retry-After)
        wait = int(resp.headers.get("Retry-After", 30))
        print(f"  429, czekam {wait} s")
        time.sleep(wait)
    resp.raise_for_status()
    page = next(iter(resp.json()["query"]["pages"].values()))
    if "missing" in page:
        raise KeyError(f"Brak artykułu: {title}")
    return {"title": page["title"], "url": page["fullurl"], "text": clean_extract(page["extract"])}


def write_pdf(path: Path, title: str, text: str):
    pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    styles = getSampleStyleSheet()
    body = styles["BodyText"].clone("pl", fontName="DejaVu", fontSize=10, leading=14)
    head = styles["Title"].clone("plt", fontName="DejaVu")
    story = [Paragraph(title, head), Spacer(1, 8)]
    for para in text.split("\n\n"):
        para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(para.replace("\n", "<br/>"), body))
        story.append(Spacer(1, 4))
    SimpleDocTemplate(str(path), pagesize=A4, title=title).build(story)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for old in list(OUT_DIR.glob("*.txt")) + list(OUT_DIR.glob("*.pdf")):
        old.unlink()
    rows = []
    today = date.today().isoformat()
    with httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT}) as client:
        for topic, titles in ARTICLES.items():
            for title in titles:
                try:
                    art = fetch_article(client, title)
                except (httpx.HTTPError, KeyError) as e:
                    print(f"pomijam {title}: {e}")
                    continue
                name = slug(art["title"])
                if title in AS_PDF:
                    fname = f"{name}.pdf"
                    write_pdf(OUT_DIR / fname, art["title"], art["text"])
                else:
                    fname = f"{name}.txt"
                    (OUT_DIR / fname).write_text(art["title"] + "\n\n" + art["text"], encoding="utf-8")
                rows.append((art["title"], topic, fname, art["url"], len(art["text"])))
                print(f"{topic:12s} {art['title']:45s} {len(art['text']):6d} znaków")
                time.sleep(3)

    lines = ["# Źródła przykładowych dokumentów", "",
             f"Wszystkie dokumenty to artykuły z polskiej Wikipedii pobrane {today} skryptem "
             "`scripts/download_corpus.py` (API MediaWiki, tekst bez formatowania).",
             "Licencja treści: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), "
             "autorzy: redaktorzy Wikipedii (historia edycji pod podanym adresem).",
             f"Pominięto sekcje końcowe (przypisy, bibliografia, linki), a artykuły dłuższe niż "
             f"{MAX_CHARS} znaków przycięto na granicy akapitu.", "",
             "| Tytuł | Temat | Plik | Adres | Znaki |", "|---|---|---|---|---|"]
    lines += [f"| {t} | {topic} | {f} | {u} | {n} |" for t, topic, f, u, n in rows]
    (OUT_DIR / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Zapisano {len(rows)} dokumentów")


if __name__ == "__main__":
    main()
