"""NER z spacy — wykrywanie encji nazwanych."""
import html
from dataclasses import dataclass


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    doc_id: str
    doc_name: str
    context: str  # zdanie zawierające encję


# Kolory dla poszczególnych typów encji (HTML)
ENTITY_COLORS: dict[str, str] = {
    "PERSON": "#ffd700",
    "ORG": "#90ee90",
    "GPE": "#87ceeb",
    "LOC": "#87ceeb",
    "DATE": "#ffb6c1",
    "TIME": "#ffb6c1",
    "MONEY": "#dda0dd",
    "PERCENT": "#dda0dd",
    "PRODUCT": "#ffa07a",
    "EVENT": "#f0e68c",
    "WORK_OF_ART": "#f0e68c",
    "LAW": "#e6e6fa",
    "LANGUAGE": "#e0ffff",
    "NORP": "#ffe4b5",
    "FAC": "#d2b48c",
    "CARDINAL": "#d3d3d3",
    "ORDINAL": "#d3d3d3",
    "QUANTITY": "#d3d3d3",
}

LABEL_NAMES: dict[str, str] = {
    "PERSON": "Osoba",
    "ORG": "Organizacja",
    "GPE": "Miejsce (GPE)",
    "LOC": "Lokalizacja",
    "DATE": "Data",
    "TIME": "Czas",
    "MONEY": "Kwota",
    "PERCENT": "Procent",
    "PRODUCT": "Produkt",
    "EVENT": "Wydarzenie",
    "WORK_OF_ART": "Dzieło",
    "LAW": "Prawo",
    "LANGUAGE": "Język",
    "NORP": "Narodowość/Religia",
    "FAC": "Obiekt",
    "CARDINAL": "Liczba",
    "ORDINAL": "Liczba porządkowa",
    "QUANTITY": "Ilość",
}


class NEREngine:
    def __init__(self):
        self._nlp = None

    def load(self) -> None:
        """Załaduj model spacy."""
        import spacy

        for model in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                self._nlp = spacy.load(model)
                return
            except OSError:
                continue
        raise RuntimeError(
            "Brak modelu spacy. Uruchom: python -m spacy download en_core_web_sm"
        )

    def is_loaded(self) -> bool:
        return self._nlp is not None

    def extract_entities(self, text: str, doc_id: str, doc_name: str) -> list[Entity]:
        if not self.is_loaded():
            self.load()

        # spacy ma limit tokenów — podziel długi tekst
        max_chars = 100_000
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        entities: list[Entity] = []
        offset = 0
        for chunk in chunks:
            spacy_doc = self._nlp(chunk)
            for ent in spacy_doc.ents:
                # Wyciągnij zdanie zawierające encję
                context = self._get_sentence(chunk, ent.start_char, ent.end_char)
                entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label_,
                        start=offset + ent.start_char,
                        end=offset + ent.end_char,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        context=context,
                    )
                )
            offset += len(chunk)

        # Deduplikacja (ten sam tekst + label)
        seen: set[tuple[str, str]] = set()
        unique: list[Entity] = []
        for e in entities:
            key = (e.text.lower(), e.label)
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    def highlight_text(self, text: str, entities: list[Entity] | None = None) -> str:
        """Zwróć HTML z podświetlonymi encjami."""
        if not self.is_loaded():
            self.load()

        if entities is not None:
            # Użyj przekazanych encji (z deduplikacją)
            return self._highlight_with_entities(text, entities)

        # Przetwórz na bieżąco
        max_chars = 50_000
        chunk = text[:max_chars]
        spacy_doc = self._nlp(chunk)

        result_parts: list[str] = []
        last_end = 0
        for ent in spacy_doc.ents:
            # Dodaj tekst przed encją (escaped)
            result_parts.append(html.escape(chunk[last_end : ent.start_char]))
            color = ENTITY_COLORS.get(ent.label_, "#e8e8e8")
            label_name = LABEL_NAMES.get(ent.label_, ent.label_)
            result_parts.append(
                f'<mark style="background-color:{color};padding:2px 4px;border-radius:3px;" '
                f'title="{label_name}">{html.escape(ent.text)}'
                f'<sup style="font-size:0.65em;color:#555;"> {html.escape(ent.label_)}</sup></mark>'
            )
            last_end = ent.end_char

        result_parts.append(html.escape(chunk[last_end:]))
        if len(text) > max_chars:
            result_parts.append(html.escape(text[max_chars:]))

        return "".join(result_parts)

    def _highlight_with_entities(self, text: str, entities: list[Entity]) -> str:
        """Podświetl encje przekazane z zewnątrz."""
        # Sortuj po pozycji start, filtruj tylko te z tego dokumentu
        sorted_ents = sorted(
            [e for e in entities if e.start < len(text)], key=lambda e: e.start
        )

        result_parts: list[str] = []
        last_end = 0
        for ent in sorted_ents:
            if ent.start < last_end:
                continue  # pomiń nakładające się
            result_parts.append(html.escape(text[last_end : ent.start]))
            color = ENTITY_COLORS.get(ent.label, "#e8e8e8")
            label_name = LABEL_NAMES.get(ent.label, ent.label)
            result_parts.append(
                f'<mark style="background-color:{color};padding:2px 4px;border-radius:3px;" '
                f'title="{label_name}">{html.escape(ent.text)}'
                f'<sup style="font-size:0.65em;color:#555;"> {html.escape(ent.label)}</sup></mark>'
            )
            last_end = ent.end

        result_parts.append(html.escape(text[last_end:]))
        return "".join(result_parts)

    def get_entities_by_type(self, entities: list[Entity], ent_type: str) -> list[Entity]:
        return [e for e in entities if e.label == ent_type]

    def _get_sentence(self, text: str, start: int, end: int, context_chars: int = 150) -> str:
        """Wyciągnij zdanie zawierające pozycję start..end."""
        s = max(0, start - context_chars)
        e = min(len(text), end + context_chars)
        fragment = text[s:e].strip()
        if s > 0:
            fragment = "..." + fragment
        if e < len(text):
            fragment = fragment + "..."
        return fragment

    @staticmethod
    def label_display_name(label: str) -> str:
        return LABEL_NAMES.get(label, label)

    @staticmethod
    def label_color(label: str) -> str:
        return ENTITY_COLORS.get(label, "#e8e8e8")
