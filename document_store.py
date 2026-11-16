"""Baza dokumentów: dodawanie TXT/PDF, metadane, zapis na dysku, usuwanie.

Układ katalogu storage/ (tworzony automatycznie, nie trafia do gita):
  documents.json        - metadane, treść, lematy i encje wszystkich dokumentów
  files/<id>.<ext>      - kopia oryginalnego pliku
  embeddings/<id>.npy   - wektory fragmentów dokumentu (wyszukiwanie semantyczne)
"""
import json
import logging
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pymupdf
import numpy as np

from nlp_pl import analyze_text, detect_language

log = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    name: str
    file_type: str
    size_bytes: int
    added_at: str
    language: str
    char_count: int
    word_count: int
    content: str
    lemmas: str = ""
    chunks: list = field(default_factory=list)        # [(start, end), ...]
    chunk_lemmas: list = field(default_factory=list)
    entities: list = field(default_factory=list)      # cache NER

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    def chunk_text(self, i: int) -> str:
        start, end = self.chunks[i]
        return self.content[start:end].strip()


def extract_text(data: bytes, suffix: str) -> str:
    if suffix == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            text = "\n".join(page.get_text() for page in pdf)
        # w PDF każda linia kończy się \n - sklejam linie w akapity
        text = re.sub(r"-\n(?=[a-ząćęłńóśźż])", "", text)   # przeniesione wyrazy
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    else:
        for encoding in ("utf-8", "cp1250", "iso-8859-2"):
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Nie rozpoznano kodowania pliku tekstowego")
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class DocumentStore:
    def __init__(self, storage_dir: str | Path = "storage"):
        self.dir = Path(storage_dir)
        self.index_file = self.dir / "documents.json"
        (self.dir / "files").mkdir(parents=True, exist_ok=True)
        (self.dir / "embeddings").mkdir(exist_ok=True)
        self.docs: dict[str, Document] = {}
        self.load_error: str | None = None   # pokazywany w GUI, jeśli odczyt się nie udał
        self._load()

    def _load(self):
        if not self.index_file.exists():
            return
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8"))
            self.docs = {d["id"]: Document(**d) for d in data}
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            # nie kasuję bazy po cichu - odkładam uszkodzony plik obok
            backup = self.index_file.with_suffix(f".bad-{datetime.now():%Y%m%d-%H%M%S}.json")
            shutil.copy(self.index_file, backup)
            self.load_error = f"Plik bazy jest uszkodzony ({e}). Kopia: {backup.name}"
            log.error(self.load_error)
        except OSError as e:
            self.load_error = f"Nie można odczytać pliku bazy: {e}"
            log.error(self.load_error)

    def _save(self):
        tmp = self.index_file.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(d) for d in self.docs.values()], ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.index_file)   # zapis atomowy - przerwany zapis nie psuje bazy

    def add(self, name: str, data: bytes, embed_fn) -> Document:
        """Dodaje dokument i od razu przygotowuje go do wyszukiwania i NER.
        embed_fn(lista tekstów) -> macierz embeddingów (z SearchEngine)."""
        suffix = Path(name).suffix.lower()
        if suffix not in (".txt", ".pdf"):
            raise ValueError("Obsługiwane są tylko pliki TXT i PDF")
        content = extract_text(data, suffix)
        if not content:
            raise ValueError(f"Plik {name} nie zawiera tekstu")

        doc_id = uuid.uuid4().hex
        while doc_id in self.docs:       # praktycznie niemożliwe, ale sprawdzam
            doc_id = uuid.uuid4().hex
        processed = analyze_text(content)
        doc = Document(
            id=doc_id, name=name, file_type=suffix[1:].upper(), size_bytes=len(data),
            added_at=datetime.now().isoformat(timespec="seconds"),
            language=detect_language(content), char_count=len(content),
            word_count=len(content.split()), content=content, **processed,
        )
        vectors = embed_fn([doc.chunk_text(i) for i in range(len(doc.chunks))])
        np.save(self.dir / "embeddings" / f"{doc_id}.npy", vectors)
        (self.dir / "files" / f"{doc_id}{suffix}").write_bytes(data)
        self.docs[doc_id] = doc
        self._save()
        return doc

    def remove(self, doc_id: str):
        doc = self.docs.pop(doc_id)
        for path in (self.dir / "embeddings" / f"{doc_id}.npy",
                     self.dir / "files" / f"{doc_id}.{doc.file_type.lower()}"):
            path.unlink(missing_ok=True)
        self._save()

    def get_all(self) -> list[Document]:
        return sorted(self.docs.values(), key=lambda d: d.name.lower())

    def get(self, doc_id: str) -> Document | None:
        return self.docs.get(doc_id)

    def embeddings(self, doc_id: str) -> np.ndarray:
        return np.load(self.dir / "embeddings" / f"{doc_id}.npy")

    def file_path(self, doc_id: str) -> Path:
        doc = self.docs[doc_id]
        return self.dir / "files" / f"{doc_id}.{doc.file_type.lower()}"

    def has_name(self, name: str) -> bool:
        return any(d.name == name for d in self.docs.values())
