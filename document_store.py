"""Zarządzanie dokumentami: dodawanie, indeksowanie, usuwanie."""
import json
import uuid
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    import fitz  # pymupdf
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


@dataclass
class Document:
    id: str
    name: str
    content: str
    path: str
    added_at: str
    char_count: int


class DocumentStore:
    def __init__(self, storage_dir: str = "storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._docs: dict[str, Document] = {}
        self._index_file = self.storage_dir / "documents.json"
        self._load()

    def add_document(self, file_path: str | Path) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Plik nie istnieje: {path}")

        content = self._extract_text(path)
        if not content.strip():
            raise ValueError(f"Nie można odczytać treści z pliku: {path.name}")

        doc_id = str(uuid.uuid4())[:8]
        doc = Document(
            id=doc_id,
            name=path.name,
            content=content,
            path=str(path.resolve()),
            added_at=datetime.now().isoformat(),
            char_count=len(content),
        )
        self._docs[doc_id] = doc
        self._save()
        return doc

    def remove_document(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            raise KeyError(f"Dokument o ID {doc_id} nie istnieje")
        del self._docs[doc_id]
        self._save()

    def get_all(self) -> list[Document]:
        return list(self._docs.values())

    def get_by_id(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        else:
            # TXT i inne pliki tekstowe
            for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1250"):
                try:
                    return path.read_text(encoding=encoding)
                except (UnicodeDecodeError, ValueError):
                    continue
            raise ValueError(f"Nie można odczytać pliku: {path.name}")

    def _extract_pdf(self, path: Path) -> str:
        if not HAS_FITZ:
            raise ImportError("pymupdf nie jest zainstalowany — PDF nie jest obsługiwany")
        doc = fitz.open(str(path))
        texts = []
        for page in doc:
            texts.append(page.get_text())
        doc.close()
        return "\n".join(texts)

    def _save(self) -> None:
        data = {doc_id: asdict(doc) for doc_id, doc in self._docs.items()}
        self._index_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._index_file.exists():
            return
        try:
            data = json.loads(self._index_file.read_text(encoding="utf-8"))
            self._docs = {doc_id: Document(**doc_data) for doc_id, doc_data in data.items()}
        except Exception:
            self._docs = {}
