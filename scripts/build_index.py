"""Dodaje wszystkie pliki z sample_docs/ do bazy (to samo co przycisk w GUI).

Uruchomienie: python scripts/build_index.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from document_store import DocumentStore  # noqa: E402
from search_engine import embed  # noqa: E402


def main():
    store = DocumentStore(ROOT / "storage")
    files = sorted(p for p in (ROOT / "sample_docs").iterdir() if p.suffix in (".txt", ".pdf"))
    for path in files:
        if store.has_name(path.name):
            print(f"jest już w bazie: {path.name}")
            continue
        t0 = time.time()
        doc = store.add(path.name, path.read_bytes(), embed)
        print(f"{path.name:50s} {doc.word_count:6d} słów, {len(doc.chunks):3d} fragm., "
              f"{doc.entity_count:4d} encji, {time.time() - t0:.1f} s")
    print(f"W bazie: {len(store.docs)} dokumentów")


if __name__ == "__main__":
    main()
