"""Wykresy do prezentacji z plików wyników notatnika (results/*.csv) -> docs/img/."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
METHODS = ["Słowa kluczowe", "TF-IDF", "Semantyczne"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rr = pd.read_csv(ROOT / "results" / "zapytania_rr.csv")
    mrr = rr.pivot_table(index="rodzaj", columns="metoda", values="RR")[METHODS]
    order = [r for r in ["dosłowne", "parafraza", "odmiana", "wieloznaczne"] if r in mrr.index]
    ax = mrr.loc[order].plot.bar(figsize=(6.4, 3.6), rot=0, width=0.75)
    ax.set_ylabel("MRR@5"); ax.set_xlabel(""); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax.set_title("Trafność wg rodzaju zapytania (13 zapytań)")
    plt.tight_layout(); plt.savefig(OUT / "mrr.png", dpi=200); plt.close()

    corr = pd.read_csv(ROOT / "results" / "korelacje_srednie.csv", index_col=0)
    ax = corr.plot.barh(figsize=(6.4, 3.0), width=0.75)
    ax.set_xlim(-0.1, 1); ax.set_ylabel("")
    ax.set_title("Podobieństwo rankingów (średnio po zapytaniach)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(OUT / "korelacje.png", dpi=200); plt.close()
    print("zapisano docs/img/mrr.png i korelacje.png")


if __name__ == "__main__":
    main()
