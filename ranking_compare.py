"""Porównanie rankingów trzech metod dla jednego zapytania.

Bierzemy sumę (unię) dokumentów z top-k wszystkich metod. Dokument, którego
metoda nie umieściła w swoim top-k, dostaje dla niej pozycję k+1 ("dalej niż k").
Na takich wektorach pozycji liczę korelację rang Spearmana i tau Kendalla,
a dodatkowo Jaccard@k = |A ∩ B| / |A ∪ B| dla zbiorów top-k.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def positions_table(rankings: dict[str, list[str]], names: dict[str, str], k: int) -> pd.DataFrame:
    """rankings: metoda -> lista id dokumentów w kolejności. Wiersz = dokument."""
    order = []
    for ids in rankings.values():
        for d in ids:
            if d not in order:
                order.append(d)
    rows = []
    for d in order:
        row = {"dokument": names.get(d, d)}
        for method, ids in rankings.items():
            row[method] = ids.index(d) + 1 if d in ids else None
        filled = [p if p is not None else k + 1 for p in (row[m] for m in rankings)]
        # rozrzut pozycji: 0 = wszystkie metody zgodne co do miejsca tego dokumentu
        row["rozrzut"] = max(filled) - min(filled)
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        best = table[list(rankings)].fillna(k + 1).min(axis=1)
        table = table.assign(_best=best).sort_values(["_best", "rozrzut"]).drop(columns="_best")
    return table.reset_index(drop=True)


def pairwise_stats(rankings: dict[str, list[str]], k: int) -> pd.DataFrame:
    union = sorted({d for ids in rankings.values() for d in ids})
    rows = []
    for a, b in combinations(rankings, 2):
        pa = [rankings[a].index(d) + 1 if d in rankings[a] else k + 1 for d in union]
        pb = [rankings[b].index(d) + 1 if d in rankings[b] else k + 1 for d in union]
        set_a, set_b = set(rankings[a]), set(rankings[b])
        jacc = len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else np.nan
        if len(union) < 2 or len(set(pa)) < 2 or len(set(pb)) < 2:
            rho = tau = np.nan   # korelacja nieokreślona (np. wszystko na tej samej pozycji)
        else:
            rho = spearmanr(pa, pb).statistic
            tau = kendalltau(pa, pb).statistic
        rows.append({"para metod": f"{a} - {b}", "Spearman": rho, "Kendall tau": tau,
                     f"Jaccard@{k}": jacc, "wspólne w top-k": len(set_a & set_b)})
    return pd.DataFrame(rows)
