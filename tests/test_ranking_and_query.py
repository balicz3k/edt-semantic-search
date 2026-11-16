import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranking_compare import pairwise_stats, positions_table  # noqa: E402
from search_engine import make_snippet, parse_query, term_pattern  # noqa: E402


def test_identical_rankings():
    r = {"A": ["d1", "d2", "d3"], "B": ["d1", "d2", "d3"]}
    s = pairwise_stats(r, k=3).iloc[0]
    assert s["Spearman"] == pytest.approx(1.0)
    assert s["Kendall tau"] == pytest.approx(1.0)
    assert s["Jaccard@3"] == 1.0


def test_reversed_rankings():
    r = {"A": ["d1", "d2", "d3"], "B": ["d3", "d2", "d1"]}
    s = pairwise_stats(r, k=3).iloc[0]
    assert s["Spearman"] == pytest.approx(-1.0)


def test_missing_document_gets_k_plus_1():
    r = {"A": ["d1", "d2"], "B": ["d2", "d3"]}
    t = positions_table(r, {}, k=2).set_index("dokument")
    assert pd.isna(t.loc["d3", "A"])
    assert t.loc["d3", "rozrzut"] == 3 - 2   # A: k+1 = 3, B: 2
    s = pairwise_stats(r, k=2).iloc[0]
    assert s["Jaccard@2"] == pytest.approx(1 / 3)


def test_parse_query_phrases():
    assert parse_query('"unia lubelska" Jagiellonowie') == ["unia lubelska", "jagiellonowie"]


def test_keyword_pattern_is_exact_word():
    p = term_pattern("kraków")
    assert p.search("Miasto Kraków leży")
    assert not p.search("w Krakowie")       # słowa kluczowe nie znają odmiany
    assert term_pattern("unia lubelska").search("Unia\nlubelska")


def test_snippet_marks_hits():
    s = make_snippet("Bitwa pod Grunwaldem w 1410 roku", [term_pattern("grunwaldem")])
    assert "<mark>Grunwaldem</mark>" in s
