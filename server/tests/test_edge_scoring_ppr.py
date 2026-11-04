"""Edge-case tests for scoring.py and ppr.py. Synthetic data only."""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scoring  # noqa: E402
from ppr import personalized_pagerank, related_notes  # noqa: E402


def test_no_replacement_characters_in_relation_keywords():
    """U+FFFD in a keyword means a file-encoding accident made it unmatchable."""
    bad = [k for k in scoring.QUERY_RELATION_KEYWORDS if "�" in k]
    assert bad == []


def test_chinese_cause_keyword_triggers_boost():
    rels = [{"source_note": "N", "type": "causes"}]
    assert scoring.get_relation_weight("N", "造成腎衰竭的原因", rels) == scoring.RELATION_BOOST
    assert scoring.get_relation_weight("N", "造成", rels) == scoring.RELATION_BOOST


def test_relation_weight_empty_inputs():
    rels = [{"source_note": "N", "type": "treats"}]
    assert scoring.get_relation_weight("N", "", rels) == 1.0
    assert scoring.get_relation_weight("N", "治療", []) == 1.0
    assert scoring.get_relation_weight("N", "   ！？", rels) == 1.0
    assert scoring.get_relation_weight("Other", "治療", rels) == 1.0


def test_relation_weight_case_insensitive_and_fullwidth_punct():
    rels = [{"source_note": "N", "type": "treats"}]
    assert scoring.get_relation_weight("N", "TREATMENT？", rels) == scoring.RELATION_BOOST


def test_recency_none_future_and_ancient():
    now = 1_700_000_000.0
    assert scoring.get_recency_weight(None, now) == 1.0
    assert scoring.get_recency_weight(now + 10**7, now) == 1.0   # future mtime
    assert scoring.get_recency_weight(0.0, now) == scoring.RECENCY_FLOOR
    assert scoring.get_recency_weight(now, now) == 1.0


def test_recency_monotonic_nonincreasing():
    now = 1_700_000_000.0
    ws = [scoring.get_recency_weight(now - d * 86400, now) for d in (0, 30, 365, 3650)]
    assert ws == sorted(ws, reverse=True)


def test_rerank_empty_list():
    assert scoring.rerank([]) == []


def test_rerank_missing_and_none_similarity():
    rs = [{"folder": "a"}, {"folder": "b", "similarity": None}, {"folder": "c", "similarity": 0.5}]
    out = scoring.rerank(rs, boost_recent=False)
    assert out[0]["folder"] == "c"
    assert out[1]["similarity"] == 0 and out[2]["similarity"] == 0


def test_rerank_ties_keep_input_order():
    rs = [{"note": n, "folder": "", "similarity": 0.5} for n in "abcd"]
    out = scoring.rerank(rs, boost_recent=False)
    assert [r["note"] for r in out] == list("abcd")


def test_rerank_sorts_desc_and_records_raw():
    rs = [{"note": "lo", "folder": "", "similarity": 0.2},
          {"note": "hi", "folder": "", "similarity": 0.9}]
    out = scoring.rerank(rs, boost_recent=False)
    assert [r["note"] for r in out] == ["hi", "lo"]
    assert out[0]["raw_similarity"] == 0.9


def test_rerank_rank_key_consumed_and_not_leaked():
    rs = [{"note": "a", "folder": "", "similarity": 0.9, "rrf": 0.01},
          {"note": "b", "folder": "", "similarity": 0.1, "rrf": 0.03}]
    out = scoring.rerank(rs, boost_recent=False, rank_key="rrf")
    assert [r["note"] for r in out] == ["b", "a"]
    assert all("rrf" not in r and "_rank_score" not in r for r in out)
    assert out[0]["similarity"] == 0.1          # similarity stays a similarity


def test_rerank_rank_key_missing_or_none_is_zero():
    rs = [{"note": "a", "folder": "", "similarity": 0.5},
          {"note": "b", "folder": "", "similarity": 0.5, "rrf": None},
          {"note": "c", "folder": "", "similarity": 0.5, "rrf": 0.02}]
    out = scoring.rerank(rs, boost_recent=False, rank_key="rrf")
    assert out[0]["note"] == "c"


def test_rerank_recent_note_beats_equal_old_note():
    now = time.time()
    rs = [{"note": "old", "folder": "", "similarity": 0.8, "mtime": now - 5000 * 86400},
          {"note": "new", "folder": "", "similarity": 0.8, "mtime": now}]
    assert scoring.rerank(rs)[0]["note"] == "new"


def test_rerank_unknown_folder_neutral_and_nan_free():
    rs = [{"note": "x", "folder": "從沒見過的資料夾 ", "similarity": 0.7}]
    out = scoring.rerank(rs, boost_recent=False)
    assert out[0]["similarity"] == 0.7 and not math.isnan(out[0]["similarity"])


# ---- ppr ------------------------------------------------------------------

def test_ppr_no_seeds_or_blank_seeds():
    assert personalized_pagerank({"a": ["b"]}, []) == []
    assert personalized_pagerank({"a": ["b"]}, ["", None]) == []


def test_ppr_empty_graph_and_isolated_seed():
    assert personalized_pagerank({}, ["a"]) == []
    assert related_notes({"a": []}, "a") == []


def test_ppr_unknown_seed_returns_empty():
    assert personalized_pagerank({"a": ["b"]}, ["zzz"]) == []


def test_ppr_excludes_seeds_and_orders_by_proximity():
    g = {"a": ["b"], "b": ["c"], "c": []}
    ranked = personalized_pagerank(g, ["a"])
    names = [n for n, _ in ranked]
    assert "a" not in names and names.index("b") < names.index("c")


def test_ppr_self_loop_and_cycle_terminate():
    g = {"a": ["a", "b"], "b": ["a"]}
    ranked = personalized_pagerank(g, ["a"])
    assert [n for n, _ in ranked] == ["b"]


def test_ppr_dangling_link_target_and_chinese_names():
    g = {"腎臟病": ["透析", "不存在的筆記"], "透析": ["腎臟病"]}
    names = [n for n, _ in related_notes(g, "腎臟病")]
    assert "透析" in names and "不存在的筆記" in names   # dangling targets are returned as-is


def test_ppr_top_k_and_exclude():
    g = {"a": ["b", "c", "d"]}
    assert len(personalized_pagerank(g, ["a"], top_k=2)) == 2
    assert personalized_pagerank(g, ["a"], top_k=0) == []
    ex = personalized_pagerank(g, ["a"], exclude={"b", "a"})
    assert "b" not in [n for n, _ in ex]


def test_ppr_duplicate_seeds_still_valid():
    g = {"a": ["b"], "b": []}
    ranked = personalized_pagerank(g, ["a", "a"])
    assert ranked and all(sc > 0 for _, sc in ranked)
