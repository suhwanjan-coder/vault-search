"""Edge-case tests for indexer.py helpers (frontmatter, chunking, file collection,
hash cache). Synthetic notes only. lancedb/ollama/pyarrow are stubbed when absent
so these run on a machine that has never installed them."""
import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def ix(monkeypatch):
    for m in ("lancedb", "ollama", "pyarrow"):
        try:
            importlib.import_module(m)
        except ImportError:
            monkeypatch.setitem(sys.modules, m, MagicMock())
    sys.modules.pop("indexer", None)
    mod = importlib.import_module("indexer")
    yield mod
    sys.modules.pop("indexer", None)


# ---- parse_frontmatter ------------------------------------------------------

def test_fm_none_and_unclosed(ix):
    assert ix.parse_frontmatter("just text") == ({}, "just text")
    assert ix.parse_frontmatter("---\ntags: [a]\nno closing") == ({}, "---\ntags: [a]\nno closing")
    assert ix.parse_frontmatter("") == ({}, "")


def test_fm_empty_block(ix):
    assert ix.parse_frontmatter("---\n---\nbody") == ({}, "body")


def test_fm_inline_and_block_lists_and_crlf(ix):
    meta, body = ix.parse_frontmatter('---\r\ntags: ["甲", \'乙\']\r\naliases:\r\n  - 別名一\r\n  - 別名二\r\n---\r\n內文')
    assert meta["tags"] == ["甲", "乙"]
    assert meta["aliases"] == ["別名一", "別名二"]
    assert body == "內文"


def test_fm_scalar_tags_and_fullwidth_colon_is_not_a_key(ix):
    meta, _ = ix.parse_frontmatter("---\ntags: solo\n標題：全形冒號\n---\nb")
    assert meta == {"tags": "solo"}


def test_fm_garbage_does_not_raise(ix):
    meta, body = ix.parse_frontmatter("---\n: bad\n  - orphan\n[[x]]\n---\nz")
    assert body == "z" and isinstance(meta, dict)


def test_fm_with_utf8_bom(ix):
    """Windows editors write a BOM; the frontmatter must still be recognised."""
    meta, body = ix.parse_frontmatter("﻿---\ntags: [a]\n---\nbody text")
    assert meta == {"tags": ["a"]}
    assert body == "body text"


def test_fm_only_frontmatter_has_empty_body(ix):
    assert ix.parse_frontmatter("---\ntags: x\n---")[1] == ""


# ---- chunk_by_headings --------------------------------------------------------

LONG = "內容" * 20


def test_chunk_empty_and_whitespace(ix):
    assert ix.chunk_by_headings("", "n") == []
    assert ix.chunk_by_headings("  \n\t\n", "n") == []


def test_chunk_short_note_is_dropped(ix):
    """Documented behaviour: <=20 chars is skipped (a one-line note is not indexed)."""
    assert ix.chunk_by_headings("太短了", "n") == []


def test_chunk_no_heading_uses_note_name(ix):
    assert ix.chunk_by_headings(LONG, "筆記") == [{"text": LONG, "section": "筆記"}]


def test_chunk_splits_on_h1_to_h3_not_h4_or_nospace(ix):
    body = f"# 一\n{LONG}\n## 二\n{LONG}\n### 三\n{LONG}\n#### 四\n{LONG}\n#五\n{LONG}"
    secs = [c["section"] for c in ix.chunk_by_headings(body, "n")]
    assert secs == ["一", "二", "三"]
    assert "#### 四" in ix.chunk_by_headings(body, "n")[2]["text"]


def test_chunk_preamble_before_first_heading_uses_note_name(ix):
    out = ix.chunk_by_headings(f"{LONG}\n# 標題\n{LONG}", "N")
    assert [c["section"] for c in out] == ["N", "標題"]


def test_chunk_heading_with_trailing_spaces_and_crlf(ix):
    out = ix.chunk_by_headings(f"# 標題  \r\n{LONG}\r\n", "n")
    assert out[0]["section"] == "標題"


def test_chunk_very_long_body_is_single_chunk_untruncated(ix):
    big = "字" * 100_000
    out = ix.chunk_by_headings(big, "n")
    assert len(out) == 1 and len(out[0]["text"]) == 100_000   # truncation is index_vault's job


# ---- small helpers ------------------------------------------------------------

def test_get_folder(ix):
    assert ix.get_folder("a.md") == ""
    assert ix.get_folder("資料夾/子/筆記 1.md") == "資料夾"
    assert ix.get_folder("") == ""


def test_file_hash_stable_and_unicode(ix):
    assert ix.file_hash("病例") == hashlib.sha256("病例".encode()).hexdigest()
    assert ix.file_hash("a") != ix.file_hash("a ")
    assert ix.file_hash("") == hashlib.sha256(b"").hexdigest()


def test_escape_sql(ix):
    assert ix._escape_sql("it's") == "it''s"
    assert ix._escape_sql("a'b'c") == "a''b''c"
    assert ix._escape_sql("無引號") == "無引號"


# ---- collect_md_files -----------------------------------------------------------

def _touch(p: Path, text="x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_collect_skips_hidden_skip_folders_and_non_md(ix, tmp_path):
    _touch(tmp_path / "ok.md")
    _touch(tmp_path / "含 空格 & 符號(1).md")
    _touch(tmp_path / "sub" / "深.md")
    _touch(tmp_path / ".hidden.md")
    _touch(tmp_path / ".git" / "x.md")
    _touch(tmp_path / ".obsidian" / "x.md")
    _touch(tmp_path / "node_modules" / "x.md")
    _touch(tmp_path / "note.txt")
    _touch(tmp_path / "UPPER.MD")   # rglob("*.md") is case-sensitive on Linux
    got = {p.relative_to(tmp_path).as_posix() for p in ix.collect_md_files(tmp_path)}
    assert got == {"ok.md", "含 空格 & 符號(1).md", "sub/深.md"}


def test_collect_duplicate_basenames_in_different_folders(ix, tmp_path):
    _touch(tmp_path / "a" / "same.md")
    _touch(tmp_path / "b" / "same.md")
    assert len(ix.collect_md_files(tmp_path)) == 2


def test_collect_empty_and_missing_vault(ix, tmp_path):
    assert ix.collect_md_files(tmp_path) == []
    assert ix.collect_md_files(tmp_path / "nope") == []


# ---- hash cache -------------------------------------------------------------------

def test_hash_cache_roundtrip_unicode(ix, tmp_path, monkeypatch):
    monkeypatch.setattr(ix, "HASH_CACHE_PATH", tmp_path / "deep" / "h.json")
    ix.save_hash_cache({"資料夾/筆記.md": "abc"})
    assert ix.load_hash_cache() == {"資料夾/筆記.md": "abc"}


def test_hash_cache_missing_file(ix, tmp_path, monkeypatch):
    monkeypatch.setattr(ix, "HASH_CACHE_PATH", tmp_path / "none.json")
    assert ix.load_hash_cache() == {}


@pytest.mark.parametrize("junk", ["", "{not json", "\x00\x01", "[1,2,3]", "null"])
def test_hash_cache_corrupt_degrades_to_empty(ix, tmp_path, monkeypatch, junk):
    """A damaged cache must not crash the incremental run; empty = reindex everything."""
    p = tmp_path / "h.json"
    p.write_text(junk, encoding="utf-8")
    monkeypatch.setattr(ix, "HASH_CACHE_PATH", p)
    assert ix.load_hash_cache() == {}


def test_hash_cache_non_utf8_bytes(ix, tmp_path, monkeypatch):
    p = tmp_path / "h.json"
    p.write_bytes(b"\xff\xfe\x00bad")
    monkeypatch.setattr(ix, "HASH_CACHE_PATH", p)
    assert ix.load_hash_cache() == {}
