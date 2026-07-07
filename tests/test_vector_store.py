"""Tests the content-hash fingerprint that powers the vector-store cache.

These exercise pure hashing logic only -- no model download or Ollama required.
"""
from src.vector_store import _fingerprint


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


def test_fingerprint_is_deterministic(tmp_path):
    a = _write(tmp_path, "a.pdf", b"hello world")
    assert _fingerprint(a) == _fingerprint(a)


def test_same_content_same_fingerprint_regardless_of_filename(tmp_path):
    a = _write(tmp_path, "a.pdf", b"identical bytes")
    b = _write(tmp_path, "b.pdf", b"identical bytes")
    assert _fingerprint(a) == _fingerprint(b)


def test_different_content_gives_different_fingerprint(tmp_path):
    a = _write(tmp_path, "a.pdf", b"content one")
    b = _write(tmp_path, "b.pdf", b"content two")
    assert _fingerprint(a) != _fingerprint(b)
