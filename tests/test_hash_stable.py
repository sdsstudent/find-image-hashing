"""Pinned regression tests: the 10 fixture hashes must not change.

The fixtures in tests/fixtures/ were hashed once with the reference FIND
implementation and pinned in known_hashes.json. Any future change that
shifts one of these hashes must be reviewed explicitly.

Fixtures named "CRASHES:<ExceptionType>" in known_hashes.json document
known bugs in the current implementation (see report). These are expected
to raise and are checked with pytest.xfail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from FINd import FINDHasher

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KNOWN = json.loads((FIXTURES_DIR / "known_hashes.json").read_text())


@pytest.fixture(scope="module")
def hasher():
    return FINDHasher()


@pytest.mark.parametrize(
    "filename,expected",
    [(f, h) for f, h in KNOWN.items() if not h.startswith("CRASHES")],
)
def test_pinned_hash_matches(hasher, filename, expected):
    actual = hasher.fromFile(str(FIXTURES_DIR / filename))
    assert str(actual) == expected, (
        f"Hash drift on {filename}: got {actual}, expected {expected}"
    )


@pytest.mark.parametrize(
    "filename,marker",
    [(f, h) for f, h in KNOWN.items() if h.startswith("CRASHES")],
)
def test_documented_bug(hasher, filename, marker):
    exc_name = marker.split(":", 1)[1]
    expected_exc = getattr(__builtins__, exc_name, Exception)
    with pytest.raises(expected_exc):
        hasher.fromFile(str(FIXTURES_DIR / filename))


def test_same_image_same_hash(hasher):
    a = hasher.fromFile(str(FIXTURES_DIR / "01_typical.jpg"))
    b = hasher.fromFile(str(FIXTURES_DIR / "01_typical.jpg"))
    assert str(a) == str(b), "hashing the same file twice must be deterministic"


def test_different_images_different_hash(hasher):
    a = hasher.fromFile(str(FIXTURES_DIR / "01_typical.jpg"))
    b = hasher.fromFile(str(FIXTURES_DIR / "08_text_heavy.jpg"))
    assert str(a) != str(b)
    assert (a - b) > 30, "sanity: two unrelated images should differ by more than a few bits"
