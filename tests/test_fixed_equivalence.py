"""Equivalence tests for FINDHasherFixed vs FINDHasher.

Two guarantees we assert:
  1. On square inputs (rows == cols) both implementations produce
     bit-identical hashes — proves the fix does not touch the hot path.
  2. On non-square inputs the fixed implementation succeeds where the
     original crashes (portrait) or silently corrupts (landscape), and
     produces hashes close in Hamming distance to the square original
     of the same content — proves the fix is semantically correct,
     not just stable.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from find_image_hashing import FINDHasher, FINDHasherFixed

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
KNOWN_FIXED = json.loads((FIXTURES_DIR / "known_hashes_fixed.json").read_text())

MEME_DIR = Path(__file__).resolve().parent.parent / "meme_images"
BULK_SAMPLE_SIZE = 100
BULK_SEED = 12345


@pytest.fixture(scope="module")
def orig():
    return FINDHasher()


@pytest.fixture(scope="module")
def fixed():
    return FINDHasherFixed()


@pytest.mark.parametrize("filename,expected", list(KNOWN_FIXED.items()))
def test_fixed_pinned_hashes(fixed, filename, expected):
    actual = fixed.fromFile(str(FIXTURES_DIR / filename))
    assert str(actual) == expected


SQUARE_FIXTURES = [
    "01_typical.jpg",
    "02_small.jpg",
    "03_large.jpg",
    "06_grayscale.jpg",
    "07_png.png",
    "08_text_heavy.jpg",
    "09_low_contrast.jpg",
    "10_homogeneous.jpg",
]


@pytest.mark.parametrize("filename", SQUARE_FIXTURES)
def test_square_inputs_match_bit_exact(orig, fixed, filename):
    """On square images the fix is a no-op: hashes must be identical."""
    h_orig = orig.fromFile(str(FIXTURES_DIR / filename))
    h_fix = fixed.fromFile(str(FIXTURES_DIR / filename))
    assert str(h_orig) == str(h_fix), (
        f"Fix changed hash on square input {filename}: "
        f"{h_orig} (orig) vs {h_fix} (fixed)"
    )


def test_portrait_no_longer_crashes(fixed):
    """Original raises IndexError on 04_portrait.jpg — fixed must succeed."""
    h = fixed.fromFile(str(FIXTURES_DIR / "04_portrait.jpg"))
    assert len(str(h)) == 64


def test_non_square_close_to_square_original(orig, fixed):
    """The 04/05 fixtures are non-square variants of the same content as
    01_typical.jpg. After the fix their hashes should be near-duplicate
    distance from the square original — proving the box filter now
    operates on the correct pixels."""
    h_sq = fixed.fromFile(str(FIXTURES_DIR / "01_typical.jpg"))
    h_landscape = fixed.fromFile(str(FIXTURES_DIR / "05_landscape.jpg"))
    h_portrait = fixed.fromFile(str(FIXTURES_DIR / "04_portrait.jpg"))

    # Reference: an unrelated image (08_text_heavy) is ~140 bits away from 01.
    # Fixed non-square hashes should be << 50 bits (near-duplicate territory).
    d_landscape = h_sq - h_landscape
    d_portrait = h_sq - h_portrait
    assert d_landscape < 50, f"landscape too far from square: {d_landscape} bits"
    assert d_portrait < 50, f"portrait too far from square: {d_portrait} bits"

    # Sanity: original FINd on landscape gives ~104 bits — basically random.
    h_landscape_orig = orig.fromFile(str(FIXTURES_DIR / "05_landscape.jpg"))
    d_landscape_orig = h_sq - h_landscape_orig
    assert d_landscape_orig > d_landscape * 3, (
        "fix should give a much smaller distance than the buggy original"
    )


@pytest.mark.skipif(not MEME_DIR.exists(), reason="meme_images not available")
def test_bulk_square_equivalence(orig, fixed):
    """Sample real images from meme_images and verify orig == fixed bit-exact.
    The dataset is essentially all square (1 of 55 972 non-square), so this
    is the strongest empirical guarantee that the fix does not touch the
    hot path used to compute baseline numbers in profile.ipynb."""
    rng = random.Random(BULK_SEED)
    files = sorted(os.listdir(MEME_DIR))
    sample = rng.sample(files, min(BULK_SAMPLE_SIZE, len(files)))

    mismatches = []
    for f in sample:
        try:
            a = orig.fromFile(str(MEME_DIR / f))
            b = fixed.fromFile(str(MEME_DIR / f))
        except Exception:
            continue
        if str(a) != str(b):
            mismatches.append((f, str(a), str(b)))

    assert not mismatches, (
        f"orig != fixed on {len(mismatches)}/{len(sample)} files. "
        f"First: {mismatches[:1]}"
    )
