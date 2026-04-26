"""Bulk equivalence test: sample 200 real images and verify hash stability.

This test is a placeholder stub for Part 2 (after optimization). Today it
just confirms the reference implementation is deterministic on a sample
of meme_images. Once an optimized hasher exists, extend this to compare
reference vs optimized outputs on the same sample.

Skipped automatically if meme_images/ is not present (e.g. in CI).
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from FINd import FINDHasher

MEME_DIR = Path(__file__).resolve().parent.parent / "meme_images"
SAMPLE_SIZE = 200
SEED = 12345


@pytest.mark.skipif(not MEME_DIR.exists(), reason="meme_images not available")
def test_reference_is_deterministic():
    """Calling the same reference implementation twice must yield identical hashes."""
    rng = random.Random(SEED)
    files = sorted(os.listdir(MEME_DIR))
    sample = rng.sample(files, min(SAMPLE_SIZE, len(files)))

    h1 = FINDHasher()
    h2 = FINDHasher()
    mismatches = []
    for f in sample:
        a = h1.fromFile(str(MEME_DIR / f))
        b = h2.fromFile(str(MEME_DIR / f))
        if str(a) != str(b):
            mismatches.append(f)

    assert not mismatches, f"non-deterministic on {len(mismatches)}/{len(sample)} files: {mismatches[:3]}"
