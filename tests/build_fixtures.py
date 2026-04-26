"""Build 10 pinned regression fixtures covering different code paths of FIND.

Run once from the summative2026 directory:
    .venv/bin/python tests/build_fixtures.py

Produces tests/fixtures/*.{jpg,png} and tests/fixtures/known_hashes.json.
Re-running overwrites the fixtures and re-records the anchors, so only run
this when the reference implementation is what you want to pin.
"""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from FINd import FINDHasher


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
MEME_DIR = REPO_ROOT / "meme_images"


def sample_candidates(n=200, seed=0):
    random.seed(seed)
    files = sorted(os.listdir(MEME_DIR))
    return random.sample(files, n)


def pick_by_variance(candidates, high=True):
    """Return the filename with highest (or lowest) luminance variance."""
    best = None
    best_var = -1 if high else float("inf")
    for f in candidates:
        with Image.open(MEME_DIR / f) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
        v = float(arr.var())
        if (high and v > best_var) or (not high and v < best_var):
            best, best_var = f, v
    return best, best_var


def main():
    FIXTURES_DIR.mkdir(exist_ok=True)
    candidates = sample_candidates()

    with Image.open(MEME_DIR / candidates[0]) as im:
        typical = im.convert("RGB").copy()
    typical.save(FIXTURES_DIR / "01_typical.jpg", quality=92)

    smallest = min(
        candidates,
        key=lambda f: max(Image.open(MEME_DIR / f).size),
    )
    with Image.open(MEME_DIR / smallest) as im:
        im.convert("RGB").save(FIXTURES_DIR / "02_small.jpg", quality=92)

    large = typical.resize((1200, 1200), Image.BICUBIC)
    large.save(FIXTURES_DIR / "03_large.jpg", quality=92)

    portrait = typical.resize((200, 400), Image.BICUBIC)
    portrait.save(FIXTURES_DIR / "04_portrait.jpg", quality=92)

    landscape = typical.resize((400, 200), Image.BICUBIC)
    landscape.save(FIXTURES_DIR / "05_landscape.jpg", quality=92)

    typical.convert("L").save(FIXTURES_DIR / "06_grayscale.jpg", quality=92)

    typical.save(FIXTURES_DIR / "07_png.png")

    text_heavy, v_high = pick_by_variance(candidates, high=True)
    with Image.open(MEME_DIR / text_heavy) as im:
        im.convert("RGB").save(FIXTURES_DIR / "08_text_heavy.jpg", quality=92)

    low_contrast, v_low = pick_by_variance(candidates, high=False)
    with Image.open(MEME_DIR / low_contrast) as im:
        im.convert("RGB").save(FIXTURES_DIR / "09_low_contrast.jpg", quality=92)

    homogeneous = Image.new("RGB", (250, 250), (128, 128, 128))
    homogeneous.save(FIXTURES_DIR / "10_homogeneous.jpg", quality=92)

    hasher = FINDHasher()
    known = {}
    for f in sorted(os.listdir(FIXTURES_DIR)):
        if f == "known_hashes.json":
            continue
        try:
            h = hasher.fromFile(str(FIXTURES_DIR / f))
            known[f] = str(h)
        except Exception as e:
            known[f] = f"CRASHES:{type(e).__name__}"

    (FIXTURES_DIR / "known_hashes.json").write_text(
        json.dumps(known, indent=2, sort_keys=True) + "\n"
    )

    print("Built fixtures:")
    for f, h in sorted(known.items()):
        print(f"  {f}: {h}")
    print(f"\nSource picks: text_heavy={text_heavy} (var={v_high:.1f}), "
          f"low_contrast={low_contrast} (var={v_low:.1f}), smallest={smallest}")


if __name__ == "__main__":
    main()
