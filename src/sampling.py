"""Dataset sampling helpers for FIND profiling runs."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path


def group_by_cluster(img_dir: str | Path) -> dict[str, list[str]]:
    """Group filenames by the 4-character cluster prefix."""
    clusters: dict[str, list[str]] = defaultdict(list)
    for f in sorted(os.listdir(img_dir)):
        clusters[f[:4]].append(f)
    return dict(clusters)


def load_subset(
    img_dir: str | Path,
    n: int = 100,
    seed: int = 42,
) -> list[str]:
    """Random subset of filenames (not paths) from the directory.

    Uses a seeded RNG for reproducibility across runs.
    """
    rng = random.Random(seed)
    files = sorted(os.listdir(img_dir))
    return rng.sample(files, min(n, len(files)))


def cluster_sample(
    img_dir: str | Path,
    n_clusters: int = 10,
    per_cluster: int = 3,
    seed: int = 42,
) -> list[str]:
    """Balanced subset: pick n_clusters distinct clusters, take per_cluster images each."""
    rng = random.Random(seed)
    clusters = group_by_cluster(img_dir)
    eligible = [cid for cid, imgs in clusters.items() if len(imgs) >= per_cluster]
    picked = rng.sample(eligible, min(n_clusters, len(eligible)))
    subset: list[str] = []
    for cid in picked:
        subset.extend(clusters[cid][:per_cluster])
    return subset
