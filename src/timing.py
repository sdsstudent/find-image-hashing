"""Timing helpers: wall-clock, CPU, I/O-vs-compute split, scaling sweep."""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path
from typing import Callable

from PIL import Image


def time_hash_series(
    hasher,
    files: list[str],
    img_dir: str | Path,
) -> list[tuple[str, float]]:
    """Wall-clock per file via fromFile (includes decode + I/O)."""
    results = []
    for f in files:
        t0 = time.perf_counter()
        hasher.fromFile(str(Path(img_dir) / f))
        results.append((f, time.perf_counter() - t0))
    return results


def time_hash_preloaded(
    hasher,
    files: list[str],
    img_dir: str | Path,
) -> list[tuple[str, float]]:
    """Wall-clock with images preloaded into PIL — excludes disk I/O + decode."""
    preloaded = [Image.open(Path(img_dir) / f).copy() for f in files]
    results = []
    for f, img in zip(files, preloaded):
        t0 = time.perf_counter()
        hasher.fromImage(img)
        results.append((f, time.perf_counter() - t0))
    return results


def measure_io_vs_compute(
    hasher,
    files: list[str],
    img_dir: str | Path,
) -> dict[str, float]:
    """Split total per-image cost into I/O+decode vs pure compute."""
    t_with_io = [dt for _, dt in time_hash_series(hasher, files, img_dir)]
    t_no_io = [dt for _, dt in time_hash_preloaded(hasher, files, img_dir)]
    mean_full = sum(t_with_io) / len(t_with_io)
    mean_compute = sum(t_no_io) / len(t_no_io)
    return {
        "mean_full": mean_full,
        "mean_compute": mean_compute,
        "mean_io_plus_decode": mean_full - mean_compute,
        "io_fraction": (mean_full - mean_compute) / mean_full,
        "n": len(files),
    }


def run_cprofile(
    target: Callable[[], None],
    sort: str = "cumulative",
    top: int = 20,
) -> tuple[pstats.Stats, str]:
    """Run cProfile on target() and return (stats, formatted_top)."""
    profiler = cProfile.Profile()
    profiler.enable()
    target()
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).strip_dirs().sort_stats(sort)
    stats.print_stats(top)
    return stats, buf.getvalue()


def scaling_sweep(
    hasher,
    all_files: list[str],
    img_dir: str | Path,
    sizes: list[int] = (50, 100, 200, 500),
) -> list[dict]:
    """Measure total runtime at different N to check linearity.

    Requires len(all_files) >= max(sizes); raises otherwise so the caller
    can't silently report fake per-image numbers (see earlier bug where
    all_files[:500] returned 100 and per_image_s came out 5x too small).
    """
    if len(all_files) < max(sizes):
        raise ValueError(
            f"scaling_sweep needs at least {max(sizes)} files, got {len(all_files)}"
        )
    rows = []
    for n in sizes:
        subset = all_files[:n]
        t0 = time.perf_counter()
        for f in subset:
            hasher.fromFile(str(Path(img_dir) / f))
        total = time.perf_counter() - t0
        rows.append({"n": n, "total_s": total, "per_image_s": total / n})
    return rows
