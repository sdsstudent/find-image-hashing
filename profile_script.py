"""Reproducible baseline profile for FINd.

Run from the summative2026 directory:
    .venv/bin/python profile_script.py

Produces:
    baseline.prof     — cProfile stats for pstats/snakeviz inspection
    figures/*.png     — summary plots (overwritten on each run)

This is the scriptable counterpart to profile.ipynb. The notebook is for
interactive exploration; this script is for CI-friendly reproducibility.
"""

from __future__ import annotations

import cProfile
import json
import pstats
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from find_image_hashing import FINDHasher
from src.plots import (
    plot_extrapolation,
    plot_io_split,
    plot_latency_distribution,
    plot_scaling,
    plot_time_breakdown,
)
from src.sampling import load_subset
from src.timing import (
    measure_io_vs_compute,
    scaling_sweep,
    time_hash_series,
)

IMG_DIR = ROOT / "meme_images"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_PROFILE = 100
N_SCALING = [50, 100, 200, 500]


def main():
    hasher = FINDHasher()
    subset = load_subset(IMG_DIR, n=N_PROFILE, seed=42)
    scaling_subset = load_subset(IMG_DIR, n=max(N_SCALING), seed=42)

    times = [dt for _, dt in time_hash_series(hasher, subset, IMG_DIR)]

    profiler = cProfile.Profile()
    profiler.enable()
    for f in subset:
        hasher.fromFile(str(IMG_DIR / f))
    profiler.disable()
    profiler.dump_stats(str(ROOT / "baseline.prof"))
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("tottime")

    split = measure_io_vs_compute(hasher, subset[:30], IMG_DIR)
    sweep = scaling_sweep(hasher, scaling_subset, IMG_DIR, sizes=N_SCALING)

    per_image = sum(times) / len(times)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_latency_distribution(times, ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_latency_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_time_breakdown(stats, ax, top=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_time_breakdown.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    plot_io_split(split, ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_io_split.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_scaling(sweep, ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_scaling.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_extrapolation(per_image, full_n=55972, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_extrapolation.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_images": len(subset),
        "mean_s": per_image,
        "median_s": sorted(times)[len(times) // 2],
        "p95_s": sorted(times)[int(len(times) * 0.95)],
        "io_split": split,
        "scaling": sweep,
        "figures": sorted(str(p.name) for p in FIG_DIR.glob("*.png")),
    }
    (ROOT / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
