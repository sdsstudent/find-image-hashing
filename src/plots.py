"""Matplotlib plotting helpers for the profiling report."""

from __future__ import annotations

import numpy as np


def plot_latency_distribution(times, ax, title="Per-image hashing latency"):
    """Histogram of wall-clock latencies with mean/median/p95 markers."""
    times = np.asarray(times, dtype=float)
    ax.hist(times, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
    mean = times.mean()
    median = np.median(times)
    p95 = np.percentile(times, 95)
    ax.axvline(mean, color="red", linestyle="--", label=f"mean {mean:.3f}s")
    ax.axvline(median, color="orange", linestyle="--", label=f"median {median:.3f}s")
    ax.axvline(p95, color="purple", linestyle=":", label=f"p95 {p95:.3f}s")
    ax.set_xlabel("seconds")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()


def plot_time_breakdown(stats, ax, top=8, title="cProfile: top functions by tottime"):
    """Horizontal bar chart of cProfile top-N functions by tottime."""
    rows = []
    for (file, line, func), (cc, nc, tt, ct, _) in stats.stats.items():
        rows.append((f"{func}\n({file.split('/')[-1]}:{line})", tt, ct, nc))
    rows.sort(key=lambda r: r[1], reverse=True)
    rows = rows[:top]

    labels = [r[0] for r in rows][::-1]
    tottimes = [r[1] for r in rows][::-1]

    ax.barh(labels, tottimes, color="coral")
    ax.set_xlabel("tottime (s)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)


def plot_scaling(sweep_rows, ax, title="Runtime scaling with N"):
    """Log-log plot of total runtime vs N, with reference O(N) line."""
    ns = np.array([r["n"] for r in sweep_rows])
    totals = np.array([r["total_s"] for r in sweep_rows])

    ax.loglog(ns, totals, "o-", label="measured", color="steelblue", linewidth=2)

    if len(ns) >= 2:
        slope_ref = totals[0] / ns[0]
        ax.loglog(ns, slope_ref * ns, "--", color="grey", label="O(N) reference", alpha=0.7)

    ax.set_xlabel("N (images)")
    ax.set_ylabel("total runtime (s)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)


def plot_io_split(split: dict, ax, title="Wall-clock: I/O vs compute"):
    """Stacked bar showing I/O+decode vs pure compute share."""
    io = split["mean_io_plus_decode"] * 1000
    compute = split["mean_compute"] * 1000
    ax.barh(["mean per image"], [io], color="salmon", label=f"I/O + decode ({io:.0f}ms)")
    ax.barh(["mean per image"], [compute], left=[io], color="steelblue", label=f"compute ({compute:.0f}ms)")
    ax.set_xlabel("milliseconds")
    ax.set_title(title)
    ax.legend()


def plot_extrapolation(per_image_s: float, full_n: int, ax,
                        cores: list[int] = (1, 2, 4, 8),
                        title="Extrapolation to full dataset"):
    """Projected full-dataset runtime at different parallelism levels (perfect scaling assumed)."""
    hours = [per_image_s * full_n / c / 3600 for c in cores]
    labels = [f"{c} core{'s' if c > 1 else ''}" for c in cores]
    ax.bar(labels, hours, color="teal")
    for i, h in enumerate(hours):
        ax.text(i, h, f"{h:.1f}h", ha="center", va="bottom")
    ax.set_ylabel("projected hours")
    ax.set_title(f"{title} (N = {full_n:,})")
    ax.grid(axis="y", alpha=0.3)
