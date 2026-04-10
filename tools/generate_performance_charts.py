#!/usr/bin/env python3
"""
Generate publication-ready performance charts for the embedded NLP article.

Outputs:
- docs/assets/checkpoint_tradeoff.(svg|png)
- docs/assets/memory_budget.(svg|png)
- docs/assets/per_class_recall_committed.(svg|png)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


ASSETS_DIR = ROOT / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


BG = "#f6f3ee"
PANEL = "#fffdfa"
TEXT = "#1e252b"
MUTED = "#66717c"
GRID = "#d8ddd7"
TEAL = "#217c7e"
TEAL_DARK = "#12585a"
GOLD = "#d89a2b"
SLATE = "#7b8794"
CORAL = "#c65f46"
GREEN = "#2d9d78"


CHECKPOINTS = [
    {
        "label": "Committed 64x64",
        "status": "Earlier committed",
        "hidden": "64x64",
        "params": 529_227,
        "size_mib": 2.02,
        "score_pct": 91.82,
        "min_recall_pct": 85.42,
        "color": SLATE,
    },
    {
        "label": "Local 96x64",
        "status": "Local experimental",
        "hidden": "96x64",
        "params": 793_451,
        "size_mib": 3.03,
        "score_pct": 91.57,
        "min_recall_pct": 82.61,
        "color": GOLD,
    },
    {
        "label": "Local 48x40",
        "status": "Best compact",
        "hidden": "48x40",
        "params": 395_675,
        "size_mib": 1.51,
        "score_pct": 91.65,
        "min_recall_pct": 84.06,
        "color": TEAL,
    },
    {
        "label": "Committed 80x112",
        "status": "Latest committed",
        "hidden": "80x112",
        "params": 665_755,
        "size_mib": 2.54,
        "score_pct": 90.90,
        "min_recall_pct": 83.77,
        "color": CORAL,
    },
]


PER_CLASS_COMMITTED = {
    "MISC": 83.77,
    "TECH": 89.92,
    "INFRA": 90.50,
    "BANKING": 90.57,
    "GOSSIP": 90.83,
    "BUSINESS": 91.06,
    "HR_COMPLAINT": 91.18,
    "LOVE": 91.30,
    "HR_HIRING": 93.19,
    "CYBER": 93.43,
    "ACCOUNTING": 94.12,
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "font.size": 10.5,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ASSETS_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(ASSETS_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def chart_checkpoint_tradeoff() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.grid(True, axis="both", alpha=0.65)

    ax.text(
        0.0,
        1.07,
        "Accuracy vs flash footprint",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.0,
        1.015,
        "The compact 48x40 checkpoint is the cleanest embedded trade-off in the workspace.",
        transform=ax.transAxes,
        fontsize=10.5,
        color=MUTED,
    )

    for cp in CHECKPOINTS:
        size = 140 + (cp["params"] / 8_000)
        edge = TEAL_DARK if cp["label"] == "Local 48x40" else "white"
        lw = 1.8 if cp["label"] == "Local 48x40" else 1.2
        ax.scatter(
            cp["size_mib"],
            cp["score_pct"],
            s=size,
            color=cp["color"],
            edgecolor=edge,
            linewidth=lw,
            zorder=3,
        )

        dy = 0.22 if cp["label"] != "Committed 80x112" else -0.42
        ax.annotate(
            f"{cp['label']}\n{cp['score_pct']:.2f}% score | {cp['min_recall_pct']:.2f}% weakest class",
            (cp["size_mib"], cp["score_pct"]),
            xytext=(10, 10 if dy > 0 else -38),
            textcoords="offset points",
            fontsize=9.2,
            color=TEXT,
            ha="left",
            va="bottom" if dy > 0 else "top",
            bbox=dict(boxstyle="round,pad=0.35", fc=PANEL, ec=GRID, lw=0.8),
        )

    ax.set_xlabel("Float32 flash footprint (MiB)")
    ax.set_ylabel("Balanced accuracy / mean recall (%)")
    ax.set_xlim(1.2, 3.3)
    ax.set_ylim(90.5, 92.1)

    save(fig, "checkpoint_tradeoff")


def bar_with_budget(ax: plt.Axes, label: str, value: float, total: float, color: str, unit: str) -> None:
    ax.barh([label], [total], color="#ebe7df", height=0.55)
    ax.barh([label], [value], color=color, height=0.55)
    pct = (value / total) * 100.0
    ax.text(
        value + total * 0.012,
        label,
        f"{value:.2f} {unit}  ({pct:.1f}%)",
        va="center",
        ha="left",
        fontsize=9.6,
        color=TEXT,
    )


def chart_memory_budget() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.8))

    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL)
        ax.grid(False)

    ax1.text(
        0.0,
        1.08,
        "Deployment budget on Teensy 4.1",
        transform=ax1.transAxes,
        fontsize=18,
        fontweight="bold",
        color=TEXT,
    )
    ax1.text(
        0.0,
        1.02,
        "Flash is the main constraint. Runtime RAM is comparatively small.",
        transform=ax1.transAxes,
        fontsize=10.5,
        color=MUTED,
    )

    flash_budget_mib = 8.0
    bar_with_budget(ax1, "Compact FP32", 1.51, flash_budget_mib, TEAL, "MiB")
    bar_with_budget(ax1, "Committed FP32", 2.54, flash_budget_mib, CORAL, "MiB")
    bar_with_budget(ax1, "Compact INT8 (proj.)", 0.377, flash_budget_mib, GREEN, "MiB")
    ax1.set_title("Flash budget", loc="left", pad=12)
    ax1.set_xlim(0, flash_budget_mib * 1.16)
    ax1.set_xlabel("Flash usage out of 8 MiB")

    ram_budget_kib = 1024.0
    bar_with_budget(ax2, "Input buffer", 32.00, ram_budget_kib, GOLD, "KiB")
    bar_with_budget(ax2, "Dense activations", 32.39, ram_budget_kib, SLATE, "KiB")
    ax2.set_title("Runtime RAM budget", loc="left", pad=12)
    ax2.set_xlim(0, ram_budget_kib * 1.16)
    ax2.set_xlabel("RAM usage out of 1024 KiB")

    save(fig, "memory_budget")


def chart_per_class_recall() -> None:
    labels = list(PER_CLASS_COMMITTED.keys())
    values = np.array(list(PER_CLASS_COMMITTED.values()))
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = values[order]

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.grid(True, axis="x", alpha=0.65)

    ax.text(
        0.0,
        1.07,
        "Per-class recall of the latest committed checkpoint",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.0,
        1.015,
        "The weakest class is MISC, while the most structured classes stay above 90%.",
        transform=ax.transAxes,
        fontsize=10.5,
        color=MUTED,
    )

    colors = []
    for label in labels:
        if label == "MISC":
            colors.append(CORAL)
        elif label in {"ACCOUNTING", "CYBER", "HR_HIRING"}:
            colors.append(TEAL)
        else:
            colors.append(SLATE)

    ax.barh(labels, values, color=colors, height=0.62)
    avg = 90.90
    ax.axvline(avg, color=GOLD, linewidth=1.8, linestyle="--")
    ax.text(avg + 0.15, len(labels) - 0.45, f"mean {avg:.2f}%", color=GOLD, fontsize=9.5)

    for y, value in enumerate(values):
        ax.text(value + 0.15, y, f"{value:.2f}%", va="center", ha="left", fontsize=9.4, color=TEXT)

    ax.set_xlim(82.5, 95.2)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("")

    save(fig, "per_class_recall_committed")


def main() -> None:
    apply_style()
    chart_checkpoint_tradeoff()
    chart_memory_budget()
    chart_per_class_recall()
    print(f"Charts written to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
