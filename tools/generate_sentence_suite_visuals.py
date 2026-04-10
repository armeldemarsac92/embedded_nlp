#!/usr/bin/env python3
"""
Generate publication-ready visuals for the sentence-suite evaluation.

Outputs:
- docs/assets/sentence_suite_dashboard.(svg|png)
- docs/assets/sentence_suite_matrix.(svg|png)
- docs/assets/sentence_suite_phrases.(svg|png)
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


RESULTS_PATH = ROOT / "artifacts" / "evaluation" / "sentence_suite_quantization_results.json"
ASSETS_DIR = ROOT / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

BG = "#f6f3ee"
PANEL = "#fffdfa"
TEXT = "#1e252b"
MUTED = "#66717c"
GRID = "#d8ddd7"
TEAL = "#217c7e"
TEAL_DARK = "#12585a"
GREEN = "#2d9d78"
GREEN_SOFT = "#dcefe6"
CORAL = "#c65f46"
CORAL_SOFT = "#f6dfd7"
GOLD = "#d89a2b"
SLATE = "#7b8794"


def load_payload() -> dict:
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


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


def blend_hex(low: str, high: str, t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, t))

    def parse(hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    lo = parse(low)
    hi = parse(high)
    return tuple(((1.0 - t) * lo[i] + t * hi[i]) / 255.0 for i in range(3))


def metric_card(ax, x: float, y: float, w: float, h: float, title: str, value: str, subtitle: str, accent: str) -> None:
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        linewidth=1.0,
        edgecolor=GRID,
        facecolor=PANEL,
    )
    ax.add_patch(card)
    ax.add_patch(Rectangle((x, y + h - 0.018), w, 0.018, facecolor=accent, edgecolor="none"))
    ax.text(x + 0.03, y + h * 0.68, title, fontsize=9.6, color=MUTED, va="center")
    ax.text(x + 0.03, y + h * 0.33, value, fontsize=18, fontweight="bold", color=TEXT, va="center")


def chart_dashboard(payload: dict) -> None:
    fig = plt.figure(figsize=(13.4, 9.2), facecolor=BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.82, 1.02, 1.26], hspace=0.42, wspace=0.18)

    ax_cards = fig.add_subplot(gs[0, :])
    ax_mem = fig.add_subplot(gs[1, 0])
    ax_conf = fig.add_subplot(gs[1, 1])
    ax_cases = fig.add_subplot(gs[2, :])

    for ax in (ax_cards, ax_mem, ax_conf, ax_cases):
        ax.set_facecolor(PANEL)

    ax_cards.axis("off")
    ax_cards.text(0.0, 1.08, "Sentence-level performance of the embedded classifier", fontsize=18, fontweight="bold", color=TEXT, transform=ax_cards.transAxes)
    ax_cards.text(0.0, 0.98, "A clean sentence suite with deletion and transposition typos, compared before and after INT8 quantization.", fontsize=10.5, color=MUTED, transform=ax_cards.transAxes)

    summary = payload["summary"]
    mem = payload["memory"]
    cards = [
        ("FP32 Accuracy", f"{summary['float']['accuracy']:.0%}", "Expected label on all 22 phrases", TEAL),
        ("INT8 Accuracy", f"{summary['quantized']['accuracy']:.0%}", "Same sentence suite after quantization", GREEN),
        ("FP32 = INT8", f"{summary['agreement_rate']:.0%}", "Prediction agreement case by case", GOLD),
        ("Typo Robustness", f"{summary['float']['typo_accuracy']:.0%}", "Deletion/transposition typo set", SLATE),
        ("Memory Saved", f"{mem['reduction_percent']:.0f}%", "Parameter storage reduction", CORAL),
    ]

    card_positions = [
        (0.00, 0.43, 0.31, 0.35),
        (0.345, 0.43, 0.31, 0.35),
        (0.69, 0.43, 0.31, 0.35),
        (0.175, 0.04, 0.31, 0.35),
        (0.515, 0.04, 0.31, 0.35),
    ]
    for (title, value, subtitle, accent), (x, y, w, h) in zip(cards, card_positions):
        metric_card(ax_cards, x, y, w, h, title, value, subtitle, accent)

    ax_mem.grid(False)
    ax_mem.text(0.0, 1.13, "Parameter footprint", transform=ax_mem.transAxes, fontsize=15.5, fontweight="bold", color=TEXT)
    ax_mem.text(0.0, 1.06, "INT8 cuts the stored weights to about one quarter of the float32 size.", transform=ax_mem.transAxes, fontsize=9.8, color=MUTED)
    labels = ["Float32 model", "INT8 quantized"]
    values = [mem["float32_mib"], mem["int8_mib"]]
    colors = [TEAL, GREEN]
    y_positions = [0.55, 1.35]
    ax_mem.barh(y_positions, values, color=colors, height=0.42)
    ax_mem.set_yticks(y_positions, labels)
    ax_mem.set_ylim(1.8, 0.0)
    ax_mem.set_xlim(0, max(values) * 1.25)
    ax_mem.set_xlabel("MiB")
    for ypos, val in zip(y_positions, values):
        ax_mem.text(val + 0.03, ypos, f"{val:.3f} MiB", va="center", ha="left", fontsize=10.2, color=TEXT)

    ax_conf.grid(True, axis="y", alpha=0.45)
    ax_conf.text(0.0, 1.13, "Average confidence by variant", transform=ax_conf.transAxes, fontsize=15.5, fontweight="bold", color=TEXT)
    ax_conf.text(0.0, 1.06, "Quantization barely moves confidence, even on typo variants.", transform=ax_conf.transAxes, fontsize=9.8, color=MUTED)
    variants = ["Clean", "Typo"]
    x = [0, 1]
    float_vals = [
        summary["float"]["clean_avg_confidence"] * 100.0,
        summary["float"]["typo_avg_confidence"] * 100.0,
    ]
    quant_vals = [
        summary["quantized"]["clean_avg_confidence"] * 100.0,
        summary["quantized"]["typo_avg_confidence"] * 100.0,
    ]
    ax_conf.bar([v - 0.16 for v in x], float_vals, width=0.28, color=TEAL, label="Float32")
    ax_conf.bar([v + 0.16 for v in x], quant_vals, width=0.28, color=GREEN, label="INT8")
    ax_conf.set_xticks(x, variants)
    ax_conf.set_ylim(91.8, 101.0)
    ax_conf.set_ylabel("Average top-class confidence (%)")
    ax_conf.legend(frameon=False, loc="lower right")
    for xpos, val in zip([v - 0.16 for v in x], float_vals):
        ax_conf.text(xpos, val + 0.15, f"{val:.2f}", ha="center", va="bottom", fontsize=9.5)
    for xpos, val in zip([v + 0.16 for v in x], quant_vals):
        ax_conf.text(xpos, val + 0.15, f"{val:.2f}", ha="center", va="bottom", fontsize=9.5)

    ax_cases.axis("off")
    ax_cases.text(0.0, 0.99, "Lowest-confidence cases", fontsize=15.5, fontweight="bold", color=TEXT, transform=ax_cases.transAxes)
    ax_cases.text(0.0, 0.92, "These are the most informative examples because the rest of the suite is essentially saturated near 100%.", fontsize=9.8, color=MUTED, transform=ax_cases.transAxes)

    lowest = sorted(payload["results"], key=lambda row: row["float"]["confidence"])[:4]
    col_width = 0.235
    for idx, row in enumerate(lowest):
        x0 = idx * (col_width + 0.02)
        y0 = 0.02
        card = FancyBboxPatch(
            (x0, y0),
            col_width,
            0.76,
            boxstyle="round,pad=0.012,rounding_size=0.024",
            linewidth=1.0,
            edgecolor=GRID,
            facecolor=PANEL,
        )
        ax_cases.add_patch(card)
        variant_color = GOLD if row["variant"] == "typo" else SLATE
        badge = FancyBboxPatch(
            (x0 + 0.02, y0 + 0.64),
            0.12,
            0.09,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0,
            facecolor=variant_color,
        )
        ax_cases.add_patch(badge)
        ax_cases.text(x0 + 0.08, y0 + 0.685, row["variant"].upper(), ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        ax_cases.text(x0 + 0.17, y0 + 0.69, row["expected"], ha="left", va="center", fontsize=10.2, color=TEXT, fontweight="bold")
        sentence = textwrap.fill(row["sentence"], width=32)
        ax_cases.text(x0 + 0.02, y0 + 0.56, sentence, ha="left", va="top", fontsize=9.6, color=TEXT)
        ax_cases.text(x0 + 0.02, y0 + 0.17, f"FP32 {row['float']['confidence'] * 100:.2f}%", fontsize=10.2, color=TEAL, fontweight="bold")
        ax_cases.text(x0 + 0.02, y0 + 0.08, f"INT8 {row['quantized']['confidence'] * 100:.2f}%", fontsize=10.2, color=GREEN, fontweight="bold")
        ax_cases.text(x0 + 0.16, y0 + 0.08, f"Δ {abs(row['float']['confidence'] - row['quantized']['confidence']):.4f}", fontsize=8.6, color=MUTED)

    save(fig, "sentence_suite_dashboard")


def chart_matrix(payload: dict) -> None:
    classes = payload["classes"]
    rows_by_key = {(row["expected"], row["variant"]): row for row in payload["results"]}

    fig, ax = plt.subplots(figsize=(12.8, 8.8))
    ax.set_facecolor(PANEL)
    ax.axis("off")

    ax.text(0.0, 1.05, "Clean vs typo sentence matrix", fontsize=18, fontweight="bold", color=TEXT, transform=ax.transAxes)
    ax.text(0.0, 0.995, "Each cell shows the top prediction confidence for the expected class. INT8 matched the float model on every case.", fontsize=10.4, color=MUTED, transform=ax.transAxes)

    left = 0.18
    top = 0.9
    row_h = 0.065
    col_w = 0.32
    gap = 0.05

    ax.text(left + col_w / 2, top + 0.035, "Clean", ha="center", va="center", fontsize=12, fontweight="bold", color=TEXT)
    ax.text(left + col_w + gap + col_w / 2, top + 0.035, "Typo", ha="center", va="center", fontsize=12, fontweight="bold", color=TEXT)

    for row_idx, expected in enumerate(classes):
        y = top - (row_idx + 1) * row_h
        display_label = expected.replace("_", " ")
        ax.text(0.02, y + row_h / 2, display_label, ha="left", va="center", fontsize=10.2, fontweight="bold", color=TEXT)

        for col_idx, variant in enumerate(("clean", "typo")):
            row = rows_by_key[(expected, variant)]
            x = left + col_idx * (col_w + gap)
            correct = row["float"]["predicted"] == expected and row["quantized"]["predicted"] == expected
            confidence = min(row["float"]["confidence"], row["quantized"]["confidence"])
            if correct:
                t = (confidence - 0.90) / 0.10
                face = blend_hex(GREEN_SOFT, GREEN, t)
                edge = GREEN
            else:
                t = min(1.0, max(0.0, confidence))
                face = blend_hex(CORAL_SOFT, CORAL, t)
                edge = CORAL

            rect = FancyBboxPatch(
                (x, y + 0.008),
                col_w,
                row_h - 0.016,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                linewidth=1.0,
                edgecolor=edge,
                facecolor=face,
            )
            ax.add_patch(rect)

            status = row["float"]["predicted"] if not correct else "OK"
            ax.text(x + 0.022, y + row_h * 0.50, f"{row['float']['confidence'] * 100:.2f}%", ha="left", va="center", fontsize=11, fontweight="bold", color=TEXT)
            ax.text(x + 0.145, y + row_h * 0.58, f"FP32 {row['float']['confidence'] * 100:.2f}", ha="left", va="center", fontsize=8.3, color=TEXT)
            ax.text(x + 0.145, y + row_h * 0.34, f"INT8  {row['quantized']['confidence'] * 100:.2f}", ha="left", va="center", fontsize=8.3, color=TEXT)
            ax.text(x + col_w - 0.02, y + row_h * 0.46, status.upper(), ha="right", va="center", fontsize=8.4, color=TEXT if correct else CORAL, fontweight="bold")

    save(fig, "sentence_suite_matrix")


def chart_phrases(payload: dict) -> None:
    classes = payload["classes"]
    rows_by_key = {(row["expected"], row["variant"]): row for row in payload["results"]}

    fig, ax = plt.subplots(figsize=(14.2, 12.4))
    ax.set_facecolor(PANEL)
    ax.axis("off")

    ax.text(
        0.0,
        1.03,
        "Sentence suite used for the evaluation",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.985,
        "Each class is shown with one clean sentence and one typo variant using missing letters or shifted letters only.",
        fontsize=10.2,
        color=MUTED,
        transform=ax.transAxes,
    )

    left_label = 0.02
    clean_x = 0.17
    typo_x = 0.57
    box_w = 0.37
    top = 0.935
    row_h = 0.078
    box_h = 0.056

    ax.text(clean_x + box_w / 2, top + 0.01, "Clean sentence", ha="center", va="center", fontsize=11.5, fontweight="bold", color=TEXT)
    ax.text(typo_x + box_w / 2, top + 0.01, "Typo variant", ha="center", va="center", fontsize=11.5, fontweight="bold", color=TEXT)

    for row_idx, expected in enumerate(classes):
        y = top - (row_idx + 1) * row_h
        label = expected.replace("_", " ")
        ax.text(
            left_label,
            y + box_h / 2,
            label,
            ha="left",
            va="center",
            fontsize=10.3,
            fontweight="bold",
            color=TEXT,
        )

        for variant, x, accent, badge_text in (
            ("clean", clean_x, TEAL, "CLEAN"),
            ("typo", typo_x, GOLD, "TYPO"),
        ):
            row = rows_by_key[(expected, variant)]
            sentence = textwrap.fill(row["sentence"], width=38)

            box = FancyBboxPatch(
                (x, y),
                box_w,
                box_h,
                boxstyle="round,pad=0.008,rounding_size=0.016",
                linewidth=1.0,
                edgecolor=GRID,
                facecolor=PANEL,
            )
            ax.add_patch(box)
            ax.add_patch(Rectangle((x, y + box_h - 0.009), box_w, 0.009, facecolor=accent, edgecolor="none"))

            badge = FancyBboxPatch(
                (x + 0.012, y + box_h - 0.032),
                0.075,
                0.022,
                boxstyle="round,pad=0.005,rounding_size=0.01",
                linewidth=0,
                facecolor=accent,
            )
            ax.add_patch(badge)
            ax.text(
                x + 0.0495,
                y + box_h - 0.021,
                badge_text,
                ha="center",
                va="center",
                fontsize=7.9,
                color="white",
                fontweight="bold",
            )

            ax.text(
                x + 0.095,
                y + box_h - 0.02,
                sentence,
                ha="left",
                va="top",
                fontsize=8.8,
                color=TEXT,
                linespacing=1.12,
            )

    save(fig, "sentence_suite_phrases")


def main() -> int:
    payload = load_payload()
    apply_style()
    chart_dashboard(payload)
    chart_matrix(payload)
    chart_phrases(payload)
    print(f"Saved visuals to {ASSETS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
