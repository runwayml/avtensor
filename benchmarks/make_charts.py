"""Render the benchmark result charts embedded in the READMEs.

Reads a results JSON produced by benchmark.py and writes light/dark SVG pairs
to benchmarks/assets/. The READMEs embed both via <picture> tags so GitHub
picks the variant matching the viewer's theme.

Usage:
    python benchmarks/make_charts.py results.json [-o benchmarks/assets]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": {
        "fg": "#1f2328",
        "grid": "#d1d9e0",
        "avtensor": "#1f77b4",
        "torchcodec": "#e8710a",
    },
    "dark": {
        "fg": "#e6edf3",
        "grid": "#3d444d",
        "avtensor": "#58a6ff",
        "torchcodec": "#f0883e",
    },
}

# (results key, chart label) for the single-decode chart, in display order.
SINGLE_DECODE_ROWS = [
    ("full_1080p_h264_threads1", "full 1080p H.264, 1 thread"),
    ("full_1080p_h264_threadsauto", "full 1080p H.264, auto threads"),
    ("full_1080p_hevc_threadsauto", "full 1080p HEVC, auto threads"),
    ("full_720p_h264_threads1", "full 720p H.264, 1 thread"),
    ("resize256x144_1080p_threads1", "1080p → 256×144, 1 thread"),
    ("resize256x144_1080p_threadsauto", "1080p → 256×144, auto threads"),
    ("seek_5s_window_1080p", "5 s window @ 15 s"),
    ("video_plus_audio_1080p", "video + audio"),
]


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor("none")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(theme["grid"])
    ax.tick_params(colors=theme["fg"], labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(theme["fg"])


def save(fig, out_dir: Path, name: str, variant: str) -> None:
    path = out_dir / f"{name}_{variant}.svg"
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def single_decode_chart(results: dict, out_dir: Path, variant: str) -> None:
    theme = THEMES[variant]
    rows = [
        (label, results["single"][key])
        for key, label in SINGLE_DECODE_ROWS
        if key in results.get("single", {})
    ]
    labels = [label for label, _ in rows]
    av = [r["avtensor"]["wall_median_s"] for _, r in rows]
    tc = [r["torchcodec"]["wall_median_s"] for _, r in rows]

    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(rows) + 1.2))
    y = range(len(rows))
    h = 0.38
    bars_av = ax.barh(
        [i - h / 2 for i in y],
        av,
        h,
        label="avtensor",
        color=theme["avtensor"],
        zorder=3,
    )
    bars_tc = ax.barh(
        [i + h / 2 for i in y],
        tc,
        h,
        label="torchcodec",
        color=theme["torchcodec"],
        zorder=3,
    )
    for bars in (bars_av, bars_tc):
        ax.bar_label(bars, fmt="%.2f s", padding=4, fontsize=9, color=theme["fg"])

    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(av + tc) * 1.18)
    ax.set_xlabel(
        "wall time, median of 5 (lower is better)", color=theme["fg"], fontsize=10
    )
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=False, labelcolor=theme["fg"], fontsize=10)
    style_axes(ax, theme)
    save(fig, out_dir, "single_decode", variant)


def concurrency_chart(results: dict, out_dir: Path, variant: str) -> None:
    theme = THEMES[variant]
    conc = results["concurrency"]
    workers = sorted(int(w) for w in conc["avtensor"])
    av = [conc["avtensor"][str(w)]["frames_per_s"] for w in workers]
    tc = [conc["torchcodec"][str(w)]["frames_per_s"] for w in workers]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(workers, av, "o-", label="avtensor", color=theme["avtensor"], linewidth=2.2)
    ax.plot(
        workers, tc, "o-", label="torchcodec", color=theme["torchcodec"], linewidth=2.2
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers, [str(w) for w in workers])
    ax.set_xlabel(
        "worker threads (720p clips, 1 FFmpeg thread per decode)",
        color=theme["fg"],
        fontsize=10,
    )
    ax.set_ylabel(
        "aggregate frames/s (higher is better)", color=theme["fg"], fontsize=10
    )
    ax.yaxis.grid(True, color=theme["grid"], linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, labelcolor=theme["fg"], fontsize=10)
    style_axes(ax, theme)
    save(fig, out_dir, "concurrency", variant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results JSON from benchmark.py")
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=Path(__file__).parent / "assets"
    )
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in THEMES:
        if results.get("single"):
            single_decode_chart(results, args.out_dir, variant)
        if results.get("concurrency"):
            concurrency_chart(results, args.out_dir, variant)


if __name__ == "__main__":
    main()
