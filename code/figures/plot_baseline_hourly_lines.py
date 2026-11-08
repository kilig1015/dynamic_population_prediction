#!/usr/bin/env python
"""
Draw a four-panel line chart for hourly baseline metrics.

Input:
- output_results/tables/baselines_hourly_metrics_20170925_filtered_full.csv

Outputs:
- output_results/figures/hourly_prediction_performance_lines.tif
- output_results/figures/hourly_prediction_performance_lines.pdf
- output_results/figures/hourly_prediction_performance_lines.svg
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "output_results").exists() and (candidate / "code").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
DEFAULT_CSV = REPO_ROOT / "output_results" / "tables" / "baselines_hourly_metrics_20170925_filtered_full.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "output_results" / "figures"

BASELINE_ORDER = ["HA", "LSTM", "CNN", "CNN-LSTM", "ConvLSTM", "STAEformer", "PredRNN", "Pop-HSTM"]
METRICS = [("r2", "R$^2$"), ("tem", "TEM"), ("mae", "MAE"), ("mape", "MAPE")]
PANEL_LABELS = {"r2": "(a)", "tem": "(b)", "mae": "(c)", "mape": "(d)"}


BASELINE_COLORS: Dict[str, str] = {
    "HA": "#8D6E63",
    "LSTM": "#7E57C2",
    "CNN": "#2F7EBB",
    "CNN-LSTM": "#2FA84F",
    "ConvLSTM": "#F28E2B",
    "STAEformer": "#18AFC7",
    "PredRNN": "#B8B72A",
    "Pop-HSTM": "#D62728",
}

BASELINE_MARKERS: Dict[str, str] = {
    "HA": "o",
    "LSTM": "s",
    "CNN": "^",
    "CNN-LSTM": "D",
    "ConvLSTM": "v",
    "STAEformer": "P",
    "PredRNN": "X",
    "Pop-HSTM": "*",
}

AXIS_LABELS: Dict[str, str] = {
    "r2": "R$^2$",
    "tem": "TEM",
    "mae": "MAE",
    "mape": "MAPE (%)",
}

HOUR_BANDS = [
    (0, 5, "#F4F7FB"),
    (6, 11, "#F6FAF1"),
    (12, 17, "#FFF8EE"),
    (18, 23, "#FBF1F1"),
]


def resolve_path(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def ensure_hourly_csv(repo: Path, csv_path: Path, date: str) -> Path:
    if csv_path.exists():
        return csv_path
    fallback = repo / "output_results" / "tables" / f"baselines_hourly_metrics_{date}_filtered_full.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Hourly metrics CSV not found: {csv_path}")


def load_hourly_csv(csv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"baseline", "hour", "r2", "TEM", "mae", "mape"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns in {csv_path}: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "baseline": row["baseline"],
                    "hour": int(row["hour"]),
                    "r2": float(row["r2"]),
                    "tem": float(row["TEM"]),
                    "mae": float(row["mae"]),
                    "mape": float(row["mape"]),
                }
            )
    return rows


def setup_style(font_size: float) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "font.size": font_size,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw four-panel hourly line charts for baseline metrics."
    )
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument("--date", type=str, default="20170925", help="Date string in YYYYMMDD format.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Hourly metrics CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output figure directory.")
    parser.add_argument(
        "--output-stem",
        type=str,
        default="hourly_prediction_performance_lines",
        help="Output file stem without extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["tif", "pdf", "svg"],
        help="Output formats, e.g. tif pdf svg.",
    )
    parser.add_argument("--figsize", nargs=2, type=float, default=[64.0, 82.0], metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font-size", type=float, default=104.0)
    parser.add_argument("--legend-cols", type=int, default=4)
    parser.add_argument("--show", action="store_true", help="Show figure interactively.")
    return parser.parse_args()


def build_metric_series(
    rows: Sequence[Dict[str, object]],
    baselines: Sequence[str],
    hours: Sequence[int],
) -> Dict[str, Dict[str, np.ndarray]]:
    lookup = {
        (str(row["baseline"]), int(row["hour"])): row
        for row in rows
    }
    series: Dict[str, Dict[str, np.ndarray]] = {}
    for metric_key, _ in METRICS:
        metric_map: Dict[str, np.ndarray] = {}
        for baseline in baselines:
            values: List[float] = []
            for hour in hours:
                row = lookup.get((baseline, hour))
                values.append(float(row[metric_key]) if row is not None else float("nan"))
            metric_map[baseline] = np.asarray(values, dtype=float)
        series[metric_key] = metric_map
    return series


def add_hour_bands(ax: plt.Axes) -> None:
    for start, end, color in HOUR_BANDS:
        ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.35, zorder=0)


def set_metric_ylim(ax: plt.Axes, metric_key: str, values: Sequence[np.ndarray]) -> None:
    combined = np.concatenate([v[np.isfinite(v)] for v in values if np.isfinite(v).any()])
    if combined.size == 0:
        return
    y_min = float(np.min(combined))
    y_max = float(np.max(combined))
    pad = max((y_max - y_min) * 0.12, 1e-6)
    if metric_key == "r2":
        ax.set_ylim(max(0.0, y_min - pad), min(1.02, y_max + pad))
    else:
        ax.set_ylim(max(0.0, y_min - pad), y_max + pad)


def draw_line_panels(
    rows: Sequence[Dict[str, object]],
    figsize: Sequence[float],
    legend_cols: int,
) -> plt.Figure:
    baselines = [name for name in BASELINE_ORDER if any(str(row["baseline"]) == name for row in rows)]
    hours = sorted({int(row["hour"]) for row in rows})
    series = build_metric_series(rows, baselines, hours)

    fig, axes = plt.subplots(2, 2, figsize=tuple(figsize), constrained_layout=False)
    fig.patch.set_facecolor("white")
    axes_flat = axes.ravel()
    legend_handles = []
    legend_labels = []
    xticks = [0, 3, 6, 9, 12, 15, 18, 21, 23]

    for idx, (metric_key, _) in enumerate(METRICS):
        ax = axes_flat[idx]
        ax.set_facecolor("white")
        add_hour_bands(ax)
        for baseline in baselines:
            y = series[metric_key][baseline]
            is_main = baseline == "Pop-HSTM"
            line, = ax.plot(
                hours,
                y,
                color=BASELINE_COLORS.get(baseline, "#4F4F4F"),
                linewidth=9.0 if is_main else 5.2,
                marker=BASELINE_MARKERS.get(baseline, "o"),
                markersize=23.0 if is_main else 15.0,
                markerfacecolor="white" if is_main else BASELINE_COLORS.get(baseline, "#4F4F4F"),
                markeredgewidth=3.8 if is_main else 1.9,
                alpha=1.0 if is_main else 0.88,
                markevery=2,
                zorder=8 if is_main else 4,
                label=baseline,
            )
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(baseline)

        ax.set_title(
            f"{PANEL_LABELS.get(metric_key, '')} {AXIS_LABELS[metric_key]}".strip(),
            loc="left",
            fontsize=126,
            fontweight="bold",
            pad=48,
        )
        ax.set_xlabel("Hour", fontsize=128, fontweight="bold", labelpad=54)
        ax.set_ylabel(AXIS_LABELS[metric_key], fontsize=128, fontweight="bold", labelpad=74)
        ax.set_xlim(min(hours) - 0.5, max(hours) + 0.5)
        ax.set_xticks(xticks)
        set_metric_ylim(ax, metric_key, list(series[metric_key].values()))
        ax.grid(True, axis="y", color="#CDD3DC", linewidth=1.6, linestyle="-", alpha=0.72)
        ax.grid(True, axis="x", color="#E4E8EE", linewidth=1.1, linestyle="--", alpha=0.45)
        ax.tick_params(axis="both", labelsize=104, width=4.8, length=30.0, pad=26, color="#333333")
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#3A3A3A")
        ax.spines["bottom"].set_color("#3A3A3A")
        ax.spines["left"].set_linewidth(2.8)
        ax.spines["bottom"].set_linewidth(2.8)

    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=legend_cols,
        frameon=True,
        fancybox=False,
        edgecolor="#D5D9E0",
        facecolor="white",
        fontsize=112,
        handlelength=2.4,
        columnspacing=1.20,
        handletextpad=0.62,
        borderpad=1.05,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.125, right=0.985, top=0.945, bottom=0.205, wspace=0.34, hspace=0.48)
    return fig


def save_line_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tight_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03, "facecolor": "white"}
    try:
        if path.suffix.lower() in {".tif", ".tiff"}:
            fig.savefig(
                path,
                dpi=dpi,
                format="tiff",
                pil_kwargs={"compression": "tiff_lzw"},
                **tight_kwargs,
            )
        elif path.suffix.lower() in {".pdf", ".svg", ".eps"}:
            fig.savefig(path, **tight_kwargs)
        else:
            fig.savefig(path, dpi=dpi, **tight_kwargs)
        print(f"[info] Saved figure: {path}")
    except Exception as exc:
        fallback = path.with_suffix(".svg")
        fig.savefig(fallback, **tight_kwargs)
        print(f"[warn] Figure export failed for {path}: {exc}")
        print(f"[info] Saved fallback SVG: {fallback}")


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    csv_path = ensure_hourly_csv(repo, resolve_path(repo, args.csv), args.date)
    out_dir = resolve_path(repo, args.out_dir)

    setup_style(args.font_size)
    rows = load_hourly_csv(csv_path)
    fig = draw_line_panels(rows, figsize=args.figsize, legend_cols=max(1, int(args.legend_cols)))
    for fmt in args.formats:
        suffix = fmt if fmt.startswith(".") else f".{fmt}"
        save_line_figure(fig, out_dir / f"{args.output_stem}{suffix}", dpi=int(args.dpi))
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
