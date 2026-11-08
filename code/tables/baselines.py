#!/usr/bin/env python
"""
Baseline hourly metric evaluation and per-baseline plotting.

This script evaluates the 24 hourly predictions on 2017-09-25 for the
following baseline folders under output_results/dynamic_population_distribution:

- baselines/HA
- baselines/CNN
- baselines/CNN-LSTM
- baselines/ConvLSTM
- baselines/LSTM
- baselines/PredRNN
- baselines/STAEformer
- Pop-HSTM

Metrics follow the raw-value protocol used in metrics_eval.py, except that
standard RMSE is replaced by TEM (i.e., ln(1+MSE)):

- R^2
- TEM = ln(1 + mean((pred - truth)^2))
- MAE
- MAPE

Outputs:
- Hourly metrics CSV (one row per baseline-hour)
- Summary metrics CSV (one row per baseline for the whole day)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except Exception:
    matplotlib = None
    mcolors = None
    plt = None
    FuncFormatter = None

import numpy as np
from PIL import Image


DATE_DEFAULT = "20170925"
EPSILON = 1e-5
LN_1_PLUS_MSE_NAME = "TEM"

BASELINE_PATTERNS: List[Tuple[str, Sequence[str]]] = [
    ("HA", ["output_results/dynamic_population_distribution/baselines/HA/Pred_{date}"]),
    ("LSTM", ["output_results/dynamic_population_distribution/baselines/LSTM/**/Pred_{date}"]),
    ("CNN", ["output_results/dynamic_population_distribution/baselines/CNN/Pred_{date}"]),
    ("CNN-LSTM", ["output_results/dynamic_population_distribution/baselines/CNN-LSTM/**/Pred_{date}"]),
    ("ConvLSTM", ["output_results/dynamic_population_distribution/baselines/ConvLSTM/**/Pred_{date}"]),
    ("STAEformer", ["output_results/dynamic_population_distribution/baselines/STAEformer/**/Pred_{date}"]),
    ("PredRNN", ["output_results/dynamic_population_distribution/baselines/PredRNN/**/Pred_{date}"]),
    ("Pop-HSTM", ["output_results/dynamic_population_distribution/Pop-HSTM/Pred_{date}"]),
]

METRICS: List[Tuple[str, str]] = [
    ("r2", "R\u00b2"),
    (LN_1_PLUS_MSE_NAME, LN_1_PLUS_MSE_NAME),
    ("mae", "MAE"),
    ("mape", "MAPE"),
]

METRIC_COLORS = {
    "r2": "#2F6EA4",
    LN_1_PLUS_MSE_NAME: "#3E9A42",
    "mae": "#D8842A",
    "mape": "#C54242",
}

BASELINE_COLORS = {
    "HA": "#C44E52",
    "CNN": "#4C72B0",
    "CNN-LSTM": "#55A868",
    "ConvLSTM": "#E39C35",
    "LSTM": "#8172B2",
    "PredRNN": "#2A9D8F",
    "STAEformer": "#9C6644",
    "Pop-HSTM": "#B56576",
}

HOUR_BANDS = [
    ("Night", 0, 5, "#EEF2F7"),
    ("Morning", 6, 11, "#F4F7EC"),
    ("Afternoon", 12, 17, "#FFF4E6"),
    ("Evening", 18, 23, "#F8ECEC"),
]

PRED_NAME_RE = re.compile(r"^PredPop_(\d{8})_(\d{2})_real\.tif$", re.IGNORECASE)


PROJECT_ROOT = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "output_results").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root(PROJECT_ROOT)
DEFAULT_TRUTH_DIR = REPO_ROOT / "data" / "mobile_phone_data"
DEFAULT_OUT_DIR = REPO_ROOT / "output_results" / "tables"
DEFAULT_HOURLY_CSV = DEFAULT_OUT_DIR / "baselines_hourly_metrics_20170925_filtered_full.csv"
DEFAULT_SUMMARY_CSV = DEFAULT_OUT_DIR / "baselines_summary_metrics_20170925_filtered_full.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hourly baseline metrics and save hourly/summary CSV files."
    )
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="Repository root.")
    parser.add_argument("--date", type=str, default=DATE_DEFAULT, help="Evaluation date in YYYYMMDD format.")
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=DEFAULT_TRUTH_DIR,
        help="Ground truth folder containing pointMMDDHH_500.tif files.",
    )
    parser.add_argument(
        "--truth-tpl",
        type=str,
        default="point{date_key}{hour:02d}_500.tif",
        help="Ground truth filename template.",
    )
    parser.add_argument(
        "--pred-tpl",
        type=str,
        default="PredPop_{date}_{hour:02d}_real.tif",
        help="Prediction filename template.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        nargs="+",
        default=list(range(24)),
        help="Hours to evaluate. Default: 0-23.",
    )
    parser.add_argument(
        "--min-truth",
        type=float,
        default=1.0,
        help="Minimum ground-truth threshold. Default: 1.0 to avoid MAPE explosion on zero-value pixels.",
    )
    parser.add_argument(
        "--clip-quantile",
        nargs=2,
        type=float,
        default=[0.01, 0.99],
        metavar=("LOW", "HIGH"),
        help="Truth quantile clipping range. Default: 0.01 0.99.",
    )
    parser.add_argument(
        "--hourly-csv",
        type=Path,
        default=DEFAULT_HOURLY_CSV,
        help="Hourly metrics CSV output path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Daily summary metrics CSV output path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory used to save one 2x2 figure per baseline.",
    )
    parser.add_argument("--figsize", nargs=2, type=float, default=[16.5, 10.5], metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=1200)
    parser.add_argument("--font-size", type=float, default=16.0)
    parser.add_argument("--tick-step", type=int, default=2)
    parser.add_argument("--box-width", type=float, default=0.78)
    parser.add_argument("--point-size", type=float, default=34.0)
    parser.add_argument("--line-width", type=float, default=1.6)
    parser.add_argument("--show", action="store_true", help="Show figure interactively.")
    return parser.parse_args()


def resolve_path(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo / path)


def is_vector_output(path: Path) -> bool:
    return path.suffix.lower() in {".pdf", ".svg", ".eps"}


def load_raster(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path), dtype=np.float64).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D raster, got {arr.shape} from {path}")
    return arr


def apply_filters(
    pred: np.ndarray,
    truth: np.ndarray,
    min_truth: Optional[float] = None,
    clip_quantile: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(pred) & np.isfinite(truth)
    pred = pred[mask]
    truth = truth[mask]

    if min_truth is not None:
        mask = truth >= min_truth
        pred = pred[mask]
        truth = truth[mask]

    if clip_quantile is not None and truth.size > 0:
        ql, qh = np.quantile(truth, [clip_quantile[0], clip_quantile[1]])
        mask = (truth >= ql) & (truth <= qh)
        pred = pred[mask]
        truth = truth[mask]

    return pred, truth


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    diff = pred - truth
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff) / (truth + EPSILON)) * 100.0)
    truth_mean = float(np.mean(truth))
    ss_tot = float(np.sum((truth - truth_mean) ** 2))
    ss_res = float(np.sum((truth - pred) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return {
        "r2": r2,
        LN_1_PLUS_MSE_NAME: float(math.log1p(mse)),
        "mae": mae,
        "mape": mape,
    }


def format_date_key(date: str) -> str:
    return date[-4:]


def discover_prediction_dirs(repo: Path, date: str) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for baseline, patterns in BASELINE_PATTERNS:
        matches: List[Path] = []
        for pattern in patterns:
            matches.extend(repo.glob(pattern.format(date=date)))
        matches = sorted({p.resolve() for p in matches if p.is_dir()})
        if not matches:
            print(f"[warn] Missing prediction directory for {baseline}")
            continue
        if len(matches) > 1:
            print(f"[warn] Multiple directories found for {baseline}; using the first one:")
            for item in matches:
                print(f"       {item}")
        found[baseline] = matches[0]
    return found


def collect_hourly_rows(
    repo: Path,
    truth_dir: Path,
    date: str,
    hours: Sequence[int],
    min_truth: Optional[float],
    clip_quantile: Optional[Tuple[float, float]],
    truth_tpl: str,
    pred_tpl: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pred_dirs = discover_prediction_dirs(repo, date)
    date_key = format_date_key(date)

    for baseline, pred_dir in pred_dirs.items():
        matched = 0
        for hour in hours:
            pred_path = pred_dir / pred_tpl.format(date=date, hour=hour)
            truth_path = truth_dir / truth_tpl.format(date=date, date_key=date_key, hour=hour)
            if not pred_path.exists():
                print(f"[warn] Missing prediction: {pred_path}")
                continue
            if not truth_path.exists():
                print(f"[warn] Missing ground truth: {truth_path}")
                continue

            pred = load_raster(pred_path).ravel()
            truth = load_raster(truth_path).ravel()
            if pred.shape != truth.shape:
                print(f"[warn] Shape mismatch for {baseline} hour {hour:02d}: pred={pred.shape}, truth={truth.shape}")
                continue

            pred, truth = apply_filters(pred, truth, min_truth=min_truth, clip_quantile=clip_quantile)
            if pred.size == 0:
                print(f"[warn] No valid pixels after filtering for {baseline} hour {hour:02d}")
                continue

            metric_vals = compute_metrics(pred, truth)
            rows.append(
                {
                    "baseline": baseline,
                    "date": date,
                    "hour": int(hour),
                    "n_pixels": int(pred.size),
                    **metric_vals,
                }
            )
            matched += 1
        print(f"[info] {baseline}: matched {matched} hourly rasters from {pred_dir}")
    return rows


def write_hourly_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["baseline", "date", "hour", "n_pixels", "r2", LN_1_PLUS_MSE_NAME, "mae", "mape"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (str(x["baseline"]), int(x["hour"]))):
            writer.writerow({
                "baseline": row.get("baseline"),
                "date": row.get("date"),
                "hour": row.get("hour"),
                "n_pixels": row.get("n_pixels"),
                "r2": row.get("r2"),
                LN_1_PLUS_MSE_NAME: row.get(LN_1_PLUS_MSE_NAME),
                "mae": row.get("mae"),
                "mape": row.get("mape"),
            })
    print(f"[info] Saved hourly metrics CSV: {out_csv}")


def collect_daily_summary_rows(
    repo: Path,
    truth_dir: Path,
    date: str,
    hours: Sequence[int],
    min_truth: Optional[float],
    clip_quantile: Optional[Tuple[float, float]],
    truth_tpl: str,
    pred_tpl: str,
) -> List[Dict[str, object]]:
    pred_dirs = discover_prediction_dirs(repo, date)
    date_key = format_date_key(date)
    out_rows: List[Dict[str, object]] = []

    for baseline, pred_dir in pred_dirs.items():
        pred_parts: List[np.ndarray] = []
        truth_parts: List[np.ndarray] = []
        for hour in hours:
            pred_path = pred_dir / pred_tpl.format(date=date, hour=hour)
            truth_path = truth_dir / truth_tpl.format(date=date, date_key=date_key, hour=hour)
            if not pred_path.exists() or not truth_path.exists():
                continue
            pred = load_raster(pred_path).ravel()
            truth = load_raster(truth_path).ravel()
            if pred.shape != truth.shape:
                continue
            pred_parts.append(pred)
            truth_parts.append(truth)

        if not pred_parts:
            continue

        pred_all = np.concatenate(pred_parts)
        truth_all = np.concatenate(truth_parts)
        pred_all, truth_all = apply_filters(pred_all, truth_all, min_truth=min_truth, clip_quantile=clip_quantile)
        if pred_all.size == 0:
            continue
        metric_vals = compute_metrics(pred_all, truth_all)
        out_rows.append(
            {
                "baseline": baseline,
                "date": date,
                "hours": len(pred_parts),
                "n_pixels": int(pred_all.size),
                **metric_vals,
            }
        )

    return out_rows


def write_summary_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["baseline", "date", "hours", "n_pixels", "r2", LN_1_PLUS_MSE_NAME, "mae", "mape"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        ordered = sorted(
            rows,
            key=lambda x: next((i for i, (name, _) in enumerate(BASELINE_PATTERNS) if name == x["baseline"]), 999),
        )
        for row in ordered:
            writer.writerow({
                "baseline": row.get("baseline"),
                "date": row.get("date"),
                "hours": row.get("hours"),
                "n_pixels": row.get("n_pixels"),
                "r2": row.get("r2"),
                LN_1_PLUS_MSE_NAME: row.get(LN_1_PLUS_MSE_NAME),
                "mae": row.get("mae"),
                "mape": row.get("mape"),
            })
    print(f"[info] Saved summary metrics CSV: {out_csv}")


def setup_style(font_size: float) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "font.size": font_size,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def lighten_color(color: str, amount: float) -> Tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    return tuple(float(v) for v in (rgb + (1.0 - rgb) * amount))


def add_hour_bands(ax: plt.Axes, hours: Sequence[int]) -> None:
    idx_map = {h: i for i, h in enumerate(hours)}
    for _, start, end, color in HOUR_BANDS:
        existing = [h for h in hours if start <= h <= end]
        if not existing:
            continue
        left = idx_map[min(existing)] - 0.5
        right = idx_map[max(existing)] + 0.5
        ax.axvspan(left, right, color=color, alpha=0.7, zorder=0)


def add_hour_band_dividers(ax: plt.Axes, hours: Sequence[int]) -> None:
    idx_map = {h: i for i, h in enumerate(hours)}
    active_ranges = []
    for _, start, end, _ in HOUR_BANDS:
        existing = [h for h in hours if start <= h <= end]
        if existing:
            active_ranges.append((idx_map[min(existing)], idx_map[max(existing)]))
    for (_, prev_right), (next_left, _) in zip(active_ranges, active_ranges[1:]):
        boundary = (prev_right + next_left) / 2.0
        ax.axvline(boundary, color="#b5b5b5", linewidth=0.8, linestyle=(0, (3, 3)), zorder=2, alpha=0.85)


def add_hour_band_labels(ax: plt.Axes, hours: Sequence[int], font_scale: float) -> None:
    idx_map = {h: i for i, h in enumerate(hours)}
    for label, start, end, _ in HOUR_BANDS:
        existing = [h for h in hours if start <= h <= end]
        if not existing:
            continue
        xpos = (idx_map[min(existing)] + idx_map[max(existing)]) / 2.0
        ax.text(
            xpos,
            1.008,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=11.0 * font_scale,
            fontweight="black",
            color="#4f4f4f",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.86, pad=1.35),
            zorder=5,
            clip_on=False,
        )


def build_metric_maps(rows: List[Dict[str, object]]) -> Dict[str, Dict[int, Dict[str, float]]]:
    data: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    for row in rows:
        baseline = str(row["baseline"])
        hour = int(row["hour"])
        data[baseline][hour] = {metric_key: float(row[metric_key]) for metric_key, _ in METRICS}
    return data


def plot_metric(
    ax: plt.Axes,
    baseline: str,
    hourly_map: Dict[int, Dict[str, float]],
    metric_key: str,
    metric_label: str,
    hours: Sequence[int],
    tick_step: int,
    box_width: float,
    point_size: float,
    line_width: float,
    font_scale: float,
) -> None:
    positions = np.arange(len(hours))
    color = BASELINE_COLORS.get(baseline, METRIC_COLORS[metric_key])
    y_vals = [float(hourly_map[h][metric_key]) if h in hourly_map else float("nan") for h in hours]

    ax.set_title(f"{metric_label} by Hour", fontsize=13.0 * font_scale, fontweight="bold", pad=15 * font_scale)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#d9d9d9", linewidth=0.85)
    ax.xaxis.grid(False)

    add_hour_bands(ax, hours)
    add_hour_band_dividers(ax, hours)

    box_data = [[y] if math.isfinite(y) else [float("nan")] for y in y_vals]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#222222", markersize=5.6 * font_scale),
        medianprops=dict(color="#2f2f2f", linewidth=1.2 * font_scale),
        boxprops=dict(linewidth=1.0 * font_scale, color="#444444"),
        whiskerprops=dict(color="#444444", linewidth=1.0 * font_scale),
        capprops=dict(color="#444444", linewidth=1.0 * font_scale),
        zorder=2,
    )
    box_colors = [lighten_color(color, amt) for amt in np.linspace(0.42, 0.05, len(hours))]
    for patch, patch_color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(patch_color)
        patch.set_alpha(0.92)

    ax.plot(
        positions,
        y_vals,
        color=color,
        linewidth=line_width * font_scale,
        linestyle="-",
        marker="o",
        markersize=4.6 * font_scale,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.1 * font_scale,
        alpha=0.92,
        zorder=4,
    )
    ax.scatter(positions, y_vals, s=point_size * font_scale, color=color, edgecolors="white", linewidths=0.7, zorder=5)

    ax.set_ylabel(metric_label, fontsize=11.5 * font_scale)
    ax.set_xlim(-0.6, len(hours) - 0.4)
    tick_positions = [i for i, hour in enumerate(hours) if hour % tick_step == 0 or hour == hours[-1]]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(hours[i]) for i in tick_positions], fontsize=10.0 * font_scale)
    ax.tick_params(axis="both", width=1.0 * font_scale, length=4.2 * font_scale)
    if metric_key == "r2":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    valid = np.asarray([v for v in y_vals if math.isfinite(v)], dtype=float)
    if valid.size > 1:
        y_min = float(valid.min())
        y_max = float(valid.max())
        pad = max((y_max - y_min) * 0.12, 1e-6)
        if metric_key == "r2":
            ax.set_ylim(max(-1.0, y_min - pad), min(1.02, y_max + pad))
        else:
            ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    add_hour_band_labels(ax, hours, font_scale=font_scale)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_vector_output(path):
        fig.savefig(path)
    else:
        fig.savefig(path, dpi=dpi)
    print(f"[info] Saved figure: {path}")


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    truth_dir = resolve_path(repo, args.truth_dir)
    hourly_csv = resolve_path(repo, args.hourly_csv)
    summary_csv = resolve_path(repo, args.summary_csv)

    clip_quantile = None
    if args.clip_quantile is not None:
        low, high = float(args.clip_quantile[0]), float(args.clip_quantile[1])
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"Invalid --clip-quantile {args.clip_quantile}; expected 0 <= low < high <= 1.")
        clip_quantile = (low, high)

    if not truth_dir.exists():
        raise FileNotFoundError(f"Ground truth folder not found: {truth_dir}")

    rows = collect_hourly_rows(
        repo=repo,
        truth_dir=truth_dir,
        date=args.date,
        hours=args.hours,
        min_truth=args.min_truth,
        clip_quantile=clip_quantile,
        truth_tpl=args.truth_tpl,
        pred_tpl=args.pred_tpl,
    )
    if not rows:
        raise RuntimeError("No hourly metrics were collected. Check baseline folders and file naming.")

    write_hourly_csv(rows, hourly_csv)
    summary_rows = collect_daily_summary_rows(
        repo=repo,
        truth_dir=truth_dir,
        date=args.date,
        hours=args.hours,
        min_truth=args.min_truth,
        clip_quantile=clip_quantile,
        truth_tpl=args.truth_tpl,
        pred_tpl=args.pred_tpl,
    )
    write_summary_csv(summary_rows, summary_csv)

    print("[info] CSV-only mode enabled; no figures will be generated.")


if __name__ == "__main__":
    main()
