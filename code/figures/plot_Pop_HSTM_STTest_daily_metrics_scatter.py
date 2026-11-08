#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "output_results").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(SCRIPT_DIR)

HEXBIN_CMAP = LinearSegmentedColormap.from_list(
    "osot_hexbin",
    ["#223A8F", "#2F62D0", "#5BB8FF", "#F1D34B", "#F28C28", "#C43C39"],
)


RING_GENERALIZATION_SCENARIOS = (
    {
        "tag": "a",
        "title": "Low-density Ring Generalization",
        "out_stem": "Spatio-temporal_generalization_low_density_ring",
        "pred_root": Path("output_results")
        / "dynamic_population_distribution"
        / "Spatial_Temporal_Generalization"
        / "low_density_ring_generalization",
        "inner_shp": Path("data") / "shapefiles" / "random_shps" / "random_polys.shp",
    },
    {
        "tag": "b",
        "title": "High-density Ring Generalization",
        "out_stem": "Spatio-temporal_generalization_high_density_ring",
        "pred_root": Path("output_results")
        / "dynamic_population_distribution"
        / "Spatial_Temporal_Generalization"
        / "high_density_ring_generalization",
        "inner_shp": Path("data") / "shapefiles" / "random_shps" / "random_polys_other.shp",
    },
)


@dataclass
class PanelData:
    pred_fit: np.ndarray
    truth_fit: np.ndarray
    pred_plot: np.ndarray
    truth_plot: np.ndarray
    panel_label: str
    metric_text: str


@dataclass
class ScenarioFigureData:
    tag: str
    title: str
    out_base: Path
    panels: List[PanelData]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot OSOT/Test daily aggregated predicted-vs-real scatter figures "
            "for 24 hours to 96 hours."
        )
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=RING_GENERALIZATION_SCENARIOS[0]["pred_root"],
        help="Root folder containing Pred_YYYYMMDD subfolders from OSIT.py.",
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=Path("data") / "mobile_phone_data",
        help="Ground-truth raster folder.",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["20170925", "20170926", "20170927", "20170928"],
        help="Dates to plot.",
    )
    parser.add_argument(
        "--hours",
        nargs="+",
        type=int,
        default=list(range(24)),
        help="Hours to aggregate for each date.",
    )
    parser.add_argument(
        "--pred-tpl",
        type=str,
        default="PredPop_{date}_{hour:02d}_real.tif",
        help="Prediction filename template.",
    )
    parser.add_argument(
        "--truth-tpl",
        type=str,
        default="point{date_key}{hour:02d}_500.tif",
        help="Ground-truth filename template.",
    )
    parser.add_argument(
        "--mask-ring-outer-shp",
        type=Path,
        default=Path("data") / "shapefiles" / "core_urban_area_Beijing.shp",
        help="Outer polygon shapefile used as the kept region.",
    )
    parser.add_argument(
        "--mask-ring-inner-shp",
        type=Path,
        default=Path("data") / "shapefiles" / "random_shps" / "random_polys.shp",
        help="Inner polygon shapefile used as the excluded training region.",
    )
    parser.add_argument(
        "--min-truth",
        type=float,
        default=1.0,
        help="Minimum ground-truth threshold for valid pixels.",
    )
    parser.add_argument(
        "--clip-quantile",
        nargs=2,
        type=float,
        default=[0.01, 0.99],
        metavar=("LOW", "HIGH"),
        help="Truth quantile clipping used for metric computation and regression fitting.",
    )
    parser.add_argument(
        "--plot-min-truth",
        type=float,
        default=1.0,
        help="Minimum truth value kept in the scatter plot. Metrics are not affected.",
    )
    parser.add_argument(
        "--hexbin-gridsize",
        type=int,
        default=65,
        help="Hexbin grid size for the density plot.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=14000,
        help="Maximum plotted points per date after random subsampling.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="2D density bins for scatter coloring.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[180.0, 300.0],
        metavar=("W", "H"),
        help="Figure size in inches for each standalone split figure.",
    )
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output figure path without extension. Defaults under pred-root.",
    )
    parser.add_argument(
        "--all-touched",
        dest="all_touched",
        action="store_true",
        help="Rasterize masks with GDAL -at so any touched pixel is included.",
    )
    parser.add_argument(
        "--no-all-touched",
        dest="all_touched",
        action="store_false",
        help="Rasterize masks without GDAL -at so only center-covered pixels are included.",
    )
    parser.add_argument(
        "--build-ring-generalization-compare",
        action="store_true",
        help=(
            "Build reproducible low-density/high-density ring generalization figures: "
            "two standalone 2x2 source figures plus combined comparison figures."
        ),
    )
    parser.add_argument(
        "--comparison-layout",
        choices=["vertical", "horizontal", "both"],
        default="both",
        help="Comparison figure layout used with --build-ring-generalization-compare.",
    )
    parser.add_argument(
        "--comparison-out-dir",
        type=Path,
        default=Path("output_results") / "figures",
        help="Output directory for combined comparison figures.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["split", "combined", "both"],
        default="split",
        help=(
            "Which ring-generalization figures to save. "
            "Default: split, saving low-density and high-density groups as separate figures."
        ),
    )
    parser.add_argument(
        "--comparison-vertical-figsize",
        nargs=2,
        type=float,
        default=[18.5, 29.5],
        metavar=("W", "H"),
        help="Vertical comparison figure size in inches.",
    )
    parser.add_argument(
        "--comparison-horizontal-figsize",
        nargs=2,
        type=float,
        default=[34.0, 15.2],
        metavar=("W", "H"),
        help="Horizontal comparison figure size in inches.",
    )
    parser.set_defaults(all_touched=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def prepare_gdal_env() -> Dict[str, str]:
    env = os.environ.copy()
    env_root = Path(sys.executable).resolve().parent

    proj_candidates = [
        env_root / "Library" / "share" / "proj",
        env_root / "share" / "proj",
    ]
    for candidate in proj_candidates:
        if candidate.exists():
            env["PROJ_LIB"] = str(candidate)
            env["PROJ_DATA"] = str(candidate)
            break

    gdal_data_candidates = [
        env_root / "Library" / "share" / "gdal",
        env_root / "share" / "gdal",
    ]
    for candidate in gdal_data_candidates:
        if candidate.exists():
            env["GDAL_DATA"] = str(candidate)
            break

    bin_candidates = [
        env_root / "Library" / "bin",
        env_root / "bin",
    ]
    existing_bins = [str(candidate) for candidate in bin_candidates if candidate.exists()]
    if existing_bins:
        current_path = env.get("PATH", "")
        env["PATH"] = os.pathsep.join(existing_bins + [current_path] if current_path else existing_bins)
    env["GTIFF_SRS_SOURCE"] = "EPSG"
    return env


def resolve_gdal_binary(binary_name: str) -> str:
    env_root = Path(sys.executable).resolve().parent
    candidates = [
        env_root / "Library" / "bin" / f"{binary_name}.exe",
        env_root / "bin" / f"{binary_name}.exe",
        env_root / "Library" / "bin" / binary_name,
        env_root / "bin" / binary_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return binary_name


def load_tif(path: Path) -> np.ndarray:
    tmp_path: Optional[Path] = None
    env = prepare_gdal_env()
    try:
        with tempfile.NamedTemporaryFile(suffix=".asc", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        cmd = [resolve_gdal_binary("gdal_translate"), "-quiet", "-of", "AAIGrid", str(path), str(tmp_path)]
        subprocess.run(cmd, check=True, env=env)
        header_lines = 0
        with tmp_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    header_lines += 1
                    continue
                token = stripped.split()[0]
                if any(ch.isalpha() for ch in token):
                    header_lines += 1
                    continue
                break
        arr = np.loadtxt(tmp_path, skiprows=header_lines, dtype=np.float64)
        tmp_path.unlink(missing_ok=True)
        return arr
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        with Image.open(path) as img:
            return np.asarray(img, dtype=np.float64)


def read_aaigrid_header(src: Path) -> Dict[str, float]:
    env = prepare_gdal_env()
    with tempfile.NamedTemporaryFile(suffix=".asc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [resolve_gdal_binary("gdal_translate"), "-quiet", "-of", "AAIGrid", str(src), str(tmp_path)]
        subprocess.run(cmd, check=True, env=env)
        header: Dict[str, float] = {}
        with tmp_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                token = parts[0]
                if not any(ch.isalpha() for ch in token):
                    break
                if len(parts) >= 2:
                    try:
                        header[token.lower()] = float(parts[1])
                    except ValueError:
                        pass
        return header
    finally:
        tmp_path.unlink(missing_ok=True)


def rasterize_mask_from_shp(shp_path: Path, ref_tif: Path, all_touched: bool = True) -> np.ndarray:
    env = prepare_gdal_env()
    header = read_aaigrid_header(ref_tif)
    ncols = int(header.get("ncols", 0))
    nrows = int(header.get("nrows", 0))
    cell = float(header.get("cellsize", 0.0))
    if ncols <= 0 or nrows <= 0 or cell <= 0:
        raise RuntimeError(f"Failed to read raster geometry from {ref_tif}")
    xll = float(header.get("xllcorner", header.get("xllcenter", 0.0)))
    yll = float(header.get("yllcorner", header.get("yllcenter", 0.0)))
    xur = xll + cell * ncols
    yur = yll + cell * nrows

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        cmd = [
            resolve_gdal_binary("gdal_rasterize"),
            "-q",
            "-burn",
            "1",
            "-a_nodata",
            "0",
            "-te",
            str(xll),
            str(yll),
            str(xur),
            str(yur),
            "-ts",
            str(ncols),
            str(nrows),
            "-of",
            "GTiff",
            str(shp_path),
            str(out_path),
        ]
        if all_touched:
            cmd.insert(2, "-at")
        subprocess.run(cmd, check=True, env=env)
        return load_tif(out_path) != 0
    finally:
        out_path.unlink(missing_ok=True)


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    diff = pred - truth
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff) / (truth + 1e-5)) * 100.0)
    truth_mean = float(np.mean(truth))
    ss_tot = float(np.sum((truth - truth_mean) ** 2))
    ss_res = float(np.sum((truth - pred) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return {"r2": r2, "rmse": math.sqrt(mse), "mae": mae, "mape": mape, "tem": math.log1p(mse)}


def collect_daily_vectors(
        pred_root: Path,
        truth_dir: Path,
        date: str,
        hours: Sequence[int],
        pred_tpl: str,
        truth_tpl: str,
        mask: np.ndarray,
        min_truth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_dir = pred_root / f"Pred_{date}"
    date_key = date[-4:]
    pred_parts: List[np.ndarray] = []
    truth_parts: List[np.ndarray] = []
    mask_flat = mask.ravel()

    for hour in hours:
        pred_path = pred_dir / pred_tpl.format(date=date, hour=hour)
        truth_path = truth_dir / truth_tpl.format(date=date, date_key=date_key, hour=hour)
        pred = load_tif(pred_path).ravel()[mask_flat]
        truth = load_tif(truth_path).ravel()[mask_flat]
        keep = np.isfinite(pred) & np.isfinite(truth)
        if min_truth is not None:
            keep &= truth >= min_truth
        pred_parts.append(pred[keep])
        truth_parts.append(truth[keep])

    return np.concatenate(pred_parts), np.concatenate(truth_parts)


def filter_points_for_plot(
        truth: np.ndarray,
        pred: np.ndarray,
        plot_min_truth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if plot_min_truth is None:
        return truth, pred
    keep = truth > plot_min_truth
    truth_f = truth[keep]
    pred_f = pred[keep]
    if truth_f.size < 2:
        return truth, pred
    return truth_f, pred_f


def apply_quantile_clip(
        pred: np.ndarray,
        truth: np.ndarray,
        clip_quantile: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    if clip_quantile is None:
        return pred, truth
    q_low, q_high = float(clip_quantile[0]), float(clip_quantile[1])
    ql, qh = np.quantile(truth, [q_low, q_high])
    keep = (truth >= ql) & (truth <= qh)
    pred_f = pred[keep]
    truth_f = truth[keep]
    if truth_f.size < 2:
        return pred, truth
    return pred_f, truth_f


def sample_points(
        x: np.ndarray,
        y: np.ndarray,
        max_points: int,
        seed: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    n = x.size
    if n == 0:
        return x, y

    rng = np.random.default_rng(seed)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]
    return x, y


def draw_hexbin_style(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        gridsize: int,
) -> None:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=gridsize, range=[[x_min, x_max], [y_min, y_max]])
    if np.max(hist) <= 0:
        return

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    dx = x_centers[1] - x_centers[0] if x_centers.size > 1 else 0.0
    cmap = HEXBIN_CMAP
    hmax = float(np.max(hist))
    positive = hist[hist > 0]
    min_count = max(2.0, float(np.quantile(positive, 0.65))) if positive.size > 0 else 1.0

    for ix, xc in enumerate(x_centers):
        for iy, yc in enumerate(y_centers):
            count = float(hist[ix, iy])
            if count < min_count:
                continue
            xc_plot = xc + (0.5 * dx if iy % 2 else 0.0)
            intensity = np.log1p(count) / np.log1p(hmax)
            color = cmap(intensity)
            ax.plot(
                xc_plot,
                yc,
                linestyle="None",
                marker="h",
                markersize=62.0,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.35 + 0.65 * intensity,
                zorder=2.6,
            )


def compute_regression_band(
        x: np.ndarray,
        y: np.ndarray,
        lo: float,
        hi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    reg_x = np.linspace(lo, hi, 240, dtype=np.float64)

    if x.size < 3 or np.allclose(np.std(x), 0.0):
        reg_y = reg_x.copy()
        return reg_x, reg_y, reg_y, reg_y

    try:
        slope, intercept = np.polyfit(x, y, deg=1)
    except np.linalg.LinAlgError:
        reg_y = reg_x.copy()
        return reg_x, reg_y, reg_y, reg_y
    reg_y = slope * reg_x + intercept

    n = x.size
    if n < 3:
        return reg_x, reg_y, reg_y, reg_y

    fitted = slope * x + intercept
    resid = y - fitted
    s_err = math.sqrt(float(np.sum(resid ** 2)) / max(n - 2, 1))
    x_mean = float(np.mean(x))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 0:
        return reg_x, reg_y, reg_y, reg_y

    # With very large samples, the true confidence interval can collapse onto the
    # regression line. A prediction-style band is visually meaningful here.
    se = s_err * np.sqrt(1.0 + (1.0 / n) + ((reg_x - x_mean) ** 2) / sxx)
    band = 1.96 * se
    return reg_x, reg_y, reg_y - band, reg_y + band


def compute_point_density(
        x: np.ndarray,
        y: np.ndarray,
        bins: int,
) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]])
    x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, hist.shape[0] - 1)
    y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, hist.shape[1] - 1)
    density = hist[x_idx, y_idx]
    return np.log1p(density)


def build_metric_text(metrics: Dict[str, float]) -> str:
    tem = metrics["tem"]
    return (
        f"R\u00b2: {metrics['r2']:.4f}\n"
        f"TEM: {tem:.2f}\n"
        f"MAE: {metrics['mae']:.2f}\n"
        f"MAPE: {metrics['mape']:.2f}%"
    )


def make_legend_handles() -> List[Line2D]:
    return [
        Line2D([0], [0], marker="h", linestyle="None", color="#F28C28", markersize=120, label="Hexbin density"),
        Line2D([0], [0], marker="o", linestyle="None", color="#2F62D0", markersize=100, label="Density-colored points"),
        Line2D([0], [0], linestyle="--", color="#5F5F5F", linewidth=30.0, label="1:1 line"),
        Line2D([0], [0], linestyle="-", color="#C43C39", linewidth=42.0, label="Regression line"),
    ]


def prepare_panel_dataset(
        *,
        pred_root: Path,
        truth_dir: Path,
        outer_shp: Path,
        inner_shp: Path,
        dates: Sequence[str],
        hours: Sequence[int],
        pred_tpl: str,
        truth_tpl: str,
        min_truth: Optional[float],
        clip_quantile: Optional[Sequence[float]],
        plot_min_truth: Optional[float],
        all_touched: bool,
) -> List[PanelData]:
    ref_truth = truth_dir / f"point{dates[0][-4:]}00_500.tif"
    outer_mask = rasterize_mask_from_shp(outer_shp, ref_truth, all_touched=all_touched)
    inner_mask = rasterize_mask_from_shp(inner_shp, ref_truth, all_touched=all_touched)
    mask = outer_mask & (~inner_mask)

    panel_titles = ["0-24hours", "24-48hours", "48-72hours", "72-96hours"]
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    panels: List[PanelData] = []

    for idx, date in enumerate(dates):
        pred, truth = collect_daily_vectors(
            pred_root=pred_root,
            truth_dir=truth_dir,
            date=date,
            hours=hours,
            pred_tpl=pred_tpl,
            truth_tpl=truth_tpl,
            mask=mask,
            min_truth=min_truth,
        )
        pred_eval, truth_eval = apply_quantile_clip(pred, truth, clip_quantile)
        computed = compute_metrics(pred_eval, truth_eval)
        metric_text = build_metric_text(computed)
        panel_name = panel_titles[idx] if idx < len(panel_titles) else f"{date[:4]}-{date[4:6]}-{date[6:]}"
        panel_label = f"({panel_letters[idx]}) {panel_name}"
        truth_plot, pred_plot = filter_points_for_plot(truth_eval, pred_eval, plot_min_truth)
        panels.append(
            PanelData(
                pred_fit=pred_eval,
                truth_fit=truth_eval,
                pred_plot=pred_plot,
                truth_plot=truth_plot,
                panel_label=panel_label,
                metric_text=metric_text,
            )
        )

    return panels


def draw_panel(
        ax: plt.Axes,
        pred_fit: np.ndarray,
        truth_fit: np.ndarray,
        pred_plot: np.ndarray,
        truth_plot: np.ndarray,
        panel_label: str,
        metric_text: str,
        max_points: int,
        hexbin_gridsize: int,
        density_bins: int,
        show_ylabel: bool = True,
) -> None:
    x_plot, y_plot = sample_points(pred_plot, truth_plot, max_points=max_points)
    point_density = compute_point_density(x_plot, y_plot, bins=density_bins)
    point_order = np.argsort(point_density)
    x_plot = x_plot[point_order]
    y_plot = y_plot[point_order]
    point_density = point_density[point_order]
    draw_hexbin_style(ax, pred_fit, truth_fit, gridsize=hexbin_gridsize)

    all_max = float(max(np.max(pred_fit), np.max(truth_fit)))
    pad = all_max * 0.12 if all_max > 0 else 1.0
    lo = 0.0
    hi = all_max + pad

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#5F5F5F", linewidth=34.0, label="1:1 line", zorder=3.8)
    reg_x, reg_y, _, _ = compute_regression_band(pred_fit, truth_fit, lo, hi)
    ax.plot(reg_x, reg_y, color="#C43C39", linewidth=46.0, label="Regression line", zorder=4.2)
    ax.scatter(
        x_plot,
        y_plot,
        c=point_density,
        cmap=HEXBIN_CMAP,
        s=520.0,
        alpha=0.94,
        linewidths=0.0,
        edgecolors="none",
        zorder=4.0,
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    sci_fmt_x = ScalarFormatter(useMathText=True)
    sci_fmt_x.set_scientific(True)
    sci_fmt_x.set_powerlimits((2, 2))
    sci_fmt_y = ScalarFormatter(useMathText=True)
    sci_fmt_y.set_scientific(True)
    sci_fmt_y.set_powerlimits((2, 2))
    ax.xaxis.set_major_formatter(sci_fmt_x)
    ax.yaxis.set_major_formatter(sci_fmt_y)
    ax.grid(True, linestyle="--", linewidth=5.5, alpha=0.30, color="#B8C3D1")
    ax.set_title(panel_label, fontsize=352, fontweight="bold", pad=72)
    ax.set_xlabel("Estimated population", fontsize=360, fontweight="bold", labelpad=72)
    if show_ylabel:
        ax.set_ylabel("Observed population", fontsize=360, fontweight="bold", labelpad=72)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=272, pad=32)
    ax.tick_params(axis="y", labelsize=272, pad=32)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    # Enlarge scientific notation offset text so the exponent "2" in ×10² is more readable.
    ax.xaxis.get_offset_text().set_fontsize(328)
    ax.yaxis.get_offset_text().set_fontsize(328)
    ax.xaxis.get_offset_text().set_fontweight("bold")
    ax.yaxis.get_offset_text().set_fontweight("bold")

    # Keep a single origin label: hide y-axis "0" so (0, 0) does not show two zeros.
    yticks = ax.get_yticks()
    ylabels = ax.get_yticklabels()
    for i, tick in enumerate(yticks):
        if i < len(ylabels) and np.isclose(tick, 0.0):
            ylabels[i].set_visible(False)
            break

    ax.text(
        0.04,
        0.96,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=336,
        fontweight="bold",
        color="#6D8FB9",
        bbox=dict(boxstyle="round,pad=0.34", facecolor="white", edgecolor="none", alpha=0.35),
        zorder=5,
    )
    return None


def render_panels_to_axes(
        axes: Sequence[plt.Axes],
        panels: Sequence[PanelData],
        *,
        max_points: int,
        hexbin_gridsize: int,
        density_bins: int,
        left_column_only: bool,
) -> None:
    for idx, panel in enumerate(panels):
        draw_panel(
            ax=axes[idx],
            pred_fit=panel.pred_fit,
            truth_fit=panel.truth_fit,
            pred_plot=panel.pred_plot,
            truth_plot=panel.truth_plot,
            panel_label=panel.panel_label,
            metric_text=panel.metric_text,
            max_points=max_points,
            hexbin_gridsize=hexbin_gridsize,
            density_bins=density_bins,
            show_ylabel=(not left_column_only or idx % 2 == 0),
        )

    for ax in axes[len(panels):]:
        ax.axis("off")


def render_single_scenario_figure(
        scenario: ScenarioFigureData,
        *,
        figsize: Sequence[float],
        dpi: int,
        max_points: int,
        hexbin_gridsize: int,
        density_bins: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=tuple(figsize))
    axes = axes.ravel()
    render_panels_to_axes(
        axes,
        scenario.panels,
        max_points=max_points,
        hexbin_gridsize=hexbin_gridsize,
        density_bins=density_bins,
        left_column_only=True,
    )
    fig.text(
        0.03,
        0.935,
        f"({scenario.tag}) {scenario.title}",
        ha="left",
        va="top",
        fontsize=448,
        fontweight="bold",
    )
    legend = fig.legend(
        handles=make_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.56, 0.915),
        ncol=4,
        fontsize=336,
        frameon=True,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.125, right=0.985, top=0.845, bottom=0.095, wspace=0.34, hspace=0.34)
    save_figure(fig, scenario.out_base, dpi=dpi)
    plt.close(fig)


def add_group_separator(fig: plt.Figure, layout: str, first_group_axes: Sequence[plt.Axes], second_group_axes: Sequence[plt.Axes]) -> None:
    if layout == "vertical":
        lower_y = min(ax.get_position().y0 for ax in first_group_axes)
        upper_y = max(ax.get_position().y1 for ax in second_group_axes)
        y = (lower_y + upper_y) * 0.5
        fig.add_artist(
            Line2D(
                [0.03, 0.985],
                [y, y],
                transform=fig.transFigure,
                linestyle=(0, (4, 4)),
                linewidth=1.2,
                color="#A7A7A7",
                alpha=0.9,
            )
        )
    else:
        right_x = max(ax.get_position().x1 for ax in first_group_axes)
        left_x = min(ax.get_position().x0 for ax in second_group_axes)
        x = (right_x + left_x) * 0.5
        fig.add_artist(
            Line2D(
                [x, x],
                [0.045, 0.975],
                transform=fig.transFigure,
                linestyle=(0, (4, 4)),
                linewidth=1.2,
                color="#A7A7A7",
                alpha=0.9,
            )
        )


def render_comparison_figure(
        scenarios: Sequence[ScenarioFigureData],
        *,
        layout: str,
        figsize: Sequence[float],
        out_base: Path,
        dpi: int,
        max_points: int,
        hexbin_gridsize: int,
        density_bins: int,
) -> None:
    if len(scenarios) != 2:
        raise ValueError("comparison figure currently expects exactly two scenarios")

    fig = plt.figure(figsize=tuple(figsize))
    outer = fig.add_gridspec(
        2, 1, left=0.055, right=0.988, top=0.92, bottom=0.055, hspace=0.24
    ) if layout == "vertical" else fig.add_gridspec(
        1, 2, left=0.04, right=0.992, top=0.91, bottom=0.08, wspace=0.14
    )

    scenario_axes: List[List[plt.Axes]] = []
    for outer_idx, scenario in enumerate(scenarios):
        slot = outer[outer_idx, 0] if layout == "vertical" else outer[0, outer_idx]
        inner = slot.subgridspec(2, 2, wspace=0.14, hspace=0.25)
        axes = [fig.add_subplot(inner[r, c]) for r in range(2) for c in range(2)]
        render_panels_to_axes(
            axes,
            scenario.panels,
            max_points=max_points,
            hexbin_gridsize=hexbin_gridsize,
            density_bins=density_bins,
            left_column_only=True,
        )
        axes[0].legend(handles=make_legend_handles(), loc="lower right", fontsize=14, frameon=True)
        scenario_axes.append(axes)

    first_title = f"({scenarios[0].tag}) {scenarios[0].title}"
    second_title = f"({scenarios[1].tag}) {scenarios[1].title}"
    if layout == "vertical":
        title_pad = 0.03
        first_title_y = min(max(ax.get_position().y1 for ax in scenario_axes[0]) + title_pad, 0.99)
        second_title_y = min(max(ax.get_position().y1 for ax in scenario_axes[1]) + title_pad, 0.99)
        fig.text(0.03, first_title_y, first_title, ha="left", va="top", fontsize=28, fontweight="bold")
        fig.text(0.03, second_title_y, second_title, ha="left", va="top", fontsize=28, fontweight="bold")
    else:
        title_pad = 0.03
        first_title_y = min(max(ax.get_position().y1 for ax in scenario_axes[0]) + title_pad, 0.99)
        second_title_y = min(max(ax.get_position().y1 for ax in scenario_axes[1]) + title_pad, 0.99)
        fig.text(0.02, first_title_y, first_title, ha="left", va="top", fontsize=28, fontweight="bold")
        fig.text(0.515, second_title_y, second_title, ha="left", va="top", fontsize=28, fontweight="bold")

    add_group_separator(fig, layout, scenario_axes[0], scenario_axes[1])
    save_figure(fig, out_base, dpi=dpi)
    plt.close(fig)


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    tif_path = out_base.with_suffix(".tif")
    fig.savefig(
        tif_path,
        dpi=dpi,
        format="tiff",
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def build_ring_generalization_scenarios(args: argparse.Namespace) -> List[ScenarioFigureData]:
    truth_dir = resolve_path(args.truth_dir)
    outer_shp = resolve_path(args.mask_ring_outer_shp)
    scenario_out_dir = resolve_path(args.comparison_out_dir)
    scenarios: List[ScenarioFigureData] = []

    for config in RING_GENERALIZATION_SCENARIOS:
        pred_root = resolve_path(config["pred_root"])
        inner_shp = resolve_path(config["inner_shp"])
        panels = prepare_panel_dataset(
            pred_root=pred_root,
            truth_dir=truth_dir,
            outer_shp=outer_shp,
            inner_shp=inner_shp,
            dates=args.dates,
            hours=args.hours,
            pred_tpl=args.pred_tpl,
            truth_tpl=args.truth_tpl,
            min_truth=args.min_truth,
            clip_quantile=args.clip_quantile,
            plot_min_truth=args.plot_min_truth,
            all_touched=args.all_touched,
        )
        scenarios.append(
            ScenarioFigureData(
                tag=config["tag"],
                title=config["title"],
                out_base=scenario_out_dir / str(config["out_stem"]),
                panels=panels,
            )
        )

    return scenarios


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.sans-serif": ["Times New Roman"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )

    scenarios = build_ring_generalization_scenarios(args)
    compare_out_dir = resolve_path(args.comparison_out_dir)

    if args.output_mode in {"split", "both"}:
        for scenario in scenarios:
            render_single_scenario_figure(
                scenario,
                figsize=args.figsize,
                dpi=args.dpi,
                max_points=args.max_points,
                hexbin_gridsize=args.hexbin_gridsize,
                density_bins=args.bins,
            )
            print(f"[ok] Saved figure to {scenario.out_base.with_suffix('.tif')}")

    if args.output_mode in {"combined", "both"}:
        out_base = compare_out_dir / "Spatio-temporal_generalization_low_vs_high_density_ring_vertical"
        render_comparison_figure(
            scenarios,
            layout="vertical",
            figsize=args.comparison_vertical_figsize,
            out_base=out_base,
            dpi=args.dpi,
            max_points=args.max_points,
            hexbin_gridsize=args.hexbin_gridsize,
            density_bins=args.bins,
        )
        print(f"[ok] Saved figure to {out_base.with_suffix('.tif')}")


if __name__ == "__main__":
    main()
