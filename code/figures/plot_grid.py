"""
Publication-style full-day hourly population grid.

This script draws all 24 hourly population rasters by default. It is intended
as a clearer replacement for a very dense 24-panel map plate: the grid uses a
shared color scale, compact hour labels, larger map panels, tight gutters, and
a high-contrast north arrow that remains visible after journal-page scaling.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch, Polygon
import matplotlib.patheffects as path_effects

from plot_pop24_grid import (
    DEFAULT_BOUNDARY,
    DEFAULT_DATE,
    DEFAULT_HOURS,
    DEFAULT_INPUT_DIR,
    REPO_ROOT,
    add_scalebar_and_compass,
    build_cmap,
    collect_hourly_tifs,
    draw_raster,
    load_boundary,
    rasterize_boundary_mask,
    read_raster,
    resolve_input_dir,
)


DEFAULT_OUTPUT = REPO_ROOT / "output_results" / "figures" / "hourly_population_full_day_grid_4x6_times_large_text.tif"
DEFAULT_FONT_FAMILY = "Times New Roman"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [DEFAULT_FONT_FAMILY, "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 13,
        "axes.linewidth": 0.6,
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "legend.frameon": False,
    }
)


def normalize_hours(hours: Sequence[int]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for hour in hours:
        h = int(hour)
        if h < 0 or h > 23:
            raise ValueError(f"Invalid hour {h}; expected 0..23.")
        if h not in seen:
            normalized.append(h)
            seen.add(h)
    if not normalized:
        raise ValueError("At least one hour is required.")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a full-day hourly population-distribution map grid.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing hourly prediction rasters, or a root containing Pred_YYYYMMDD.",
    )
    parser.add_argument("--date", type=str, default=DEFAULT_DATE, help="Date in YYYYMMDD format.")
    parser.add_argument(
        "--hours",
        nargs="+",
        type=int,
        default=list(DEFAULT_HOURS),
        help="Hours to draw. Default is the full day: 0 1 ... 23.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of map columns. Default 4 gives the manuscript-style 4x6 layout for 24 hours.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--fig-width",
        type=float,
        default=14.0,
        help="Figure width in inches before raster export.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=21.0,
        help="Figure height in inches before raster export.",
    )
    parser.add_argument(
        "--wspace",
        type=float,
        default=0.002,
        help="Horizontal gutter between map panels. Lower values enlarge each map.",
    )
    parser.add_argument(
        "--hspace",
        type=float,
        default=0.190,
        help="Vertical gutter between map panels. Keep enough room for panel labels.",
    )
    parser.add_argument(
        "--panel-title-size",
        type=float,
        default=17.0,
        help="Panel-title font size in points.",
    )
    parser.add_argument(
        "--colorbar-label-size",
        type=float,
        default=17.5,
        help="Colorbar label font size in points.",
    )
    parser.add_argument(
        "--colorbar-tick-size",
        type=float,
        default=15.5,
        help="Colorbar tick-label font size in points.",
    )
    parser.add_argument(
        "--vmax-quantile",
        type=float,
        default=0.995,
        help="Upper color-limit quantile from positive population values.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title. Empty by default for manuscript layout.",
    )
    parser.add_argument(
        "--repeat-map-elements",
        action="store_true",
        help="Repeat both north arrow and scale bar in every panel. Default repeats only the north arrow.",
    )
    parser.add_argument(
        "--north-arrow",
        choices=("all", "first", "none"),
        default="all",
        help="Where to draw the north arrow. Default: all panels for readability in a 24-panel plate.",
    )
    parser.add_argument(
        "--north-arrow-scale",
        type=float,
        default=1.55,
        help="Scale factor for the high-contrast north arrow.",
    )
    parser.add_argument(
        "--hide-scale-bar",
        action="store_true",
        help="Hide the scale bar. By default a small scale bar is drawn in the first panel only.",
    )
    parser.add_argument(
        "--show-boundary",
        action="store_true",
        help="Overlay the study-area boundary outline. Disabled by default to keep cells clear.",
    )
    return parser.parse_args()


def load_hourly_samples(pop_dir: Path, date: str, hours: Sequence[int]) -> list[dict]:
    tif_paths = collect_hourly_tifs(pop_dir, date, hours)
    samples: list[dict] = []
    for path in tif_paths:
        sample = read_raster(path)
        sample["hour"] = int(path.stem.split("_")[-2])
        sample["path"] = path
        samples.append(sample)
    return samples


def apply_study_mask(samples: list[dict]) -> None:
    study_mask = rasterize_boundary_mask(samples[0], DEFAULT_BOUNDARY)
    if study_mask is None:
        raise RuntimeError(
            "Failed to rasterize the study-area boundary. "
            f"Check boundary file: {DEFAULT_BOUNDARY}"
        )
    for sample in samples:
        arr = np.ma.filled(sample["array"], np.nan).astype(np.float32, copy=False)
        arr[study_mask & ~np.isfinite(arr)] = 0.0
        arr[arr < 0] = 0.0
        arr[~study_mask] = np.nan
        sample["array"] = np.ma.masked_invalid(arr)


def compute_shared_norm(samples: list[dict], vmax_quantile: float) -> Normalize:
    all_vals = np.hstack([sample["array"].compressed() for sample in samples])
    positive_vals = all_vals[all_vals > 0]
    if positive_vals.size == 0:
        positive_vals = np.asarray([1.0], dtype=np.float32)
    q = min(max(float(vmax_quantile), 0.50), 1.00)
    return Normalize(vmin=0.0, vmax=float(np.quantile(positive_vals, q)))


def set_clean_map_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_high_contrast_north_arrow(
    ax: plt.Axes,
    x: float = 0.060,
    y: float = 0.825,
    scale: float = 1.0,
    label_size: float | None = None,
) -> None:
    """Draw a compact north arrow in axes coordinates with a white halo/backing."""
    s = max(float(scale), 0.35)
    width = 0.066 * s
    height = 0.178 * s

    backing = FancyBboxPatch(
        (x - width * 0.50, y - height * 0.43),
        width,
        height,
        boxstyle=f"round,pad={0.006 * s:.4f},rounding_size={0.012 * s:.4f}",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="none",
        alpha=0.78,
        zorder=18,
        clip_on=False,
    )
    ax.add_patch(backing)

    arrow_head = Polygon(
        [
            (x, y + 0.055 * s),
            (x - 0.024 * s, y - 0.014 * s),
            (x + 0.024 * s, y - 0.014 * s),
        ],
        closed=True,
        transform=ax.transAxes,
        facecolor="#111111",
        edgecolor="#111111",
        linewidth=0.65,
        zorder=20,
        clip_on=False,
    )
    arrow_head.set_path_effects(
        [path_effects.Stroke(linewidth=2.0 * s, foreground="white"), path_effects.Normal()]
    )
    ax.add_patch(arrow_head)

    shaft = Polygon(
        [
            (x - 0.0055 * s, y - 0.014 * s),
            (x + 0.0055 * s, y - 0.014 * s),
            (x + 0.0055 * s, y - 0.070 * s),
            (x - 0.0055 * s, y - 0.070 * s),
        ],
        closed=True,
        transform=ax.transAxes,
        facecolor="#111111",
        edgecolor="#111111",
        linewidth=0.45,
        zorder=20,
        clip_on=False,
    )
    shaft.set_path_effects(
        [path_effects.Stroke(linewidth=1.6 * s, foreground="white"), path_effects.Normal()]
    )
    ax.add_patch(shaft)

    north_label = ax.text(
        x,
        y + 0.066 * s,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=(10.0 * s if label_size is None else float(label_size)),
        fontfamily=DEFAULT_FONT_FAMILY,
        fontweight="bold",
        color="#111111",
        zorder=21,
        clip_on=False,
    )
    north_label.set_path_effects(
        [path_effects.Stroke(linewidth=1.8 * s, foreground="white"), path_effects.Normal()]
    )


def save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        tmp = output.with_name(f"{output.stem}__tmp_rgba{output.suffix}")
        fig.savefig(tmp, dpi=dpi, format="tiff", bbox_inches="tight", pad_inches=0.03)
        try:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            with Image.open(tmp) as image:
                image.load()
                image_dpi = image.info.get("dpi", (dpi, dpi))
                rgb = image.convert("RGB")
            try:
                rgb.save(output, format="TIFF", dpi=image_dpi, compression="tiff_adobe_deflate")
            finally:
                rgb.close()
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except PermissionError:
                print(f"[warn] Temporary TIFF is still locked and was kept: {tmp}")
    else:
        fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.03)


def main() -> None:
    args = parse_args()
    hours = normalize_hours(args.hours)
    pop_dir = resolve_input_dir(args.input_dir, args.date)
    samples = load_hourly_samples(pop_dir, args.date, hours)
    apply_study_mask(samples)

    cmap = build_cmap()
    cmap.set_bad(color="white")
    norm = compute_shared_norm(samples, args.vmax_quantile)

    n_panels = len(samples)
    ncols = max(1, min(int(args.columns), n_panels))
    nrows = int(math.ceil(n_panels / ncols))

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi)
    gs = fig.add_gridspec(
        nrows=nrows + 1,
        ncols=ncols,
        height_ratios=[*[1.0] * nrows, 0.125],
        left=0.018,
        right=0.992,
        top=0.958 if args.title else 0.982,
        bottom=0.070,
        wspace=max(float(args.wspace), 0.0),
        hspace=max(float(args.hspace), 0.0),
    )

    boundary = None
    if args.show_boundary and samples[0]["transform"] is not None:
        boundary = load_boundary(DEFAULT_BOUNDARY, samples[0]["crs"])

    for idx, sample in enumerate(samples):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        draw_raster(ax, sample["array"], sample["transform"], cmap, norm)

        if boundary is not None:
            try:
                boundary.plot(ax=ax, facecolor="none", edgecolor="#6f1d45", linewidth=0.45)
            except Exception as exc:
                print(f"[warn] Skip boundary overlay: {exc}")

        if args.north_arrow == "all" or args.repeat_map_elements or (args.north_arrow == "first" and idx == 0):
            add_high_contrast_north_arrow(
                ax,
                x=0.060,
                y=0.825,
                scale=args.north_arrow_scale,
                label_size=10.5 * args.north_arrow_scale,
            )

        if not args.hide_scale_bar and (args.repeat_map_elements or idx == 0):
            add_scalebar_and_compass(
                ax,
                sample["array"],
                sample["transform"],
                draw_compass=False,
                color="white",
                font_scale=0.70,
            )

        ax.text(
            0.018,
            1.010,
            f"({chr(97 + idx)}) Hour {sample['hour']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=float(args.panel_title_size),
            fontfamily=DEFAULT_FONT_FAMILY,
            fontweight="bold",
            color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.65),
            zorder=12,
            clip_on=False,
        )
        set_clean_map_axis(ax)

    # Blank cells when the requested hour count is not a full rectangular grid.
    for empty_idx in range(n_panels, nrows * ncols):
        ax = fig.add_subplot(gs[empty_idx // ncols, empty_idx % ncols])
        ax.set_axis_off()

    cax = fig.add_subplot(gs[nrows, 0:ncols])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(
        labelsize=float(args.colorbar_tick_size),
        length=3.5,
        width=0.75,
        pad=3,
    )
    cbar.set_label(
        "Estimated population (people per 500 m grid)",
        fontsize=float(args.colorbar_label_size),
        fontfamily=DEFAULT_FONT_FAMILY,
        fontweight="bold",
        labelpad=4,
    )
    for tick_label in cbar.ax.get_xticklabels():
        tick_label.set_fontfamily(DEFAULT_FONT_FAMILY)
        tick_label.set_fontsize(float(args.colorbar_tick_size))
        tick_label.set_fontweight("bold")
    cbar.ax.xaxis.label.set_fontfamily(DEFAULT_FONT_FAMILY)
    cbar.ax.xaxis.label.set_fontsize(float(args.colorbar_label_size))
    cbar.ax.xaxis.label.set_fontweight("bold")

    if args.title:
        fig.suptitle(
            args.title,
            y=0.993,
            fontsize=max(float(args.panel_title_size), 18.0),
            fontfamily=DEFAULT_FONT_FAMILY,
            fontweight="bold",
        )

    save_figure(fig, args.output, args.dpi)
    plt.close(fig)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
