"""
Plot hourly population rasters in a compact grid.

By default the script draws the full 24-hour sequence in a 6x4 layout.
Pass --hours to draw a selected subset.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import sys
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Avoid importing local stub packages (e.g., ./rasterio) from this repo.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != SCRIPT_DIR]

try:
    import rasterio  # type: ignore
    RASTERIO_AVAILABLE = True
    RASTERIO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - env-dependent fallback
    rasterio = None  # type: ignore
    RASTERIO_AVAILABLE = False
    RASTERIO_IMPORT_ERROR = exc
    import tifffile  # type: ignore

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)

PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"
PROJECT_ROOT = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "output_results").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root(PROJECT_ROOT)
DEFAULT_DATE = "20170925"
DEFAULT_HOURS = tuple(range(24))
DEFAULT_INPUT_ROOT = REPO_ROOT / "output_results" / "dynamic_population_distribution" / "Pop-HSTM"
DEFAULT_INPUT_DIR = DEFAULT_INPUT_ROOT / "Pred_20170925"
DEFAULT_OUTPUT = REPO_ROOT / "output_results" / "figures" / "pop_24hours_grid_6x4.tif"
DEFAULT_BOUNDARY = REPO_ROOT / "data" / "shapefiles" / "core_urban_area_Beijing.shp"


def resolve_input_dir(input_dir: Path, date: str = DEFAULT_DATE) -> Path:
    direct_sample = input_dir / f"PredPop_{date}_00_real.tif"
    if direct_sample.exists():
        return input_dir
    nested_dir = input_dir / f"Pred_{date}"
    if nested_dir.is_dir():
        return nested_dir
    pop_hstm_dir = input_dir / "Pop-HSTM"
    nested_pop_hstm_dir = pop_hstm_dir / f"Pred_{date}"
    if nested_pop_hstm_dir.is_dir():
        return nested_pop_hstm_dir
    return input_dir


def load_boundary(shp_path: Path, target_crs) -> Optional["gpd.GeoDataFrame"]:
    """
    Load an outline shapefile if it exists and geopandas is available.
    Returns None when either condition is not met.
    """
    if not shp_path.exists():
        return None
    try:
        import geopandas as gpd  # type: ignore
    except Exception:
        return None
    gdf = gpd.read_file(shp_path)
    gdf = gdf[~gdf.is_empty & gdf.is_valid]
    if target_crs is not None and not gdf.crs == target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf if not gdf.empty else None


def build_cmap() -> LinearSegmentedColormap:
    """
    Blue-to-amber ramp tuned for population heat (inspired by provided figures).
    Darker blue for low, bright amber for high.
    """
    return LinearSegmentedColormap.from_list(
        "pop_heat",
        [
            "#0b2c4d",
            "#1f4f7a",
            "#3b7cb5",
            "#77b7e5",
            "#f6d880",
            "#f3a444",
            "#d9531e",
        ],
    )


def compute_bounds(arr: np.ndarray, transform) -> tuple[float, float, float, float]:
    """Return left, right, bottom, top in map units for an array+transform."""
    height, width = arr.shape
    if transform is None:
        # Pixel coordinate fallback when no georeference backend is available.
        return 0.0, float(width), 0.0, float(height)
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return left, right, bottom, top


def draw_raster(ax: plt.Axes, arr: np.ndarray, transform, cmap, norm) -> None:
    """
    Draw raster with map extent using matplotlib only (no rasterio.plot dependency).
    """
    left, right, bottom, top = compute_bounds(arr, transform)
    ax.imshow(
        arr,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=(left, right, bottom, top),
        origin="upper",
    )


def read_raster(path: Path) -> dict:
    """
    Read one raster file and return array + optional georeference metadata.
    Falls back to tifffile when rasterio is unavailable in current environment.
    """
    if RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:  # type: ignore[attr-defined]
            arr = src.read(1)
            valid_mask = src.read_masks(1) > 0
            nodata = src.nodata
            transform = src.transform
            crs = src.crs
    else:
        arr = tifffile.imread(str(path))
        if arr.ndim > 2:
            arr = arr[0]
        nodata = None
        transform = None
        crs = None
        valid_mask = np.isfinite(arr)

    masked = (
        np.ma.masked_equal(arr, nodata)
        if nodata is not None
        else np.ma.masked_invalid(arr)
    )
    return {
        "array": masked,
        "valid_mask": valid_mask,
        "transform": transform,
        "crs": crs,
    }


def rasterize_boundary_mask(sample: dict, boundary_path: Path) -> Optional[np.ndarray]:
    """
    Rasterize the study-area boundary to keep valid zero-valued cells visible.
    """
    if not RASTERIO_AVAILABLE or sample["transform"] is None:
        return None

    try:
        from rasterio.features import rasterize  # type: ignore[attr-defined]
        from rasterio.warp import transform_geom  # type: ignore[attr-defined]
    except Exception:
        return None

    geometries = []
    try:
        import fiona  # type: ignore

        with fiona.open(boundary_path) as src:
            source_crs = src.crs_wkt or src.crs
            for feature in src:
                geom = feature.get("geometry")
                if not geom:
                    continue
                if source_crs and sample["crs"] is not None:
                    geom = transform_geom(source_crs, sample["crs"], geom)
                geometries.append(geom)
    except Exception:
        boundary = load_boundary(boundary_path, sample["crs"])
        if boundary is not None and not boundary.empty:
            geometries = [
                geom
                for geom in boundary.geometry
                if geom is not None and not geom.is_empty
            ]
    if not geometries:
        geometries = read_polygon_geometries_from_shp(boundary_path)

    if not geometries:
        return None

    mask = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=sample["array"].shape,
        transform=sample["transform"],
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return mask.astype(bool)


def read_polygon_geometries_from_shp(shp_path: Path) -> list[dict]:
    """
    Read Polygon/PolygonM/PolygonZ records from a shapefile without fiona/geopandas.
    This fallback is intentionally minimal and is only used for study-area masking.
    """
    geometries: list[dict] = []
    if not shp_path.exists():
        return geometries

    data = shp_path.read_bytes()
    offset = 100
    while offset + 8 <= len(data):
        content_words = struct.unpack(">i", data[offset + 4 : offset + 8])[0]
        content_bytes = content_words * 2
        start = offset + 8
        end = start + content_bytes
        if end > len(data) or content_bytes < 4:
            break

        shape_type = struct.unpack("<i", data[start : start + 4])[0]
        if shape_type in {5, 15, 25} and content_bytes >= 44:
            pos = start + 36
            num_parts = struct.unpack("<i", data[pos : pos + 4])[0]
            num_points = struct.unpack("<i", data[pos + 4 : pos + 8])[0]
            pos += 8
            parts = list(struct.unpack(f"<{num_parts}i", data[pos : pos + 4 * num_parts]))
            pos += 4 * num_parts
            points = [
                struct.unpack("<2d", data[pos + i * 16 : pos + (i + 1) * 16])
                for i in range(num_points)
            ]

            rings = []
            for part_idx, part_start in enumerate(parts):
                part_end = parts[part_idx + 1] if part_idx + 1 < len(parts) else num_points
                ring = [(float(x), float(y)) for x, y in points[part_start:part_end]]
                if len(ring) >= 3:
                    if ring[0] != ring[-1]:
                        ring.append(ring[0])
                    rings.append(ring)
            if rings:
                geometries.append({"type": "Polygon", "coordinates": rings})

        offset = end

    return geometries


def add_scalebar_and_compass(
    ax: plt.Axes,
    arr: np.ndarray,
    transform,
    bar_length_km: float = 10.0,
    color: str = "#111",
    draw_compass: bool = True,
    font_scale: float = 1.0,
) -> None:
    """
    Add a scale bar and a compass-rose style north arrow inside the axis.
    bar_length_km uses the raster coordinate system units (meters for UTM).
    """
    left, right, bottom, top = compute_bounds(arr, transform)
    width = right - left
    height = top - bottom

    if draw_compass:
        # Minimal arrow-style north indicator (matching user-provided example).
        cx, cy = 0.058, 0.805
        compass_scale = min(max(font_scale * 1.08, 1.0), 1.85)
        arrow_h = 0.096 * compass_scale
        arrow_w = 0.032 * compass_scale
        shaft_h = 0.034 * compass_scale
        shaft_w = 0.0070 * compass_scale

        # Black outer arrow head.
        ax.add_patch(
            Polygon(
                [
                    (cx, cy + arrow_h * 0.50),
                    (cx - arrow_w * 0.50, cy - arrow_h * 0.12),
                    (cx + arrow_w * 0.50, cy - arrow_h * 0.12),
                ],
                closed=True,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                linewidth=0.9 * compass_scale,
                zorder=7,
            )
        )
        # White inner cut to form the slim, stylized pointer.
        ax.add_patch(
            Polygon(
                [
                    (cx, cy + arrow_h * 0.28),
                    (cx - arrow_w * 0.18, cy - arrow_h * 0.02),
                    (cx + arrow_w * 0.18, cy - arrow_h * 0.02),
                ],
                closed=True,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="white",
                linewidth=0.0,
                zorder=8,
            )
        )
        # Small black shaft under the arrow head.
        ax.add_patch(
            Polygon(
                [
                    (cx - shaft_w * 0.50, cy - arrow_h * 0.12),
                    (cx + shaft_w * 0.50, cy - arrow_h * 0.12),
                    (cx + shaft_w * 0.50, cy - arrow_h * 0.12 - shaft_h),
                    (cx - shaft_w * 0.50, cy - arrow_h * 0.12 - shaft_h),
                ],
                closed=True,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8 * compass_scale,
                zorder=7,
            )
    )
        ax.text(
            cx,
            cy + arrow_h * 0.62,
            "N",
            transform=ax.transAxes,
            fontsize=62 * font_scale,
            color=color,
            ha="center",
            va="bottom",
            fontweight="black",
            zorder=9,
        )

    # Scale bar near bottom-left (only when map units are meaningful).
    if transform is None:
        return
    bar_m = bar_length_km * 1000
    if bar_m <= 0 or bar_m >= width:
        return
    x0 = left + 0.05 * width
    y0 = bottom + 0.08 * height
    ax.plot(
        [x0, x0 + bar_m],
        [y0, y0],
        color=color,
        linewidth=3.2 * font_scale,
        solid_capstyle="butt",
    )
    ax.plot([x0, x0], [y0 - 0.012 * height, y0 + 0.012 * height], color=color, linewidth=1.8 * font_scale)
    ax.plot(
        [x0 + bar_m, x0 + bar_m],
        [y0 - 0.012 * height, y0 + 0.012 * height],
        color=color,
        linewidth=1.8 * font_scale,
    )
    ax.text(
        x0,
        y0 - 0.03 * height,
        "0",
        fontsize=21 * font_scale,
        color=color,
        va="top",
        fontweight="bold",
    )
    ax.text(
        x0 + bar_m,
        y0 - 0.03 * height,
        f"{int(bar_length_km)} km",
        fontsize=21 * font_scale,
        color=color,
        va="top",
        ha="right",
        fontweight="bold",
    )


def style_panel_frame(ax: plt.Axes, color: str = "#d4d4d4", linewidth: float = 1.0) -> None:
    """
    Give each panel a clean light-gray frame so the 24 maps read as one block.
    """
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(color)
        ax.spines[side].set_linewidth(linewidth)


def normalize_hours(hours: Sequence[int]) -> list[int]:
    """Return unique hours in input order after range validation."""
    normalized: list[int] = []
    seen: set[int] = set()
    for hour in hours:
        hour_int = int(hour)
        if hour_int < 0 or hour_int > 23:
            raise ValueError(f"Invalid hour {hour_int}; expected values in 0..23.")
        if hour_int not in seen:
            normalized.append(hour_int)
            seen.add(hour_int)
    if not normalized:
        raise ValueError("At least one hour must be provided.")
    return normalized


def collect_hourly_tifs(pop_dir: Path, date: str, hours: Sequence[int]) -> list[Path]:
    """Collect exact prediction rasters for the requested hours."""
    selected: list[Path] = []
    missing: list[str] = []
    for hour in hours:
        candidates = [
            pop_dir / f"PredPop_{date}_{hour:02d}_real.tif",
            pop_dir / f"clipped_PredPop_{date}_{hour:02d}_real.tif",
        ]
        match = next((path for path in candidates if path.exists()), None)
        if match is None:
            missing.append(f"{hour:02d}")
            continue
        selected.append(match)
    if missing:
        raise FileNotFoundError(
            f"Missing prediction rasters for date={date}, hours={', '.join(missing)} under: {pop_dir}"
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hourly prediction rasters in a compact grid.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing hourly *_real.tif files, or a root folder containing Pred_YYYYMMDD subdirectories.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=DEFAULT_DATE,
        help="Date string in YYYYMMDD format.",
    )
    parser.add_argument(
        "--hours",
        nargs="+",
        type=int,
        default=list(DEFAULT_HOURS),
        help="Hours to plot. Default: 0 1 2 ... 23.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=0,
        help="Number of columns in the output grid. Default: auto, using 4 columns for all 24 hours.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path.",
    )
    parser.add_argument("--dpi", type=int, default=240, help="Output DPI for TIFF/raster exports.")
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Additional multiplier for all figure text.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=14.0,
        help="Figure width in inches. Default is memory-safe for the 4x6 layout.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=21.0,
        help="Figure height in inches. Default is memory-safe for the 4x6 layout.",
    )
    parser.add_argument(
        "--panel-title-size",
        type=float,
        default=17.0,
        help="Panel title font size in points before --font-scale is applied.",
    )
    parser.add_argument(
        "--colorbar-label-size",
        type=float,
        default=17.5,
        help="Colorbar label font size in points before --font-scale is applied.",
    )
    parser.add_argument(
        "--colorbar-tick-size",
        type=float,
        default=15.5,
        help="Colorbar tick-label font size in points before --font-scale is applied.",
    )
    parser.add_argument(
        "--map-element-scale",
        type=float,
        default=0.32,
        help="Compass/scale-bar size multiplier before --font-scale is applied.",
    )
    parser.add_argument(
        "--grid-wspace",
        type=float,
        default=0.020,
        help="Horizontal space between panels.",
    )
    parser.add_argument(
        "--grid-hspace",
        type=float,
        default=0.240,
        help="Vertical space between panels.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    selected_hours = normalize_hours(args.hours)
    pop_dir = resolve_input_dir(args.input_dir, args.date)

    tif_paths = collect_hourly_tifs(pop_dir, args.date, selected_hours)
    if not RASTERIO_AVAILABLE:
        print(
            "[warn] rasterio unavailable, using tifffile fallback. "
            f"Reason: {RASTERIO_IMPORT_ERROR}"
        )
        print(
            "[warn] Georeference-aware overlays may be disabled "
            "(boundary/true map-unit scalebar)."
        )

    samples = []
    for path in tif_paths:
        sample = read_raster(path)
        sample["hour"] = int(path.stem.split("_")[-2])
        samples.append(sample)

    # Keep the study area in the map extent. Pixels outside the study area stay blank,
    # while zero values inside the study area are rendered with the leftmost color.
    study_mask = rasterize_boundary_mask(samples[0], DEFAULT_BOUNDARY)
    if study_mask is None:
        raise RuntimeError(
            "Failed to rasterize the study-area boundary. "
            f"Check boundary file: {DEFAULT_BOUNDARY}"
        )
    for s in samples:
        arr = np.ma.filled(s["array"], np.nan).astype(np.float32, copy=False)
        arr[study_mask & ~np.isfinite(arr)] = 0.0
        arr[arr < 0] = 0.0
        arr[~study_mask] = np.nan
        s["array"] = np.ma.masked_invalid(arr)

    all_vals = np.hstack([s["array"].compressed() for s in samples])
    positive_vals = all_vals[all_vals > 0]
    if positive_vals.size == 0:
        positive_vals = np.asarray([1.0], dtype=np.float32)
    vmin = 0.0
    vmax = float(np.quantile(positive_vals, 0.995))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = build_cmap()
    cmap.set_bad(color="white")

    n_panels = len(samples)
    if int(args.columns) > 0:
        ncols = max(1, min(int(args.columns), n_panels))
    elif n_panels == 24:
        ncols = 4
    elif n_panels <= 4:
        ncols = min(2, n_panels)
    else:
        ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    title_text: Optional[str] = None
    frame_top_margin = 0.992 if title_text else 0.980
    panel_top_margin = 0.936 if title_text else 0.936
    left_margin = 0.030
    right_margin = 0.985
    if n_panels == 24 and ncols == 4:
        bottom_margin = 0.105
        fig_width = float(args.fig_width)
        fig_height = float(args.fig_height)
        cbar_box = [0.08, 0.055, 0.84, 0.018]
    else:
        bottom_margin = 0.215
        fig_width = max(float(args.fig_width), 3.2 * ncols)
        fig_height = max(float(args.fig_height), 3.2 * nrows + 2.0)
        cbar_box = [0.14, 0.112, 0.72, 0.036]
    font_scale = float(args.font_scale)
    # Enlarge the full canvas first, then open more spacing around and between panels.
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=args.dpi, constrained_layout=False)
    grid_hspace = float(args.grid_hspace) if n_panels == 24 and ncols == 4 else 0.180
    grid_wspace = float(args.grid_wspace) if n_panels == 24 and ncols == 4 else 0.042
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        hspace=grid_hspace,
        wspace=grid_wspace,
    )

    boundary = None
    if samples[0]["transform"] is not None:
        boundary = load_boundary(DEFAULT_BOUNDARY, samples[0]["crs"])

    for idx, sample in enumerate(samples):
        row = idx // ncols
        col = idx % ncols

        ax = fig.add_subplot(gs[row, col])
        draw_raster(
            ax=ax,
            arr=sample["array"],
            transform=sample["transform"],
            cmap=cmap,
            norm=norm,
        )
        if boundary is not None:
            try:
                boundary.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="#8a2040",
                    linewidth=1.0,
                )
            except Exception as exc:  # pragma: no cover - visual fallback
                print(f"Skip boundary overlay: {exc}")
        add_scalebar_and_compass(
            ax,
            sample["array"],
            sample["transform"],
            draw_compass=True,
            font_scale=float(args.map_element_scale) * font_scale,
        )
        ax.text(
            0.018,
            1.058,
            f"({PANEL_LETTERS[idx]}) Hour {sample['hour']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=float(args.panel_title_size) * font_scale,
            fontweight="bold",
            fontfamily="Times New Roman",
            color="#111",
            zorder=12,
            clip_on=False,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        style_panel_frame(ax)

    # Manual horizontal colorbar. Zero is included in the continuous gradient.
    cax = fig.add_axes(cbar_box)
    cax.set_facecolor(cmap(0.0))
    # Draw the colorbar as a dense strip of rectangles instead of imshow.
    # This avoids the right-edge blank band seen in some SVG renderers.
    n_segments = 1024
    seg_w = 1.0 / n_segments
    overlap = seg_w * 0.08
    for i in range(n_segments):
        x0 = i * seg_w
        color = cmap(i / (n_segments - 1))
        cax.add_patch(
            Rectangle(
                (x0 - overlap, 0.0),
                seg_w + overlap * 2.0,
                1.0,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                antialiased=False,
            )
        )
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(0.0, 1.0)
    cax.margins(x=0)
    cax.set_yticks([])
    real_ticks = np.linspace(vmin, vmax, 6)
    cax.set_xticks(np.linspace(0.0, 1.0, 6))
    cax.set_xticklabels([f"{t:.0f}" for t in real_ticks])
    cax.set_xlabel(
        "Estimated population (People / 500 m grid)",
        fontsize=float(args.colorbar_label_size) * font_scale,
        fontweight="bold",
        fontfamily="Times New Roman",
        labelpad=4 * font_scale,
    )
    cax.tick_params(
        labelsize=float(args.colorbar_tick_size) * font_scale,
        length=3.8 * font_scale,
        width=0.9,
        pad=3.0 * font_scale,
    )
    for lbl in cax.get_xticklabels():
        lbl.set_fontsize(float(args.colorbar_tick_size) * font_scale)
        lbl.set_fontweight("bold")
        lbl.set_fontfamily("Times New Roman")

    if title_text:
        fig.suptitle(
            title_text,
            y=0.992,
            fontsize=30,
            fontweight="bold",
        )

    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=panel_top_margin,
        bottom=bottom_margin,
        wspace=grid_wspace,
        hspace=grid_hspace,
    )

    # Add one clean outer frame so the full 24-panel layout reads as a single figure block.
    outer_frame = Rectangle(
        (left_margin, bottom_margin),
        right_margin - left_margin,
        frame_top_margin - bottom_margin,
        transform=fig.transFigure,
        facecolor="none",
        edgecolor="#d4d4d4",
        linewidth=1.4,
        zorder=20,
    )
    fig.add_artist(outer_frame)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".tif", ".tiff"}:
        # Matplotlib writes RGBA TIFFs, which are unnecessarily large for this figure.
        # Save once, then convert to RGB with lossless Deflate compression.
        temp_path = out_path.with_name(f"{out_path.stem}__tmp_rgba{out_path.suffix}")
        # Avoid LZW compression during the initial Matplotlib render: PIL may
        # duplicate the full RGBA canvas internally and exhaust memory for
        # dense 24-panel figures. Render a plain temporary TIFF first, then
        # convert/compress it below.
        fig.savefig(temp_path, dpi=args.dpi, format="tiff")
        try:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            with Image.open(temp_path) as image:
                image.convert("RGB").save(
                    out_path,
                    format="TIFF",
                    dpi=image.info.get("dpi", (args.dpi, args.dpi)),
                    compression="tiff_adobe_deflate",
                )
        except MemoryError:
            temp_path.replace(out_path)
            print(
                "[warn] RGB conversion ran out of memory; "
                "kept the high-resolution RGBA TIFF instead."
            )
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except PermissionError:
                print(f"[warn] Temporary TIFF is still locked and was kept: {temp_path}")
    else:
        fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
