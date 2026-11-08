from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import subprocess
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "output_results").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root(PROJECT_ROOT)
DEFAULT_BASE = REPO_ROOT / "output_results" / "dynamic_population_distribution"
DEFAULT_TRUTH_DIR = REPO_ROOT / "data" / "mobile_phone_data"
DEFAULT_DATE = "20170925"
PROJECT_PRESET_CHOICES = [
    "project_pop_hstm_20170925",
    "project_baselines_20170925",
    "project_osit_low_20170925",
    "project_osit_high_20170925",
]
PROJECT_BASELINE_PATTERNS = {
    "HA": "baselines/HA/Pred_{date}",
    "LSTM": "baselines/LSTM/**/Pred_{date}",
    "CNN": "baselines/CNN/Pred_{date}",
    "CNN-LSTM": "baselines/CNN-LSTM/**/Pred_{date}",
    "ConvLSTM": "baselines/ConvLSTM/**/Pred_{date}",
    "STAEformer": "baselines/STAEformer/**/Pred_{date}",
    "PredRNN": "baselines/PredRNN/**/Pred_{date}",
    "Pop-HSTM": "Pop-HSTM/Pred_{date}",
}


def _configure_proj_env() -> None:
    """
    Prefer PROJ data shipped with the active Python/Conda environment.
    This avoids picking an incompatible proj.db from other installations.
    """
    exe_dir = Path(sys.executable).resolve().parent
    proj_candidates = [
        exe_dir.parent / "Library" / "share" / "proj",  # conda on Windows
        exe_dir.parent / "share" / "proj",
    ]
    for candidate in proj_candidates:
        if candidate.exists():
            os.environ["PROJ_LIB"] = str(candidate)
            os.environ["PROJ_DATA"] = str(candidate)
            break
    # Use EPSG parameters when GeoTIFF keys conflict with registry definitions.
    os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")


_configure_proj_env()
_GDAL_ENV = os.environ.copy()

try:
    from osgeo import gdal
except Exception:
    gdal = None

try:
    import rasterio
except Exception:
    rasterio = None

try:
    import fiona
except Exception:
    fiona = None

from PIL import Image

try:
    from sklearn.metrics import r2_score as _r2_score
    from sklearn.metrics import mean_absolute_error as _mae
    from sklearn.metrics import mean_squared_error as _mse
except Exception:
    _r2_score = _mae = _mse = None

EPSILON = 1e-5


def tem(pred: np.ndarray, gold: np.ndarray) -> float:
    """TEM used in this project, numerically equal to ln(1 + MSE)."""
    if _mse is not None:
        mse = float(_mse(gold, pred))
    else:
        diff = pred - gold
        mse = float(np.mean(diff ** 2))
    return float(np.log1p(mse))


def mae(pred: np.ndarray, gold: np.ndarray) -> float:
    """Mean Absolute Error on raw values."""
    if _mae is not None:
        return float(_mae(gold, pred))
    return float(np.mean(np.abs(pred - gold)))


def mape(pred: np.ndarray, gold: np.ndarray, eps: float = EPSILON) -> float:
    """Mean Absolute Percentage Error (in percent)."""
    out = np.abs((pred - gold) / (gold + eps))
    return float(out.mean() * 100.0)


def r2(pred: np.ndarray, gold: np.ndarray) -> float:
    """R^2 on raw values."""
    if _r2_score is not None:
        return float(_r2_score(gold, pred))
    gold_mean = float(np.mean(gold))
    ss_tot = float(np.sum((gold - gold_mean) ** 2))
    ss_res = float(np.sum((gold - pred) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")


def load_tif(path: Path) -> np.ndarray:
    """Load a single-band tif as float64 ndarray."""
    # First try PIL (quiet and robust for the single-band tif files used here).
    try:
        with Image.open(path) as img:
            return np.array(img, dtype=np.float64)
    except Exception:
        pass

    # Then try rasterio.
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float64)
            nodata = src.nodata
        if nodata is not None:
            data = np.where(np.isclose(data, nodata), np.nan, data)
        return data
    except Exception:
        pass

    # Fallback to GDAL CLI for tif variants that PIL/rasterio fail to read.
    try:
        with tempfile.NamedTemporaryFile(suffix=".asc", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        cmd = ["gdal_translate", "-quiet", "-of", "AAIGrid", str(path), str(tmp_path)]
        subprocess.run(cmd, check=True, env=_GDAL_ENV)
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
        data = np.loadtxt(tmp_path, skiprows=header_lines, dtype=np.float64)
        tmp_path.unlink(missing_ok=True)
        return data
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    # Final fallback: re-raise as a clear file-level read error.
    raise RuntimeError(f"Failed to read tif: {path}")


def load_mask_from_tif(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Load a mask tif and ensure shape matches expected."""
    arr = load_tif(path)
    if arr.shape != expected_shape:
        raise ValueError(f"Mask tif shape {arr.shape} != expected {expected_shape}")
    return arr != 0


def rasterize_mask_from_shp(shp_path: Path, ref_tif: Path) -> np.ndarray:
    """Rasterize shapefile to match reference tif (requires rasterio+fiona)."""
    try:
        import rasterio
        from rasterio.features import rasterize
        import fiona
    except Exception:
        rasterio = None
        fiona = None

    if rasterio is not None and fiona is not None:
        try:
            with rasterio.open(ref_tif) as src:
                transform = src.transform
                shape = (src.height, src.width)
                crs = src.crs
            with fiona.open(shp_path, "r") as shp:
                if crs and shp.crs and shp.crs != crs:
                    print(
                        f"[warn] Shapefile CRS {shp.crs} != raster CRS {crs}, proceeding without reprojection."
                    )
                shapes = [(feat["geometry"], 1) for feat in shp]
            mask = rasterize(shapes=shapes, out_shape=shape, transform=transform, fill=0, all_touched=True)
            return mask != 0
        except Exception:
            pass

    # Fallback: use gdal_rasterize + gdal_translate (CLI)
    def _read_aaigrid_header(src: Path) -> dict[str, float]:
        with tempfile.NamedTemporaryFile(suffix=".asc", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        cmd = ["gdal_translate", "-quiet", "-of", "AAIGrid", str(src), str(tmp_path)]
        subprocess.run(cmd, check=True, env=_GDAL_ENV)
        header: dict[str, float] = {}
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
        tmp_path.unlink(missing_ok=True)
        return header

    header = _read_aaigrid_header(ref_tif)
    ncols = int(header.get("ncols", 0))
    nrows = int(header.get("nrows", 0))
    cell = float(header.get("cellsize", 0.0))
    if ncols <= 0 or nrows <= 0 or cell <= 0:
        raise RuntimeError("Failed to read grid geometry for mask rasterization.")
    xll = float(header.get("xllcorner", header.get("xllcenter", 0.0)))
    yll = float(header.get("yllcorner", header.get("yllcenter", 0.0)))
    xur = xll + cell * ncols
    yur = yll + cell * nrows

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        out_path = Path(tmp.name)
    cmd = [
        "gdal_rasterize",
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
    subprocess.run(cmd, check=True, env=_GDAL_ENV)
    arr = load_tif(out_path)
    out_path.unlink(missing_ok=True)
    return arr != 0


def stack_hourly_arrays(
    pred_dir: Path,
    truth_dir: Path,
    date: str,
    hours: Iterable[int],
    pred_tpl: str = "PredPop_{date}_{hour:02d}_real.tif",
    truth_tpl: str = "point{date_key}{hour:02d}_500.tif",
    date_key: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate all hours into one long vector for metrics."""
    if date_key is None:
        date_key = date
    preds, trues = [], []
    for h in hours:
        preds.append(load_tif(pred_dir / pred_tpl.format(date=date, hour=h)).ravel())
        trues.append(
            load_tif(truth_dir / truth_tpl.format(date=date, date_key=date_key, hour=h)).ravel()
        )
    return np.concatenate(preds), np.concatenate(trues)


def load_hourly_vectors(
    pred_dir: Path,
    truth_dir: Path,
    date: str,
    hours: Iterable[int],
    pred_tpl: str = "PredPop_{date}_{hour:02d}_real.tif",
    truth_tpl: str = "point{date_key}{hour:02d}_500.tif",
    date_key: str = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load per-hour flattened arrays (pred, truth)."""
    if date_key is None:
        date_key = date
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for h in hours:
        pred = load_tif(pred_dir / pred_tpl.format(date=date, hour=h)).ravel()
        truth = load_tif(truth_dir / truth_tpl.format(date=date, date_key=date_key, hour=h)).ravel()
        pairs.append((pred, truth))
    return pairs


def compute_all_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    tem_value = tem(pred, truth)
    return {
        "r2": r2(pred, truth),
        "tem": tem_value,
        "mae": mae(pred, truth),
        "mape": mape(pred, truth),
    }


def apply_filters(
    pred: np.ndarray,
    truth: np.ndarray,
    min_truth: float = None,
    clip_quantile: Tuple[float, float] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Apply min_truth mask and optional quantile clipping mask (based on truth)."""
    info: List[str] = []
    if min_truth is not None:
        mask = truth >= min_truth
        pred = pred[mask]
        truth = truth[mask]
        info.append(f"min_truth>={min_truth}, kept {mask.sum()}")
    if clip_quantile is not None:
        q_low, q_high = clip_quantile
        ql, qh = np.quantile(truth, [q_low, q_high])
        mask = (truth >= ql) & (truth <= qh)
        pred = pred[mask]
        truth = truth[mask]
        info.append(f"clip[{q_low},{q_high}] -> [{ql:.3f},{qh:.3f}], kept {mask.sum()}")
    return pred, truth, "; ".join(info)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def resolve_run_path(base: Path, rel: str) -> Path:
    rel_path = Path(rel)
    if rel_path.is_absolute():
        return rel_path

    rel_str = rel.replace("\\", "/")
    if any(token in rel_str for token in ("*", "?", "[")):
        matches = sorted(base.glob(rel_str))
        if not matches:
            raise FileNotFoundError(
                f"No prediction directory matched pattern: {rel_str} under {base}"
            )
        return matches[0]
    return base / rel_path


def build_project_preset_runs(base: Path, preset: str, date: str) -> Dict[str, Path]:
    if preset == "project_pop_hstm_20170925":
        return {"Pop-HSTM": base / "Pop-HSTM" / f"Pred_{date}"}
    if preset == "project_osit_low_20170925":
        return {
            "OSIT-low-density": (
                base
                / "Spatial_Temporal_Generalization"
                / "low_density_ring_generalization"
                / f"Pred_{date}"
            )
        }
    if preset == "project_osit_high_20170925":
        return {
            "OSIT-high-density": (
                base
                / "Spatial_Temporal_Generalization"
                / "high_density_ring_generalization"
                / f"Pred_{date}"
            )
        }
    if preset == "project_baselines_20170925":
        runs: Dict[str, Path] = {}
        for name, pattern in PROJECT_BASELINE_PATTERNS.items():
            runs[name] = resolve_run_path(base, pattern.format(date=date))
        return runs
    raise ValueError(f"Unsupported preset: {preset}")


def cli():
    parser = argparse.ArgumentParser(
        description="Compute R2/TEM/MAE/MAPE for hourly population predictions."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Base folder that contains current-project prediction outputs.",
    )
    parser.add_argument(
        "--preset",
        choices=[],
        default=None,
        help=(
            "No legacy presets are shipped for the current project. "
            "Use --pred-runs or project_cli presets."
        ),
    )
    parser.add_argument(
        "--date",
        type=str,
        default="20170925",
        help="Date string used in filenames, e.g., 20170925.",
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=DEFAULT_TRUTH_DIR,
        help="Folder with ground truth tif files (point{date}{hh}_500.tif).",
    )
    parser.add_argument(
        "--truth-date-key",
        type=str,
        default=None,
        help=(
            "String used in ground-truth filenames in place of {date_key}. "
            "If not set, defaults to the last 4 chars of --date (to match pointMMDDhh_500.tif)."
        ),
    )
    parser.add_argument(
        "--truth-tpl",
        type=str,
        default="point{date_key}{hour:02d}_500.tif",
        help="Template for ground-truth filenames.",
    )
    parser.add_argument(
        "--pred-tpl",
        type=str,
        default="PredPop_{date}_{hour:02d}_real.tif",
        help="Template for prediction filenames.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        nargs="+",
        default=list(range(24)),
        help="Hours to evaluate, default 0-23.",
    )
    parser.add_argument(
        "--min-truth",
        type=float,
        default=None,
        help="If set, only pixels with ground truth >= this value are used for metrics.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="If set, drop pixels outside truth quantiles [LOW, HIGH], e.g., 0.01 0.99.",
    )
    parser.add_argument(
        "--per-hour",
        action="store_true",
        help="Also report per-hour metrics (after the same filters).",
    )
    parser.add_argument(
        "--mask-shp",
        type=Path,
        default=None,
        help="Optional shapefile to rasterize as mask; requires rasterio+fiona and assumes same CRS as truth.",
    )
    parser.add_argument(
        "--mask-ring-inner-shp",
        type=Path,
        default=None,
        help="Inner ring shapefile to exclude (e.g., 五环). Used with --mask-ring-outer-shp.",
    )
    parser.add_argument(
        "--mask-ring-outer-shp",
        type=Path,
        default=None,
        help="Outer ring shapefile to include (e.g., 六环). Used with --mask-ring-inner-shp.",
    )
    parser.add_argument(
        "--mask-tif",
        type=Path,
        default=None,
        help="Optional mask tif aligned with truth; nonzero pixels are kept for metrics.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Save summary metrics (run,R2,TEM,MAE,MAPE) to CSV.",
    )
    parser.add_argument(
        "--per-hour-csv",
        type=Path,
        default=None,
        help="Save per-hour metrics (run,hour,R2,TEM,MAE,MAPE) to CSV when --per-hour is set.",
    )
    parser.add_argument(
        "--pred-runs",
        nargs="*",
        default=[f"Pop-HSTM=Pop-HSTM/Pred_{DEFAULT_DATE}"],
        help=(
            "List of name=relative_path pairs for prediction folders, relative to --base."
        ),
    )
    args = parser.parse_args()

    runs: Dict[str, Path] = {}
    for item in args.pred_runs:
        if "=" not in item:
            parser.error(f"Invalid --pred-runs item: {item}")
        name, rel = item.split("=", 1)
        runs[name] = args.base / rel

    # Default: use last 4 digits (MMDD) to match pointMMDDhh_500.tif naming.
    date_key = args.truth_date_key if args.truth_date_key is not None else args.date[-4:]

    # Build mask if requested (using first truth file as reference)
    mask_array = None
    mask_flat = None
    if args.mask_shp or args.mask_tif or args.mask_ring_inner_shp or args.mask_ring_outer_shp:
        first_hour = args.hours[0]
        ref_truth = args.truth_dir / args.truth_tpl.format(date=args.date, date_key=date_key, hour=first_hour)
        ref_arr = load_tif(ref_truth)
        # Ring mask: outer AND NOT inner
        if args.mask_ring_inner_shp or args.mask_ring_outer_shp:
            if not (args.mask_ring_inner_shp and args.mask_ring_outer_shp):
                parser.error("Both --mask-ring-inner-shp and --mask-ring-outer-shp are required.")
            outer_mask = rasterize_mask_from_shp(args.mask_ring_outer_shp, ref_truth)
            inner_mask = rasterize_mask_from_shp(args.mask_ring_inner_shp, ref_truth)
            ring_mask = outer_mask & (~inner_mask)
            mask_array = ring_mask
        # Optional additional mask (AND)
        if args.mask_tif:
            extra = load_mask_from_tif(args.mask_tif, expected_shape=ref_arr.shape)
            mask_array = extra if mask_array is None else (mask_array & extra)
        if args.mask_shp:
            extra = rasterize_mask_from_shp(args.mask_shp, ref_truth)
            mask_array = extra if mask_array is None else (mask_array & extra)
        if mask_array is not None:
            mask_flat = mask_array.ravel().astype(bool)

    summary_rows: list[dict] = []
    per_hour_rows: list[dict] = []

    for name, pred_dir in runs.items():
        hourly = load_hourly_vectors(
            pred_dir=pred_dir,
            truth_dir=args.truth_dir,
            date=args.date,
            hours=args.hours,
            pred_tpl=args.pred_tpl,
            truth_tpl=args.truth_tpl,
            date_key=date_key,
        )
        # apply mask per hour if provided
        if mask_flat is not None:
            hourly = [(p[mask_flat], t[mask_flat]) for (p, t) in hourly]
        pred = np.concatenate([p for p, _ in hourly])
        truth = np.concatenate([t for _, t in hourly])
        pred, truth, info = apply_filters(
            pred=pred,
            truth=truth,
            min_truth=args.min_truth,
            clip_quantile=tuple(args.clip_quantile) if args.clip_quantile else None,
        )
        if info:
            print(f"{name}: {info}")
        metrics = compute_all_metrics(pred, truth)
        print(name)
        for k, v in metrics.items():
            print(f"  {k.upper():4s} {v}")
        summary_rows.append({"run": name, **metrics})
        if args.per_hour:
            print("  Per-hour:")
            for h, (p_h, t_h) in zip(args.hours, hourly):
                p_f, t_f, info_h = apply_filters(
                    pred=p_h,
                    truth=t_h,
                    min_truth=args.min_truth,
                    clip_quantile=tuple(args.clip_quantile) if args.clip_quantile else None,
                )
                if p_f.size == 0:
                    print(f"    h={h:02d}: no pixels after filtering")
                    continue
                m_h = compute_all_metrics(p_f, t_f)
                per_hour_rows.append({"run": name, "hour": h, **m_h})
                extra = f" ({info_h})" if info_h else ""
                print(
                    f"    h={h:02d}: R2={m_h['r2']:.4f}, TEM={m_h['tem']:.3f}, "
                    f"MAE={m_h['mae']:.3f}, MAPE={m_h['mape']:.3f}{extra}"
                )

    if args.output_csv and summary_rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run", "r2", "TEM", "mae", "mape"]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(
                    {
                        "run": row.get("run"),
                        "r2": row.get("r2"),
                        "TEM": row.get("tem"),
                        "mae": row.get("mae"),
                        "mape": row.get("mape"),
                    }
                )
        print(f"[info] Saved summary metrics to {args.output_csv}")

    if args.per_hour and args.per_hour_csv and per_hour_rows:
        args.per_hour_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run", "hour", "r2", "TEM", "mae", "mape"]
        with open(args.per_hour_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_hour_rows:
                writer.writerow(
                    {
                        "run": row.get("run"),
                        "r2": row.get("r2"),
                        "TEM": row.get("tem"),
                        "mae": row.get("mae"),
                        "mape": row.get("mape"),
                    }
                )
        print(f"[info] Saved per-hour metrics to {args.per_hour_csv}")


def project_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Compute R2/TEM/MAE/MAPE for current-project hourly population predictions."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help="Base folder that contains current-project prediction outputs.",
    )
    parser.add_argument(
        "--preset",
        choices=PROJECT_PRESET_CHOICES,
        default=None,
        help=(
            "Quick project presets. Overrides --base/--date/--pred-runs with "
            "current-project output folders."
        ),
    )
    parser.add_argument(
        "--date",
        type=str,
        default=DEFAULT_DATE,
        help="Date string used in filenames, e.g., 20170925.",
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=DEFAULT_TRUTH_DIR,
        help="Folder with ground-truth tif files (point{date_key}{hh}_500.tif).",
    )
    parser.add_argument(
        "--truth-date-key",
        type=str,
        default=None,
        help=(
            "String used in ground-truth filenames in place of {date_key}. "
            "If not set, defaults to the last 4 chars of --date."
        ),
    )
    parser.add_argument(
        "--truth-tpl",
        type=str,
        default="point{date_key}{hour:02d}_500.tif",
        help="Template for ground-truth filenames.",
    )
    parser.add_argument(
        "--pred-tpl",
        type=str,
        default="PredPop_{date}_{hour:02d}_real.tif",
        help="Template for prediction filenames.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        nargs="+",
        default=list(range(24)),
        help="Hours to evaluate, default 0-23.",
    )
    parser.add_argument(
        "--min-truth",
        type=float,
        default=None,
        help="If set, only pixels with ground truth >= this value are used for metrics.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="If set, drop pixels outside truth quantiles [LOW, HIGH], e.g. 0.01 0.99.",
    )
    parser.add_argument(
        "--per-hour",
        action="store_true",
        help="Also report per-hour metrics after the same filters.",
    )
    parser.add_argument(
        "--mask-shp",
        type=Path,
        default=None,
        help="Optional shapefile mask; requires rasterio+fiona.",
    )
    parser.add_argument(
        "--mask-ring-inner-shp",
        type=Path,
        default=None,
        help="Inner-region shapefile to exclude. Used with --mask-ring-outer-shp.",
    )
    parser.add_argument(
        "--mask-ring-outer-shp",
        type=Path,
        default=None,
        help="Outer-region shapefile to include. Used with --mask-ring-inner-shp.",
    )
    parser.add_argument(
        "--mask-tif",
        type=Path,
        default=None,
        help="Optional mask tif aligned with truth; nonzero pixels are kept.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Save summary metrics (run,R2,TEM,MAE,MAPE) to CSV.",
    )
    parser.add_argument(
        "--per-hour-csv",
        type=Path,
        default=None,
        help="Save per-hour metrics (run,hour,R2,TEM,MAE,MAPE) to CSV.",
    )
    parser.add_argument(
        "--pred-runs",
        nargs="*",
        default=[f"Pop-HSTM=Pop-HSTM/Pred_{DEFAULT_DATE}"],
        help=(
            "List of name=relative_path pairs for prediction folders, relative to --base. "
            "Glob patterns such as baselines/LSTM/**/Pred_20170925 are supported."
        ),
    )
    args = parser.parse_args()

    args.base = resolve_path(args.base)
    args.truth_dir = resolve_path(args.truth_dir)
    if args.mask_shp is not None:
        args.mask_shp = resolve_path(args.mask_shp)
    if args.mask_tif is not None:
        args.mask_tif = resolve_path(args.mask_tif)
    if args.mask_ring_inner_shp is not None:
        args.mask_ring_inner_shp = resolve_path(args.mask_ring_inner_shp)
    if args.mask_ring_outer_shp is not None:
        args.mask_ring_outer_shp = resolve_path(args.mask_ring_outer_shp)
    if args.output_csv is not None:
        args.output_csv = resolve_path(args.output_csv)
    if args.per_hour_csv is not None:
        args.per_hour_csv = resolve_path(args.per_hour_csv)

    runs: Dict[str, Path] = {}
    for item in args.pred_runs:
        if "=" not in item:
            parser.error(f"Invalid --pred-runs item: {item}")
        name, rel = item.split("=", 1)
        runs[name] = resolve_run_path(args.base, rel)

    if args.preset is not None:
        args.base = DEFAULT_BASE
        args.date = DEFAULT_DATE
        runs = build_project_preset_runs(args.base, args.preset, args.date)

    date_key = args.truth_date_key if args.truth_date_key is not None else args.date[-4:]

    mask_array = None
    mask_flat = None
    if args.mask_shp or args.mask_tif or args.mask_ring_inner_shp or args.mask_ring_outer_shp:
        first_hour = args.hours[0]
        ref_truth = args.truth_dir / args.truth_tpl.format(
            date=args.date,
            date_key=date_key,
            hour=first_hour,
        )
        ref_arr = load_tif(ref_truth)
        if args.mask_ring_inner_shp or args.mask_ring_outer_shp:
            if not (args.mask_ring_inner_shp and args.mask_ring_outer_shp):
                parser.error("Both --mask-ring-inner-shp and --mask-ring-outer-shp are required.")
            outer_mask = rasterize_mask_from_shp(args.mask_ring_outer_shp, ref_truth)
            inner_mask = rasterize_mask_from_shp(args.mask_ring_inner_shp, ref_truth)
            mask_array = outer_mask & (~inner_mask)
        if args.mask_tif:
            extra = load_mask_from_tif(args.mask_tif, expected_shape=ref_arr.shape)
            mask_array = extra if mask_array is None else (mask_array & extra)
        if args.mask_shp:
            extra = rasterize_mask_from_shp(args.mask_shp, ref_truth)
            mask_array = extra if mask_array is None else (mask_array & extra)
        if mask_array is not None:
            mask_flat = mask_array.ravel().astype(bool)

    summary_rows: list[dict] = []
    per_hour_rows: list[dict] = []

    for name, pred_dir in runs.items():
        hourly = load_hourly_vectors(
            pred_dir=pred_dir,
            truth_dir=args.truth_dir,
            date=args.date,
            hours=args.hours,
            pred_tpl=args.pred_tpl,
            truth_tpl=args.truth_tpl,
            date_key=date_key,
        )
        if mask_flat is not None:
            hourly = [(p[mask_flat], t[mask_flat]) for (p, t) in hourly]
        pred = np.concatenate([p for p, _ in hourly])
        truth = np.concatenate([t for _, t in hourly])
        pred, truth, info = apply_filters(
            pred=pred,
            truth=truth,
            min_truth=args.min_truth,
            clip_quantile=tuple(args.clip_quantile) if args.clip_quantile else None,
        )
        if info:
            print(f"{name}: {info}")
        metrics = compute_all_metrics(pred, truth)
        print(name)
        print(f"  R2   {metrics['r2']}")
        print(f"  TEM  {metrics['tem']}")
        print(f"  MAE  {metrics['mae']}")
        print(f"  MAPE {metrics['mape']}")
        summary_rows.append({"run": name, **metrics})
        if args.per_hour:
            print("  Per-hour:")
            for h, (p_h, t_h) in zip(args.hours, hourly):
                p_f, t_f, info_h = apply_filters(
                    pred=p_h,
                    truth=t_h,
                    min_truth=args.min_truth,
                    clip_quantile=tuple(args.clip_quantile) if args.clip_quantile else None,
                )
                if p_f.size == 0:
                    print(f"    h={h:02d}: no pixels after filtering")
                    continue
                m_h = compute_all_metrics(p_f, t_f)
                per_hour_rows.append({"run": name, "hour": h, **m_h})
                extra = f" ({info_h})" if info_h else ""
                print(
                    f"    h={h:02d}: R2={m_h['r2']:.4f}, TEM={m_h['tem']:.3f}, "
                    f"MAE={m_h['mae']:.3f}, MAPE={m_h['mape']:.3f}{extra}"
                )

    if args.output_csv and summary_rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run", "r2", "TEM", "mae", "mape"]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(
                    {
                        "run": row.get("run"),
                        "r2": row.get("r2"),
                        "TEM": row.get("tem"),
                        "mae": row.get("mae"),
                        "mape": row.get("mape"),
                    }
                )
        print(f"[info] Saved summary metrics to {args.output_csv}")

    if args.per_hour and args.per_hour_csv and per_hour_rows:
        args.per_hour_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["run", "hour", "r2", "TEM", "mae", "mape"]
        with open(args.per_hour_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_hour_rows:
                writer.writerow(
                    {
                        "run": row.get("run"),
                        "hour": row.get("hour"),
                        "r2": row.get("r2"),
                        "TEM": row.get("tem"),
                        "mae": row.get("mae"),
                        "mape": row.get("mape"),
                    }
                )
        print(f"[info] Saved per-hour metrics to {args.per_hour_csv}")


cli = project_cli


if __name__ == "__main__":
    project_cli()


__all__ = [
    "tem",
    "mae",
    "mape",
    "r2",
    "compute_all_metrics",
    "stack_hourly_arrays",
    "load_tif",
    "load_hourly_vectors",
    "apply_filters",
]
