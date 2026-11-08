from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


PRED_ROOT = (
    Path("output_results")
    / "dynamic_population_distribution"
    / "Pop-HSTM_Historical_Results"
)
TRUTH_DIR = Path("data") / "mobile_phone_data"
DEFAULT_DATE = "20170925"
DEFAULT_OUTPUT_DIR = Path("output_results") / "tables"
WINDOW_CONFIGS = (
    ("w24", 24, "Pop-HSTM_w24"),
    ("w12", 12, "Pop-HSTM_w12"),
    ("w8", 8, "Pop-HSTM_w8"),
    ("w6", 6, "Pop-HSTM_w6"),
)
EPSILON = 1e-5


def relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("Use a relative path, not an absolute path.")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Pop-HSTM historical-window metrics and save them to CSV."
    )
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="Prediction date, for example 20170925. Default matches the figure.",
    )
    parser.add_argument(
        "--pred-root",
        type=relative_path,
        default=PRED_ROOT,
        help="Relative path to Pop-HSTM_Historical_Results.",
    )
    parser.add_argument(
        "--truth-dir",
        type=relative_path,
        default=TRUTH_DIR,
        help="Relative path to hourly ground-truth tif files.",
    )
    parser.add_argument(
        "--output",
        type=relative_path,
        default=None,
        help="Relative output CSV path. Defaults under output_results.",
    )
    parser.add_argument(
        "--hours",
        nargs="+",
        type=int,
        default=None,
        help="Hours to evaluate. Default: 0 1 ... 23.",
    )
    parser.add_argument(
        "--min-truth",
        type=float,
        default=1.0,
        help="Keep pixels whose ground-truth population is at least this value.",
    )
    parser.add_argument(
        "--clip-quantile",
        nargs=2,
        type=float,
        default=(0.01, 0.99),
        metavar=("LOW", "HIGH"),
        help="Truth quantile clipping after min-truth filtering.",
    )
    return parser.parse_args()


def load_tif(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img, dtype=np.float64)


def hourly_paths(
    pred_root: Path,
    truth_dir: Path,
    window_folder: str,
    date: str,
    hours: Iterable[int],
) -> Iterable[tuple[Path, Path]]:
    pred_dir = pred_root / window_folder / f"Pred_{date}"
    truth_date_key = date[-4:]
    for hour in hours:
        yield (
            pred_dir / f"PredPop_{date}_{hour:02d}_real.tif",
            truth_dir / f"point{truth_date_key}{hour:02d}_500.tif",
        )


def load_window_vectors(
    pred_root: Path,
    truth_dir: Path,
    window_folder: str,
    date: str,
    hours: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    pred_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []

    for pred_path, truth_path in hourly_paths(pred_root, truth_dir, window_folder, date, hours):
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")
        if not truth_path.exists():
            raise FileNotFoundError(f"Truth file not found: {truth_path}")
        pred_parts.append(load_tif(pred_path).ravel())
        truth_parts.append(load_tif(truth_path).ravel())

    return np.concatenate(pred_parts), np.concatenate(truth_parts)


def apply_filters(
    pred: np.ndarray,
    truth: np.ndarray,
    min_truth: float,
    clip_quantile: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(pred) & np.isfinite(truth)
    if min_truth is not None:
        keep &= truth >= min_truth
    pred = pred[keep]
    truth = truth[keep]

    if clip_quantile is not None:
        q_low, q_high = float(clip_quantile[0]), float(clip_quantile[1])
        low, high = np.quantile(truth, [q_low, q_high])
        keep = (truth >= low) & (truth <= high)
        pred = pred[keep]
        truth = truth[keep]

    if pred.size == 0:
        raise RuntimeError("No pixels remain after filtering.")
    return pred, truth


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    diff = pred - truth
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff) / (truth + EPSILON)) * 100.0)
    truth_mean = float(np.mean(truth))
    ss_tot = float(np.sum((truth - truth_mean) ** 2))
    ss_res = float(np.sum((truth - pred) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return {"r2": r2, "TEM": math.log1p(mse), "mae": mae, "mape": mape}


def output_path(date: str, requested_output: Path | None) -> Path:
    if requested_output is not None:
        return requested_output
    return DEFAULT_OUTPUT_DIR / f"pop_hstm_historical_window_metrics_{date}.csv"


def main() -> None:
    args = parse_args()
    hours = list(range(24)) if args.hours is None else [int(hour) for hour in args.hours]
    out_csv = output_path(args.date, args.output)

    rows: list[dict[str, str]] = []
    for window_name, window_hours, window_folder in WINDOW_CONFIGS:
        pred, truth = load_window_vectors(
            pred_root=args.pred_root,
            truth_dir=args.truth_dir,
            window_folder=window_folder,
            date=args.date,
            hours=hours,
        )
        pred, truth = apply_filters(
            pred=pred,
            truth=truth,
            min_truth=args.min_truth,
            clip_quantile=args.clip_quantile,
        )
        metrics = compute_metrics(pred, truth)
        rows.append(
            {
                "window": window_name,
                "window_hours": str(window_hours),
                "date": args.date,
                "hours": "00-23" if hours == list(range(24)) else " ".join(f"{h:02d}" for h in hours),
                "n_pixels": str(pred.size),
                "r2": f"{metrics['r2']:.4f}",
                "TEM": f"{metrics['TEM']:.2f}",
                "mae": f"{metrics['mae']:.2f}",
                "mape": f"{metrics['mape']:.2f}",
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["window", "window_hours", "date", "hours", "n_pixels", "r2", "TEM", "mae", "mape"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] saved: {out_csv}")
    for row in rows:
        print(
            f"{row['window']}: R2={row['r2']}, TEM={row['TEM']}, "
            f"MAE={row['mae']}, MAPE={row['mape']}"
        )


if __name__ == "__main__":
    main()
