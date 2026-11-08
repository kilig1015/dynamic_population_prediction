from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


SETTINGS = (
    "01_project_current",
    "02_no_correction",
    "03_pooled_validation_hourly",
    "04_day_type_specific_hourly",
)
EPSILON = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate Pop-HSTM bias-correction sensitivity metrics using the "
            "original project protocol: concatenate a reporting group first, "
            "then apply the truth threshold and quantile filter once."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "output_results/dynamic_population_distribution/"
            "Pop-HSTM_Bias_Correction_Sensitivity"
        ),
        help="Sensitivity-analysis root containing the four setting folders.",
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=Path("data/mobile_phone_data"),
    )
    parser.add_argument("--min-truth", type=float, default=1.0)
    parser.add_argument(
        "--clip-quantile",
        nargs=2,
        type=float,
        default=(0.01, 0.99),
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <root>/metrics_summary_original_protocol.csv.",
    )
    return parser.parse_args()


def load_raster(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.float64).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D raster, got {array.shape}: {path}")
    return array.ravel()


def discover_dates(root: Path) -> list[str]:
    date_sets: list[set[str]] = []
    for setting in SETTINGS:
        setting_dir = root / setting
        if not setting_dir.is_dir():
            raise FileNotFoundError(f"Missing setting directory: {setting_dir}")
        dates = {
            path.name[len("Pred_") :]
            for path in setting_dir.glob("Pred_????????")
            if path.is_dir()
        }
        date_sets.append(dates)
    common_dates = set.intersection(*date_sets)
    if not common_dates:
        raise RuntimeError("No common prediction dates were found across all settings.")
    return sorted(common_dates)


def day_type(date_tag: str) -> str:
    return "weekday" if datetime.strptime(date_tag, "%Y%m%d").weekday() < 5 else "weekend"


def map_paths(
    root: Path,
    truth_dir: Path,
    setting: str,
    date_tag: str,
) -> Iterable[tuple[Path, Path]]:
    for hour in range(24):
        yield (
            root
            / setting
            / f"Pred_{date_tag}"
            / f"PredPop_{date_tag}_{hour:02d}_real.tif",
            truth_dir / f"point{date_tag[-4:]}{hour:02d}_500.tif",
        )


def load_group_vectors(
    root: Path,
    truth_dir: Path,
    setting: str,
    dates: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    map_count = 0
    for date_tag in dates:
        for prediction_path, truth_path in map_paths(root, truth_dir, setting, date_tag):
            if not prediction_path.is_file():
                raise FileNotFoundError(f"Missing prediction raster: {prediction_path}")
            if not truth_path.is_file():
                raise FileNotFoundError(f"Missing truth raster: {truth_path}")
            prediction = load_raster(prediction_path)
            truth = load_raster(truth_path)
            if prediction.shape != truth.shape:
                raise ValueError(
                    f"Shape mismatch: {prediction_path} {prediction.shape} != "
                    f"{truth_path} {truth.shape}"
                )
            predictions.append(prediction)
            truths.append(truth)
            map_count += 1
    return np.concatenate(predictions), np.concatenate(truths), map_count


def apply_original_filters(
    prediction: np.ndarray,
    truth: np.ndarray,
    min_truth: float | None,
    clip_quantile: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    keep = np.isfinite(prediction) & np.isfinite(truth)
    if min_truth is not None:
        keep &= truth >= min_truth
    prediction = prediction[keep]
    truth = truth[keep]
    if prediction.size == 0:
        raise RuntimeError("No pixels remain after finite/min-truth filtering.")

    quantile_low_value = float("nan")
    quantile_high_value = float("nan")
    if clip_quantile is not None:
        quantile_low_value, quantile_high_value = np.quantile(
            truth, clip_quantile
        ).astype(float)
        keep = (truth >= quantile_low_value) & (truth <= quantile_high_value)
        prediction = prediction[keep]
        truth = truth[keep]
    if prediction.size == 0:
        raise RuntimeError("No pixels remain after quantile filtering.")
    return prediction, truth, quantile_low_value, quantile_high_value


def compute_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    squared_error_sum = float(np.sum(error * error))
    mse = squared_error_sum / float(error.size)
    truth_mean = float(np.mean(truth))
    total_sum_squares = float(np.sum((truth - truth_mean) ** 2))
    truth_sum = float(np.sum(truth))
    return {
        "r2": (
            float("nan")
            if total_sum_squares == 0.0
            else 1.0 - squared_error_sum / total_sum_squares
        ),
        "TEM": float(math.log1p(mse)),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error) / (truth + EPSILON)) * 100.0),
        "mbe": float(np.mean(error)),
        "nmbe_percent": (
            float("nan") if abs(truth_sum) <= EPSILON else 100.0 * float(np.sum(error)) / truth_sum
        ),
    }


def evaluate_group(
    root: Path,
    truth_dir: Path,
    setting: str,
    scope: str,
    group: str,
    dates: Sequence[str],
    min_truth: float | None,
    clip_quantile: tuple[float, float] | None,
) -> dict[str, object]:
    prediction, truth, map_count = load_group_vectors(
        root=root,
        truth_dir=truth_dir,
        setting=setting,
        dates=dates,
    )
    prediction, truth, quantile_low, quantile_high = apply_original_filters(
        prediction=prediction,
        truth=truth,
        min_truth=min_truth,
        clip_quantile=clip_quantile,
    )
    metrics = compute_metrics(prediction, truth)
    return {
        "setting": setting,
        "scope": scope,
        "group": group,
        "day_type": day_type(dates[0]) if scope == "date" else (group if scope == "day_type" else "all"),
        "hours": map_count,
        "n_pixels": int(prediction.size),
        **metrics,
        "filter_quantile_low_value": quantile_low,
        "filter_quantile_high_value": quantile_high,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    truth_dir = args.truth_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "metrics_summary_original_protocol.csv"
    )
    low, high = float(args.clip_quantile[0]), float(args.clip_quantile[1])
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("--clip-quantile must satisfy 0 <= LOW < HIGH <= 1.")
    clip_quantile = (low, high)
    dates = discover_dates(root)
    dates_by_day_type = {
        label: [date_tag for date_tag in dates if day_type(date_tag) == label]
        for label in ("weekday", "weekend")
    }

    rows: list[dict[str, object]] = []
    for setting in SETTINGS:
        for date_tag in dates:
            rows.append(
                evaluate_group(
                    root,
                    truth_dir,
                    setting,
                    "date",
                    date_tag,
                    [date_tag],
                    args.min_truth,
                    clip_quantile,
                )
            )
        for label in ("weekday", "weekend"):
            rows.append(
                evaluate_group(
                    root,
                    truth_dir,
                    setting,
                    "day_type",
                    label,
                    dates_by_day_type[label],
                    args.min_truth,
                    clip_quantile,
                )
            )
        rows.append(
            evaluate_group(
                root,
                truth_dir,
                setting,
                "overall",
                "all_target_dates",
                dates,
                args.min_truth,
                clip_quantile,
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] dates: {dates}")
    print(f"[done] saved: {output}")
    for row in rows:
        if row["scope"] == "overall":
            print(
                f"{row['setting']}: R2={row['r2']:.6f}, TEM={row['TEM']:.6f}, "
                f"MAE={row['mae']:.3f}, MAPE={row['mape']:.3f}"
            )


if __name__ == "__main__":
    main()
