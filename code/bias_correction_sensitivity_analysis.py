from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rasterio
import torch

from pop_hstm.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FINAL_PATCHES_PER_SAMPLE,
    DEFAULT_OVERLAP,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_PREDICTION_OVERLAP,
    DEFAULT_PREDICTION_PATCH_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WINDOW_SIZE,
)
from pop_hstm.data import (
    build_hourly_population_dataset,
    chronological_split_indices,
    create_dataloader,
    recompute_norm_with_indices,
)
from pop_hstm.inference import predict_full_image
from pop_hstm.model import DSConvLSTMNet, build_model_input_channels
from pop_hstm.runtime import synchronize_device
from pop_hstm.training import train_model
from pop_hstm.utils import resolve_project_root, save_json, set_global_determinism


SETTING_PROJECT_CURRENT = "01_project_current"
SETTING_NO_CORRECTION = "02_no_correction"
SETTING_POOLED_HOURLY = "03_pooled_validation_hourly"
SETTING_DAY_TYPE_HOURLY = "04_day_type_specific_hourly"
SETTING_ORDER = (
    SETTING_PROJECT_CURRENT,
    SETTING_NO_CORRECTION,
    SETTING_POOLED_HOURLY,
    SETTING_DAY_TYPE_HOURLY,
)
DAY_TYPES = ("weekday", "weekend")
METRIC_EPSILON = 1e-5
DEFAULT_FULL_VALIDATION_DATES = tuple(
    f"2017-09-{day:02d}" for day in range(19, 25)
)


@dataclass
class LinearSufficientStatistics:
    """Sufficient statistics for y_true = a * y_pred + b."""

    sum_pred: float = 0.0
    sum_true: float = 0.0
    sum_pred2: float = 0.0
    sum_pred_true: float = 0.0
    count: int = 0
    sample_count: int = 0
    dates: set[str] = field(default_factory=set)

    def update(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        target_date: date,
    ) -> None:
        pred64 = np.asarray(pred, dtype=np.float64)
        truth64 = np.asarray(truth, dtype=np.float64)
        self.sum_pred += float(np.sum(pred64))
        self.sum_true += float(np.sum(truth64))
        self.sum_pred2 += float(np.sum(pred64 * pred64))
        self.sum_pred_true += float(np.sum(pred64 * truth64))
        self.count += int(pred64.size)
        self.sample_count += 1
        self.dates.add(target_date.isoformat())

    def solve(self, eps: float = 1e-6) -> Tuple[float, float]:
        if self.count <= 0 or self.sum_pred2 <= eps:
            return 1.0, 0.0
        count = float(self.count)
        mean_pred = self.sum_pred / count
        mean_true = self.sum_true / count
        denominator = self.sum_pred2 - count * mean_pred * mean_pred
        slope = (
            self.sum_pred_true - count * mean_pred * mean_true
        ) / max(denominator, eps)
        intercept = mean_true - slope * mean_pred
        return float(slope), float(intercept)


@dataclass
class MetricAccumulator:
    """Streaming population-domain metrics, including signed bias."""

    n: int = 0
    sum_truth: float = 0.0
    sum_truth2: float = 0.0
    sum_squared_error: float = 0.0
    sum_absolute_error: float = 0.0
    sum_absolute_percentage_error: float = 0.0
    sum_error: float = 0.0

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        pred64 = np.asarray(pred, dtype=np.float64)
        truth64 = np.asarray(truth, dtype=np.float64)
        if pred64.size != truth64.size:
            raise ValueError(
                f"Prediction/truth size mismatch: {pred64.size} != {truth64.size}"
            )
        if pred64.size == 0:
            return
        error = pred64 - truth64
        self.n += int(error.size)
        self.sum_truth += float(np.sum(truth64))
        self.sum_truth2 += float(np.sum(truth64 * truth64))
        self.sum_squared_error += float(np.sum(error * error))
        self.sum_absolute_error += float(np.sum(np.abs(error)))
        self.sum_absolute_percentage_error += float(
            np.sum(np.abs(error) / (truth64 + METRIC_EPSILON))
        )
        self.sum_error += float(np.sum(error))

    def as_metrics(self) -> Dict[str, float]:
        if self.n <= 0:
            return {
                "r2": float("nan"),
                "TEM": float("nan"),
                "mae": float("nan"),
                "mape": float("nan"),
                "mbe": float("nan"),
                "nmbe_percent": float("nan"),
            }
        n = float(self.n)
        truth_mean = self.sum_truth / n
        ss_total = self.sum_truth2 - n * truth_mean * truth_mean
        r2 = (
            float("nan")
            if ss_total <= 0.0
            else 1.0 - self.sum_squared_error / ss_total
        )
        mse = self.sum_squared_error / n
        nmbe = (
            float("nan")
            if abs(self.sum_truth) <= METRIC_EPSILON
            else 100.0 * self.sum_error / self.sum_truth
        )
        return {
            "r2": float(r2),
            "TEM": float(math.log1p(mse)),
            "mae": float(self.sum_absolute_error / n),
            "mape": float(100.0 * self.sum_absolute_percentage_error / n),
            "mbe": float(self.sum_error / n),
            "nmbe_percent": float(nmbe),
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run four Pop-HSTM bias-correction settings from identical raw "
            "model predictions and report weekday/weekend sensitivity metrics."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional existing Pop-HSTM state_dict checkpoint. When omitted, "
            "the script first trains the standard Pop-HSTM model and saves its "
            "best checkpoint inside the sensitivity-analysis output directory."
        ),
    )
    parser.add_argument(
        "--current-calibration-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON containing the exact coefficients used by the current/"
            "published run. Expected mapping: hour -> [a, b]. When omitted, "
            "setting 1 is fitted from the legacy/current-project validation subset "
            "created by the chronological split."
        ),
    )
    parser.add_argument(
        "--full-validation-dates",
        nargs="+",
        default=list(DEFAULT_FULL_VALIDATION_DATES),
        help=(
            "Complete validation-date range used by settings 3 and 4. "
            "Default: 2017-09-19 through 2017-09-24. Setting 1 keeps the "
            "legacy/current-project validation subset unless an exact "
            "--current-calibration-json is supplied."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "output_results/dynamic_population_distribution/"
            "Pop-HSTM_Bias_Correction_Sensitivity"
        ),
    )
    parser.add_argument(
        "--target-dates",
        nargs="*",
        default=None,
        help=(
            "Prediction dates in YYYY-MM-DD form. By default, use every date in "
            "the chronological test split that is not part of the complete "
            "validation range (currently 2017-09-25 through 2017-09-30)."
        ),
    )
    parser.add_argument("--included-months", nargs="+", type=int, default=[9])
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--training-patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--training-overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--prediction-patch-size", type=int, default=DEFAULT_PREDICTION_PATCH_SIZE
    )
    parser.add_argument(
        "--prediction-overlap", type=int, default=DEFAULT_PREDICTION_OVERLAP
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--patches-per-sample",
        type=int,
        default=DEFAULT_FINAL_PATCHES_PER_SAMPLE,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="DataLoader worker count used only when training is performed.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. 'auto' selects CUDA when available.",
    )
    parser.add_argument(
        "--min-truth",
        type=float,
        default=1.0,
        help="Minimum truth population retained for reported metrics.",
    )
    parser.add_argument(
        "--clip-quantile",
        nargs=2,
        type=float,
        default=[0.01, 0.99],
        metavar=("LOW", "HIGH"),
        help="Truth-based per-map quantile filter used for all four settings.",
    )
    parser.add_argument(
        "--disable-quantile-clipping",
        action="store_true",
        help="Disable the --clip-quantile filter.",
    )
    parser.add_argument(
        "--skip-raster-export",
        action="store_true",
        help="Compute coefficients and metrics without writing the four GeoTIFF sets.",
    )
    return parser.parse_args(argv)


def resolve_user_path(project_root: Path, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.is_absolute() else project_root / path


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def day_type_for_datetime(dt: datetime) -> str:
    return "weekend" if dt.weekday() >= 5 else "weekday"


def identity_hourly_calibration() -> Dict[int, Tuple[float, float]]:
    return {hour: (1.0, 0.0) for hour in range(24)}


def load_hourly_calibration(path: Path) -> Dict[int, Tuple[float, float]]:
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    calibration: Dict[int, Tuple[float, float]] = {}
    for hour in range(24):
        value = raw.get(str(hour), raw.get(hour))
        if value is None or not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                f"Calibration JSON {path} is missing a valid [a,b] pair for hour {hour}."
            )
        calibration[hour] = (float(value[0]), float(value[1]))
    return calibration


def calibration_json_ready(
    calibration: Mapping[int, Tuple[float, float]],
) -> Dict[str, List[float]]:
    return {
        str(hour): [float(calibration[hour][0]), float(calibration[hour][1])]
        for hour in range(24)
    }


def new_hourly_statistics() -> Dict[int, LinearSufficientStatistics]:
    return {hour: LinearSufficientStatistics() for hour in range(24)}


def fit_calibration_from_statistics(
    statistics: Mapping[int, LinearSufficientStatistics],
) -> Dict[int, Tuple[float, float]]:
    return {hour: statistics[hour].solve() for hour in range(24)}


def valid_population_vectors(
    pred_population: np.ndarray,
    true_population: np.ndarray,
    study_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(pred_population) & np.isfinite(true_population)
    if study_mask is not None:
        valid &= study_mask > 0
    return (
        pred_population[valid].astype(np.float64, copy=False),
        true_population[valid].astype(np.float64, copy=False),
    )


def estimate_validation_calibrations(
    model: torch.nn.Module,
    dataset,
    full_validation_indices: Sequence[int],
    current_project_indices: Sequence[int],
    device: torch.device,
    patch_size: int,
    overlap: int,
) -> Tuple[
    Dict[int, Tuple[float, float]],
    Dict[int, Tuple[float, float]],
    Dict[str, Dict[int, Tuple[float, float]]],
    Dict[int, LinearSufficientStatistics],
    Dict[int, LinearSufficientStatistics],
    Dict[str, Dict[int, LinearSufficientStatistics]],
]:
    """Estimate partial-current, full-pooled, and full-day-type coefficients."""
    current_project_stats = new_hourly_statistics()
    pooled_stats = new_hourly_statistics()
    day_type_stats = {day_type: new_hourly_statistics() for day_type in DAY_TYPES}
    study_mask = getattr(dataset, "study_mask_full", None)
    current_project_index_set = {int(index) for index in current_project_indices}

    model.eval()
    with torch.no_grad():
        for position, sample_idx in enumerate(full_validation_indices, start=1):
            sample = dataset.samples[sample_idx]
            target_key = sample["key_y"]
            target_meta = dataset._meta_by_key[target_key]
            target_dt = target_meta["dt"]
            target_hour = int(target_meta["hour"])
            day_type = day_type_for_datetime(target_dt)

            input_full, target_sqrt_full = dataset.get_full_sequence(sample_idx)
            pred_sqrt_full = predict_full_image(
                model=model,
                X_full=input_full,
                device=device,
                patch_size=patch_size,
                overlap=overlap,
            )
            pred_population = np.square(np.maximum(pred_sqrt_full, 0.0))
            true_population = np.square(
                np.maximum(target_sqrt_full.squeeze(-1), 0.0)
            )
            pred_vector, true_vector = valid_population_vectors(
                pred_population, true_population, study_mask
            )
            pooled_stats[target_hour].update(
                pred_vector, true_vector, target_dt.date()
            )
            day_type_stats[day_type][target_hour].update(
                pred_vector, true_vector, target_dt.date()
            )
            if sample_idx in current_project_index_set:
                current_project_stats[target_hour].update(
                    pred_vector, true_vector, target_dt.date()
                )

            if position % 24 == 0 or position == len(full_validation_indices):
                print(
                    f"[calibration] processed {position}/{len(full_validation_indices)} "
                    "full-validation samples"
                )

    current_project = fit_calibration_from_statistics(current_project_stats)
    pooled = fit_calibration_from_statistics(pooled_stats)
    day_type_calibration = {
        day_type: fit_calibration_from_statistics(day_type_stats[day_type])
        for day_type in DAY_TYPES
    }
    return (
        current_project,
        pooled,
        day_type_calibration,
        current_project_stats,
        pooled_stats,
        day_type_stats,
    )


def apply_linear_calibration(
    population: np.ndarray,
    coefficients: Tuple[float, float],
) -> np.ndarray:
    slope, intercept = coefficients
    return np.maximum(slope * population + intercept, 0.0).astype(
        np.float32, copy=False
    )


def save_population_map(
    population: np.ndarray,
    reference_tif: Path,
    output_tif: Path,
    study_mask: Optional[np.ndarray],
) -> None:
    output = np.maximum(population, 0.0).astype(np.float32, copy=False)
    if study_mask is not None:
        output = output * study_mask.astype(np.float32, copy=False)
    with rasterio.open(reference_tif) as source:
        profile = source.profile.copy()
    profile.update({"count": 1, "dtype": "float32"})
    profile.pop("photometric", None)
    profile.pop("colormap", None)
    profile.pop("nodata", None)
    profile.pop("NODATA", None)
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_tif, "w", **profile) as destination:
        destination.write(output, 1)


def filter_metric_vectors(
    pred_population: np.ndarray,
    true_population: np.ndarray,
    study_mask: Optional[np.ndarray],
    min_truth: Optional[float],
    clip_quantile: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    pred, truth = valid_population_vectors(
        pred_population, true_population, study_mask
    )
    if min_truth is not None:
        keep = truth >= float(min_truth)
        pred, truth = pred[keep], truth[keep]
    if clip_quantile is not None and truth.size > 0:
        low, high = np.quantile(truth, clip_quantile)
        keep = (truth >= low) & (truth <= high)
        pred, truth = pred[keep], truth[keep]
    return pred, truth


def metrics_row(
    accumulator: MetricAccumulator,
    **identifiers,
) -> Dict[str, object]:
    row: Dict[str, object] = dict(identifiers)
    row["n_pixels"] = int(accumulator.n)
    row.update(accumulator.as_metrics())
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_model(device: torch.device) -> torch.nn.Module:
    return DSConvLSTMNet(
        in_channels=build_model_input_channels(), use_transformer=True
    ).to(device)


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(device)
    state = torch.load(str(checkpoint), map_location=device)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported checkpoint content in {checkpoint}")
    if state and all(str(key).startswith("module.") for key in state.keys()):
        state = {str(key)[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def split_date_list(dataset, indices: Iterable[int]) -> List[str]:
    dates = {
        dataset._meta_by_key[dataset.samples[index]["key_y"]]["dt"].date()
        for index in indices
    }
    return [value.isoformat() for value in sorted(dates)]


def indices_for_target_dates(
    dataset,
    target_dates: Sequence[str],
) -> List[int]:
    requested = {
        datetime.strptime(value, "%Y-%m-%d").date() for value in target_dates
    }
    selected: List[int] = []
    for sample_idx, sample in enumerate(dataset.samples):
        target_dt = dataset._meta_by_key[sample["key_y"]]["dt"]
        if target_dt.date() in requested:
            selected.append(sample_idx)
    found = {
        dataset._meta_by_key[dataset.samples[index]["key_y"]]["dt"].date()
        for index in selected
    }
    missing = sorted(requested - found)
    if missing:
        raise ValueError(
            "No model samples were found for target dates: "
            + ", ".join(value.isoformat() for value in missing)
        )
    return selected


def coefficient_rows(
    project_current: Mapping[int, Tuple[float, float]],
    pooled: Mapping[int, Tuple[float, float]],
    day_type_calibration: Mapping[str, Mapping[int, Tuple[float, float]]],
    project_current_stats: Mapping[int, LinearSufficientStatistics],
    pooled_stats: Mapping[int, LinearSufficientStatistics],
    day_type_stats: Mapping[str, Mapping[int, LinearSufficientStatistics]],
    current_source: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for setting, calibration, source in (
        (SETTING_PROJECT_CURRENT, project_current, current_source),
        (SETTING_NO_CORRECTION, identity_hourly_calibration(), "identity"),
        (SETTING_POOLED_HOURLY, pooled, "all_validation_dates"),
    ):
        for hour in range(24):
            stats = (
                project_current_stats[hour]
                if setting == SETTING_PROJECT_CURRENT
                else pooled_stats[hour]
            )
            slope, intercept = calibration[hour]
            is_fitted = setting != SETTING_NO_CORRECTION
            rows.append(
                {
                    "setting": setting,
                    "coefficient_day_type": "all",
                    "hour": hour,
                    "a": slope,
                    "b": intercept,
                    "source": source,
                    "validation_samples": stats.sample_count if is_fitted else 0,
                    "validation_dates": (
                        "|".join(sorted(stats.dates)) if is_fitted else ""
                    ),
                    "fitted_pixels": stats.count if is_fitted else 0,
                }
            )
    for day_type in DAY_TYPES:
        for hour in range(24):
            stats = day_type_stats[day_type][hour]
            slope, intercept = day_type_calibration[day_type][hour]
            rows.append(
                {
                    "setting": SETTING_DAY_TYPE_HOURLY,
                    "coefficient_day_type": day_type,
                    "hour": hour,
                    "a": slope,
                    "b": intercept,
                    "source": f"all_validation_{day_type}_dates",
                    "validation_samples": stats.sample_count,
                    "validation_dates": "|".join(sorted(stats.dates)),
                    "fitted_pixels": stats.count,
                }
            )
    return rows


def run_analysis(args: argparse.Namespace) -> None:
    set_global_determinism(args.seed)
    project_root = resolve_project_root(Path(__file__))
    checkpoint = resolve_user_path(project_root, args.checkpoint)
    output_dir = resolve_user_path(project_root, args.output_dir)
    current_calibration_json = resolve_user_path(
        project_root, args.current_calibration_json
    )
    assert output_dir is not None

    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(
            f"Pop-HSTM checkpoint not found: {checkpoint}\n"
            "Omit --checkpoint to train Pop-HSTM inside this analysis, or pass "
            "an existing checkpoint path."
        )
    if current_calibration_json is not None and not current_calibration_json.is_file():
        raise FileNotFoundError(
            f"Current calibration JSON not found: {current_calibration_json}"
        )
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not (0.0 <= args.validation_ratio < 1.0):
        raise ValueError("--validation-ratio must be in [0, 1).")
    if args.train_ratio + args.validation_ratio >= 1.0:
        raise ValueError("train ratio + validation ratio must be less than 1.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if args.batch_size <= 0 or args.patches_per_sample <= 0:
        raise ValueError("--batch-size and --patches-per-sample must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    clip_quantile: Optional[Tuple[float, float]]
    if args.disable_quantile_clipping:
        clip_quantile = None
    else:
        clip_quantile = (
            float(args.clip_quantile[0]),
            float(args.clip_quantile[1]),
        )
        if not (0.0 <= clip_quantile[0] < clip_quantile[1] <= 1.0):
            raise ValueError(
                "--clip-quantile must satisfy 0 <= LOW < HIGH <= 1."
            )

    device = choose_device(args.device)
    synchronize_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] project root: {project_root}")
    print(f"[setup] device: {device}")
    print(
        "[setup] model source: "
        + (f"existing checkpoint {checkpoint}" if checkpoint else "train in this run")
    )

    dataset = build_hourly_population_dataset(
        project_root=project_root,
        window_size=args.window_size,
        patch_size=args.training_patch_size,
        overlap=args.training_overlap,
        random_seed=args.seed,
    )
    split = chronological_split_indices(
        dataset,
        included_months=tuple(args.included_months),
        train_ratio=args.train_ratio,
        val_ratio=args.validation_ratio,
    )
    # Match the normalization used by the standard Pop-HSTM workflow/checkpoint.
    recompute_norm_with_indices(dataset, split.train_indices)

    train_dates = split_date_list(dataset, split.train_indices)
    current_project_validation_dates = split_date_list(dataset, split.val_indices)
    chronological_test_dates = split_date_list(dataset, split.test_indices)
    full_validation_dates = [
        datetime.strptime(value, "%Y-%m-%d").date().isoformat()
        for value in args.full_validation_dates
    ]
    if len(full_validation_dates) != len(set(full_validation_dates)):
        raise ValueError("--full-validation-dates contains duplicate dates.")
    full_validation_dates = sorted(full_validation_dates)
    full_validation_indices = indices_for_target_dates(dataset, full_validation_dates)

    current_date_set = set(current_project_validation_dates)
    full_validation_date_set = set(full_validation_dates)
    if not current_date_set.issubset(full_validation_date_set):
        missing = sorted(current_date_set - full_validation_date_set)
        raise ValueError(
            "The complete validation range must contain every current-project "
            "validation date; missing: " + ", ".join(missing)
        )
    if current_calibration_json is None and current_date_set == full_validation_date_set:
        raise ValueError(
            "Settings 1 and 3 would use the same validation dates. Expand "
            "--full-validation-dates or supply --current-calibration-json."
        )

    default_target_dates = [
        value
        for value in chronological_test_dates
        if value not in full_validation_date_set
    ]
    target_dates = (
        list(args.target_dates) if args.target_dates else default_target_dates
    )
    overlap_dates = sorted(set(target_dates) & full_validation_date_set)
    if overlap_dates:
        raise ValueError(
            "Target/test dates overlap the complete validation range: "
            + ", ".join(overlap_dates)
        )
    if not target_dates:
        raise ValueError("No target/test dates remain after excluding validation dates.")
    target_indices = indices_for_target_dates(dataset, target_dates)

    validation_day_types = {
        day_type: sorted(
            {
                dataset._meta_by_key[dataset.samples[index]["key_y"]]["dt"]
                .date()
                .isoformat()
                for index in full_validation_indices
                if day_type_for_datetime(
                    dataset._meta_by_key[dataset.samples[index]["key_y"]]["dt"]
                )
                == day_type
            }
        )
        for day_type in DAY_TYPES
    }
    target_day_types = {
        day_type: sorted(
            {
                dataset._meta_by_key[dataset.samples[index]["key_y"]]["dt"]
                .date()
                .isoformat()
                for index in target_indices
                if day_type_for_datetime(
                    dataset._meta_by_key[dataset.samples[index]["key_y"]]["dt"]
                )
                == day_type
            }
        )
        for day_type in DAY_TYPES
    }

    print(f"[split] train dates: {train_dates[0]} to {train_dates[-1]}")
    print(
        "[split] current-project calibration dates (partial): "
        f"{current_project_validation_dates}"
    )
    print(f"[split] complete validation dates (settings 3/4): {full_validation_dates}")
    print(f"[split] chronological test candidates: {chronological_test_dates}")
    print(f"[analysis] target dates: {target_dates}")
    if not validation_day_types["weekend"]:
        raise RuntimeError(
            "Day-type-specific calibration cannot be fitted because the "
            "validation subset contains no weekend date."
        )
    if not validation_day_types["weekday"]:
        raise RuntimeError(
            "Day-type-specific calibration cannot be fitted because the "
            "validation subset contains no weekday date."
        )
    if not target_day_types["weekend"]:
        print(
            "[warn] No weekend target date was selected; weekend transferability "
            "cannot be assessed in the requested output dates."
        )

    training_performed = checkpoint is None
    training_history: List[Dict[str, object]] = []
    if training_performed:
        model_dir = output_dir / "model"
        best_loss_path = model_dir / "best_loss_Pop-HSTM.pt"
        best_tem_path = model_dir / "best_tem_Pop-HSTM.pt"
        print(
            "[training] No --checkpoint supplied. Starting standard Pop-HSTM "
            f"training for up to {args.epochs} epochs."
        )
        train_loader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            augment=True,
            sample_indices=split.train_indices,
            patches_per_sample=args.patches_per_sample,
            num_workers=args.num_workers,
        )
        validation_loader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            augment=False,
            sample_indices=split.val_indices,
            patches_per_sample=args.patches_per_sample,
            num_workers=args.num_workers,
        )
        test_loader = create_dataloader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            augment=False,
            sample_indices=target_indices,
            patches_per_sample=args.patches_per_sample,
            num_workers=args.num_workers,
        )
        training_model = build_model(device)
        training_history = train_model(
            train_loader=train_loader,
            val_loader=validation_loader,
            test_loader=test_loader,
            model=training_model,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            best_loss_path=str(best_loss_path),
            best_tem_path=str(best_tem_path),
            checkpoint_map=None,
        )
        if not best_tem_path.is_file():
            raise RuntimeError(
                f"Training finished without creating the expected checkpoint: {best_tem_path}"
            )
        save_json(training_history, model_dir / "training_history.json")
        write_csv(model_dir / "training_history.csv", training_history)
        checkpoint = best_tem_path
        del training_model, train_loader, validation_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[training] best validation-TEM checkpoint: {checkpoint}")

    assert checkpoint is not None
    model = load_model(checkpoint, device)
    (
        fitted_current_project_calibration,
        pooled_calibration,
        day_type_calibration,
        current_project_stats,
        pooled_stats,
        day_type_stats,
    ) = estimate_validation_calibrations(
        model=model,
        dataset=dataset,
        full_validation_indices=full_validation_indices,
        current_project_indices=split.val_indices,
        device=device,
        patch_size=args.prediction_patch_size,
        overlap=args.prediction_overlap,
    )

    if current_calibration_json is None:
        project_current_calibration = dict(fitted_current_project_calibration)
        current_source = "legacy_current_project_partial_validation_dates"
    else:
        project_current_calibration = load_hourly_calibration(
            current_calibration_json
        )
        current_source = str(current_calibration_json)

    current_equals_pooled = all(
        np.allclose(
            project_current_calibration[hour],
            pooled_calibration[hour],
            rtol=1e-10,
            atol=1e-10,
        )
        for hour in range(24)
    )
    print(
        "[audit] setting 1 and setting 3 coefficient equivalence: "
        f"{current_equals_pooled}"
    )
    if current_calibration_json is None:
        print(
            "[audit] No --current-calibration-json was supplied; setting 1 was "
            "fitted from the partial current-project validation dates, whereas "
            "setting 3 was fitted from the complete validation date range."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        calibration_json_ready(project_current_calibration),
        output_dir / "project_current_hourly_calibration.json",
    )
    save_json(
        calibration_json_ready(pooled_calibration),
        output_dir / "pooled_validation_hourly_calibration.json",
    )
    save_json(
        {
            day_type: calibration_json_ready(day_type_calibration[day_type])
            for day_type in DAY_TYPES
        },
        output_dir / "day_type_specific_hourly_calibration.json",
    )
    write_csv(
        output_dir / "calibration_coefficients.csv",
        coefficient_rows(
            project_current=project_current_calibration,
            pooled=pooled_calibration,
            day_type_calibration=day_type_calibration,
            project_current_stats=current_project_stats,
            pooled_stats=pooled_stats,
            day_type_stats=day_type_stats,
            current_source=current_source,
        ),
    )

    daily_accumulators: Dict[Tuple[str, str], MetricAccumulator] = {}
    day_type_accumulators: Dict[Tuple[str, str], MetricAccumulator] = {}
    day_type_hourly_accumulators: Dict[
        Tuple[str, str, int], MetricAccumulator
    ] = {}
    overall_accumulators = {
        setting: MetricAccumulator() for setting in SETTING_ORDER
    }
    hourly_rows: List[Dict[str, object]] = []
    study_mask = getattr(dataset, "study_mask_full", None)

    model.eval()
    with torch.no_grad():
        for position, sample_idx in enumerate(target_indices, start=1):
            sample = dataset.samples[sample_idx]
            target_key = sample["key_y"]
            target_meta = dataset._meta_by_key[target_key]
            target_dt = target_meta["dt"]
            target_date_tag = target_dt.strftime("%Y%m%d")
            target_hour = int(target_meta["hour"])
            day_type = day_type_for_datetime(target_dt)

            input_full, target_sqrt_full = dataset.get_full_sequence(sample_idx)
            raw_pred_sqrt = predict_full_image(
                model=model,
                X_full=input_full,
                device=device,
                patch_size=args.prediction_patch_size,
                overlap=args.prediction_overlap,
            )
            raw_population = np.square(np.maximum(raw_pred_sqrt, 0.0)).astype(
                np.float32, copy=False
            )
            true_population = np.square(
                np.maximum(target_sqrt_full.squeeze(-1), 0.0)
            ).astype(np.float32, copy=False)

            corrected_by_setting = {
                SETTING_PROJECT_CURRENT: apply_linear_calibration(
                    raw_population, project_current_calibration[target_hour]
                ),
                SETTING_NO_CORRECTION: raw_population,
                SETTING_POOLED_HOURLY: apply_linear_calibration(
                    raw_population, pooled_calibration[target_hour]
                ),
                SETTING_DAY_TYPE_HOURLY: apply_linear_calibration(
                    raw_population, day_type_calibration[day_type][target_hour]
                ),
            }

            reference_tif = Path(dataset.pop_path_map[target_key])
            for setting in SETTING_ORDER:
                pred_population = corrected_by_setting[setting]
                if not args.skip_raster_export:
                    output_tif = (
                        output_dir
                        / setting
                        / f"Pred_{target_date_tag}"
                        / f"PredPop_{target_date_tag}_{target_hour:02d}_real.tif"
                    )
                    save_population_map(
                        pred_population,
                        reference_tif,
                        output_tif,
                        study_mask,
                    )

                pred_vector, true_vector = filter_metric_vectors(
                    pred_population=pred_population,
                    true_population=true_population,
                    study_mask=study_mask,
                    min_truth=args.min_truth,
                    clip_quantile=clip_quantile,
                )
                hourly_accumulator = MetricAccumulator()
                hourly_accumulator.update(pred_vector, true_vector)
                hourly_rows.append(
                    metrics_row(
                        hourly_accumulator,
                        setting=setting,
                        date=target_date_tag,
                        day_type=day_type,
                        hour=target_hour,
                    )
                )

                daily_accumulators.setdefault(
                    (setting, target_date_tag), MetricAccumulator()
                ).update(pred_vector, true_vector)
                day_type_accumulators.setdefault(
                    (setting, day_type), MetricAccumulator()
                ).update(pred_vector, true_vector)
                day_type_hourly_accumulators.setdefault(
                    (setting, day_type, target_hour), MetricAccumulator()
                ).update(pred_vector, true_vector)
                overall_accumulators[setting].update(pred_vector, true_vector)

            if position % 24 == 0 or position == len(target_indices):
                print(
                    f"[prediction] processed {position}/{len(target_indices)} "
                    "held-out target samples"
                )

    summary_rows: List[Dict[str, object]] = []
    for setting in SETTING_ORDER:
        for target_date_tag in sorted(
            {date_tag for candidate, date_tag in daily_accumulators if candidate == setting}
        ):
            target_dt = datetime.strptime(target_date_tag, "%Y%m%d")
            summary_rows.append(
                metrics_row(
                    daily_accumulators[(setting, target_date_tag)],
                    setting=setting,
                    scope="date",
                    group=target_date_tag,
                    day_type=day_type_for_datetime(target_dt),
                    hours=24,
                )
            )
        for day_type in DAY_TYPES:
            accumulator = day_type_accumulators.get(
                (setting, day_type), MetricAccumulator()
            )
            summary_rows.append(
                metrics_row(
                    accumulator,
                    setting=setting,
                    scope="day_type",
                    group=day_type,
                    day_type=day_type,
                    hours=len(target_day_types[day_type]) * 24,
                )
            )
        summary_rows.append(
            metrics_row(
                overall_accumulators[setting],
                setting=setting,
                scope="overall",
                group="all_target_dates",
                day_type="all",
                hours=len(target_dates) * 24,
            )
        )

    day_type_hourly_rows: List[Dict[str, object]] = []
    for setting in SETTING_ORDER:
        for day_type in DAY_TYPES:
            for hour in range(24):
                accumulator = day_type_hourly_accumulators.get(
                    (setting, day_type, hour), MetricAccumulator()
                )
                day_type_hourly_rows.append(
                    metrics_row(
                        accumulator,
                        setting=setting,
                        day_type=day_type,
                        hour=hour,
                    )
                )

    write_csv(output_dir / "metrics_hourly.csv", hourly_rows)
    write_csv(output_dir / "metrics_day_type_hourly.csv", day_type_hourly_rows)
    write_csv(output_dir / "metrics_summary.csv", summary_rows)

    save_json(
        {
            "analysis": "Pop-HSTM bias-correction sensitivity analysis",
            "device": str(device),
            "model": {
                "training_performed_in_this_run": training_performed,
                "checkpoint": str(checkpoint),
                "epochs_requested": args.epochs if training_performed else None,
                "epochs_completed": len(training_history),
                "patience": args.patience if training_performed else None,
                "batch_size": args.batch_size if training_performed else None,
                "patches_per_sample": (
                    args.patches_per_sample if training_performed else None
                ),
                "num_workers": args.num_workers if training_performed else None,
            },
            "settings": {
                SETTING_PROJECT_CURRENT: {
                    "description": (
                        "Current-project hourly correction fitted from the partial "
                        "legacy validation-date subset"
                    ),
                    "coefficient_source": current_source,
                    "coefficient_dates": current_project_validation_dates,
                },
                SETTING_NO_CORRECTION: {
                    "description": "Raw main-model output without a_h,b_h"
                },
                SETTING_POOLED_HOURLY: {
                    "description": (
                        "One hourly (a_h,b_h) pair fitted from all validation dates"
                    ),
                    "coefficient_dates": full_validation_dates,
                },
                SETTING_DAY_TYPE_HOURLY: {
                    "description": (
                        "Separate weekday/weekend hourly coefficient pairs fitted "
                        "from all validation dates"
                    ),
                    "coefficient_dates": validation_day_types,
                },
            },
            "setting_1_equals_setting_3": current_equals_pooled,
            "split": {
                "method": (
                    "legacy_chronological_split_plus_explicit_complete_"
                    "validation_range"
                ),
                "included_months": list(args.included_months),
                "train_ratio": args.train_ratio,
                "validation_ratio": args.validation_ratio,
                "train_dates": train_dates,
                "validation_dates": full_validation_dates,
                "validation_dates_by_day_type": validation_day_types,
                "test_dates": target_dates,
                "current_project_partial_validation_dates": (
                    current_project_validation_dates
                ),
                "full_validation_dates": full_validation_dates,
                "full_validation_dates_by_day_type": validation_day_types,
                "chronological_test_candidate_dates": chronological_test_dates,
                "effective_test_dates": target_dates,
            },
            "target_dates": target_dates,
            "target_dates_by_day_type": target_day_types,
            "metric_filter": {
                "min_truth": args.min_truth,
                "clip_quantile_per_target_map": clip_quantile,
            },
            "caution": (
                "The complete validation range contains "
                f"{len(validation_day_types['weekend'])} weekend date(s) "
                f"({', '.join(validation_day_types['weekend'])}); day-type-specific "
                "weekend coefficients still have limited temporal support and "
                "should be interpreted as a sensitivity analysis."
            ),
            "outputs": {
                "coefficients": str(output_dir / "calibration_coefficients.csv"),
                "hourly_metrics": str(output_dir / "metrics_hourly.csv"),
                "day_type_hourly_metrics": str(
                    output_dir / "metrics_day_type_hourly.csv"
                ),
                "summary_metrics": str(output_dir / "metrics_summary.csv"),
                "raster_exported": not args.skip_raster_export,
                "trained_model_directory": (
                    str(output_dir / "model") if training_performed else None
                ),
            },
        },
        output_dir / "analysis_config.json",
    )

    print(f"[done] sensitivity-analysis outputs saved under: {output_dir}")
    print(f"[done] summary metrics: {output_dir / 'metrics_summary.csv'}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_analysis(args)


if __name__ == "__main__":
    main()
