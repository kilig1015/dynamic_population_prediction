from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import metrics_eval as me


PROJECT_ROOT = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "output_results").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {start}")


REPO_ROOT = find_repo_root(PROJECT_ROOT)
DEFAULT_PRED_DIR = REPO_ROOT / "output_results" / "dynamic_population_distribution" / "Pop-HSTM"
DEFAULT_TRUTH_DIR = REPO_ROOT / "data" / "mobile_phone_data"
DEFAULT_OUTPUT = REPO_ROOT / "output_results" / "tables" / "time_series_analysis.csv"
DEFAULT_FORECAST_CONFIGS = [
    ("24hours", "20170925"),
    ("48hours", "20170926"),
    ("72hours", "20170927"),
    ("96hours", "20170928"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute hourly and daily-summary metrics for available forecast settings "
            "and save a merged time_series_analysis.csv table."
        )
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO_ROOT,
        help="Project root directory.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=DEFAULT_PRED_DIR,
        help="Prediction root folder containing Pred_YYYYMMDD subdirectories.",
    )
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=DEFAULT_TRUTH_DIR,
        help="Ground-truth folder containing pointMMDDHH_500.tif files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--min-truth",
        type=float,
        default=1.0,
        help="Keep pixels with truth >= min_truth.",
    )
    parser.add_argument(
        "--clip-quantile",
        nargs=2,
        type=float,
        default=[0.01, 0.99],
        metavar=("LOW", "HIGH"),
        help="Truth quantile clipping range after min-truth filtering.",
    )
    parser.add_argument(
        "--hours",
        nargs="+",
        type=int,
        default=list(range(24)),
        help="Hours to evaluate. Default: 0..23.",
    )
    return parser.parse_args()


def resolve_path(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo / path)


def resolve_pred_root(pred_root: Path) -> Path:
    if (pred_root / "Pred_20170925").is_dir():
        return pred_root
    pop_hstm_root = pred_root / "Pop-HSTM"
    if (pop_hstm_root / "Pred_20170925").is_dir():
        return pop_hstm_root
    return pred_root


def day_configs(pred_root: Path, hours: List[int]) -> List[Tuple[str, str, Path]]:
    configs: List[Tuple[str, str, Path]] = []
    for hours_tag, date_str in DEFAULT_FORECAST_CONFIGS:
        pred_dir = pred_root / f"Pred_{date_str}"
        required_files = [pred_dir / f"PredPop_{date_str}_{hour:02d}_real.tif" for hour in hours]
        if pred_dir.is_dir() and all(p.exists() for p in required_files):
            configs.append((hours_tag, date_str, pred_dir))
    if not configs:
        raise FileNotFoundError(
            f"No complete forecast-date groups found under: {pred_root}. "
            f"Expected subfolders: {[f'Pred_{date}' for _, date in DEFAULT_FORECAST_CONFIGS]}"
        )
    return configs


def fmt_row(hours_tag: str, hour_tag: str, metrics: dict) -> dict:
    return {
        "hours": hours_tag,
        "hour": hour_tag,
        "r2": f"{metrics['r2']:.4f}",
        "TEM": f"{metrics['tem']:.2f}",
        "mae": f"{metrics['mae']:.2f}",
        "mape": f"{metrics['mape']:.2f}",
    }


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    pred_dir_root = resolve_pred_root(resolve_path(repo, args.pred_dir))
    truth_dir = resolve_path(repo, args.truth_dir)
    output = resolve_path(repo, args.output)
    q_low, q_high = float(args.clip_quantile[0]), float(args.clip_quantile[1])
    clip_quantile = (q_low, q_high)
    eval_hours = [int(h) for h in args.hours]

    if not pred_dir_root.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir_root}")
    if not truth_dir.exists():
        raise FileNotFoundError(f"Truth directory not found: {truth_dir}")

    rows: List[dict] = []
    for hours_tag, date_str, pred_dir in day_configs(pred_dir_root, eval_hours):
        if not pred_dir.exists():
            raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

        date_key = date_str[-4:]
        hourly_pairs = me.load_hourly_vectors(
            pred_dir=pred_dir,
            truth_dir=truth_dir,
            date=date_str,
            hours=eval_hours,
            date_key=date_key,
        )

        day_pred_parts: List[np.ndarray] = []
        day_truth_parts: List[np.ndarray] = []

        for hour, (pred, truth) in zip(eval_hours, hourly_pairs):
            pred_f, truth_f, _ = me.apply_filters(
                pred=pred,
                truth=truth,
                min_truth=args.min_truth,
                clip_quantile=clip_quantile,
            )
            if pred_f.size == 0:
                raise RuntimeError(f"No valid pixels after filtering: {date_str} hour {hour:02d}")
            metrics = me.compute_all_metrics(pred_f, truth_f)
            rows.append(fmt_row(hours_tag, f"{hour:02d}", metrics))
            day_pred_parts.append(pred)
            day_truth_parts.append(truth)

        day_pred = np.concatenate(day_pred_parts)
        day_truth = np.concatenate(day_truth_parts)
        day_pred_f, day_truth_f, _ = me.apply_filters(
            pred=day_pred,
            truth=day_truth,
            min_truth=args.min_truth,
            clip_quantile=clip_quantile,
        )
        day_metrics = me.compute_all_metrics(day_pred_f, day_truth_f)
        rows.append(fmt_row(hours_tag, "summary", day_metrics))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hours", "hour", "r2", "TEM", "mae", "mape"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] saved: {output}")
    for row in rows:
        if row["hour"] == "summary":
            print(
                f"[summary] {row['hours']}: R2={row['r2']}, "
                f"TEM={row['TEM']}, MAE={row['mae']}, MAPE={row['mape']}"
            )


if __name__ == "__main__":
    main()
