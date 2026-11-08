from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import rasterio
import torch
import torch.nn as nn

from .config import DEFAULT_PREDICTION_OVERLAP, DEFAULT_PREDICTION_PATCH_SIZE
from .runtime import synchronize_device
from .utils import _patch_start_indices, numpy_sequence_to_tensor

if TYPE_CHECKING:
    from .data import HourlyPopulationDataset

def save_real_population_map(
        sqrt_map: np.ndarray,
        reference_tif: str,
        output_tif: str,
        mask: Optional[np.ndarray] = None,
) -> None:
    """Convert predicted sqrt(pop) to population and save it as GeoTIFF using the reference raster."""
    real_map = np.square(np.maximum(sqrt_map, 0.0)).astype(np.float32)
    if mask is not None:
        real_map = real_map * mask.astype(np.float32)
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    profile.update({"count": 1, "dtype": "float32"})
    # Remove nodata/photometric fields so zeros are not treated as NoData and create holes.
    profile.pop("photometric", None)
    profile.pop("colormap", None)
    profile.pop("nodata", None)
    profile.pop("NODATA", None)
    out_path = Path(output_tif)
    if out_path.exists():
        try:
            out_path.unlink()
        except PermissionError as exc:
            raise RuntimeError(f"Failed to delete existing file {output_tif}, please close the file and try again.") from exc
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(real_map, 1)
    print(f"Preserved predicted population grid saved to {output_tif}.")

def apply_hourly_linear_calibration(
        pop_map: np.ndarray,
        hour: int,
        hourly_calib: Optional[dict[int, tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Apply hour-specific linear calibration in the population domain: pop_corr = a_h * pop_map + b_h.
    - hourly_calib: {hour: (a, b)}; if missing, no calibration is applied for that hour.
    - the output is clamped to be non-negative.
    """
    if not hourly_calib:
        return pop_map
    a, b = hourly_calib.get(hour, (1.0, 0.0))
    return np.maximum(a * pop_map + b, 0.0)

def compute_hourly_calibration(
        model: nn.Module,
        base_dataset: HourlyPopulationDataset,
        device: torch.device,
        sample_indices: Sequence[int],
        patch_size: int = 96,
        overlap: int = 32,
        eps: float = 1e-6,
) -> dict[int, tuple[float, float]]:
    """
    Estimate per-hour linear calibration coefficients (a_h, b_h) from the given samples, usually validation data,
    under the relation pop_true ~= a_h * pop_pred + b_h.
    Least-squares fitting is done in the population domain using the full reconstructed image.
    """
    model.eval()
    dataset_mask = getattr(base_dataset, "study_mask_full", None)
    # Accumulate per-hour statistics.
    stats = {
        h: {
            "sum_pred": 0.0,
            "sum_true": 0.0,
            "sum_pred2": 0.0,
            "sum_pred_true": 0.0,
            "count": 0.0,
        }
        for h in range(24)
    }

    with torch.no_grad():
        for sample_idx in sample_indices:
            sample_meta = base_dataset.samples[sample_idx]
            target_pop_key = sample_meta["key_y"]
            target_time_meta = base_dataset._meta_by_key[target_pop_key]
            target_hour = target_time_meta["hour"]

            input_sequence_full, target_sqrt_full = base_dataset.get_full_sequence(sample_idx)  # (T,H,W,C), (H,W,1) sqrt(pop)
            pred_sqrt_map = predict_full_image(
                model=model,
                X_full=input_sequence_full,
                device=device,
                patch_size=patch_size,
                overlap=overlap,
            )
            true_population_map = np.square(np.maximum(target_sqrt_full.squeeze(-1), 0.0))
            pred_population_map = np.square(np.maximum(pred_sqrt_map, 0.0))

            if dataset_mask is not None:
                true_population_map = true_population_map * dataset_mask
                pred_population_map = pred_population_map * dataset_mask

            # Count only valid pixels (non-NaN); when a study-area mask exists, only count pixels inside it.
            valid_population_mask = np.isfinite(true_population_map) & np.isfinite(pred_population_map)
            if dataset_mask is not None:
                valid_population_mask &= dataset_mask > 0

            if not np.any(valid_population_mask):
                continue
            valid_true_population = true_population_map[valid_population_mask].astype(np.float64)
            valid_pred_population = pred_population_map[valid_population_mask].astype(np.float64)

            sum_pred_population = float(valid_pred_population.sum())
            sum_true_population = float(valid_true_population.sum())
            sum_pred_population_sq = float(np.sum(valid_pred_population * valid_pred_population))
            sum_pred_true_population = float(np.sum(valid_pred_population * valid_true_population))
            valid_pixel_count = float(valid_pred_population.size)

            stats[target_hour]["sum_pred"] += sum_pred_population
            stats[target_hour]["sum_true"] += sum_true_population
            stats[target_hour]["sum_pred2"] += sum_pred_population_sq
            stats[target_hour]["sum_pred_true"] += sum_pred_true_population
            stats[target_hour]["count"] += valid_pixel_count

    hourly_linear_calibration: dict[int, tuple[float, float]] = {}
    for hour_idx, hour_stats in stats.items():
        if hour_stats["count"] <= 0 or hour_stats["sum_pred2"] <= eps:
            hourly_linear_calibration[hour_idx] = (1.0, 0.0)
            continue
        mean_pred_population = hour_stats["sum_pred"] / hour_stats["count"]
        mean_true_population = hour_stats["sum_true"] / hour_stats["count"]
        # Least-squares line fit: true = a * pred + b
        calibration_slope = (hour_stats["sum_pred_true"] - hour_stats["count"] * mean_pred_population * mean_true_population) / max(
            hour_stats["sum_pred2"] - hour_stats["count"] * mean_pred_population * mean_pred_population, eps
        )
        calibration_intercept = mean_true_population - calibration_slope * mean_pred_population
        hourly_linear_calibration[hour_idx] = (float(calibration_slope), float(calibration_intercept))
    return hourly_linear_calibration

def split_into_patches(
        input_sequence: np.ndarray, patch_size: int = 30, overlap: int = 10
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Split (time, H, W, C) into overlapping spatial patches."""
    _, height, width, _ = input_sequence.shape
    if not patch_size:
        return [input_sequence], [(0, 0)]
    patches: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for row_start in _patch_start_indices(height, patch_size, overlap):
        for col_start in _patch_start_indices(width, patch_size, overlap):
            patch = input_sequence[:, row_start: row_start + patch_size, col_start: col_start + patch_size, :]
            patches.append(patch)
            origins.append((row_start, col_start))
    return patches, origins

def predict_full_image(
        model: nn.Module,
        X_full: np.ndarray,
        device: torch.device,
        patch_size: Optional[int],
        overlap: int,
) -> np.ndarray:
    """Predict probability map for a full spatial tile using optional patching."""
    model.eval()
    with torch.no_grad():
        if not patch_size:
            X_tensor = numpy_sequence_to_tensor(X_full).unsqueeze(0).to(device)
            pred = model(X_tensor).squeeze(0, 1).cpu().numpy()
            return pred

        patches, origins = split_into_patches(X_full, patch_size, overlap)
        t, H, W, _ = X_full.shape
        pred_canvas = np.zeros((H, W), dtype=np.float32)
        count_canvas = np.zeros((H, W), dtype=np.float32)
        for patch, (i, j) in zip(patches, origins):
            X_tensor = numpy_sequence_to_tensor(patch).unsqueeze(0).to(device)
            pred_patch = model(X_tensor).squeeze(0, 1).cpu().numpy()
            pred_canvas[i: i + patch_size, j: j + patch_size] += pred_patch
            count_canvas[i: i + patch_size, j: j + patch_size] += 1.0
        mask = count_canvas > 0
        pred_canvas[mask] /= count_canvas[mask]
        return pred_canvas

def save_predicted_map(pred_map: np.ndarray, reference_tif: str, output_tif: str) -> None:
    """Save predicted probability map as GeoTiff using reference metadata."""
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    profile.update({"count": 1, "dtype": "float32"})
    output_path = Path(output_tif)
    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError as exc:
            raise RuntimeError(
                f"Cannot overwrite {output_path}. Close any application using this file and retry."
            ) from exc
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(pred_map.astype(np.float32), 1)
    print(f"Saved predicted map to {output_tif}")

def predict_one_day(
        model: nn.Module,
        base_dataset: HourlyPopulationDataset,
        device: torch.device,
        target_date_str: str,
        output_dir: Path,
        patch_size: int = 96,# Patch-based large-image prediction.
        overlap: int = 32,
        hourly_calib: Optional[dict[int, tuple[float, float]]] = None,
) -> dict:
    """Predict one day and return auditable end-to-end and per-hour runtime records."""
    synchronize_device(device)
    day_started = perf_counter()
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    model.eval()

    # Find all sample indices belonging to the target date.
    selected: list[tuple[int, datetime]] = []
    for idx, sample in enumerate(base_dataset.samples):
        key_y = sample["key_y"]
        meta = base_dataset._meta_by_key[key_y]
        dt = meta["dt"]
        if dt.date() == target_date:
            selected.append((idx, dt))

    if not selected:
        print(f"[warn] No samples found for target date {target_date_str}.")
        return {
            "target_date": target_date_str,
            "status": "no_samples",
            "predicted_hours": 0,
            "end_to_end_seconds": 0.0,
            "model_inference_seconds": 0.0,
            "hourly_records": [],
        }

    selected.sort(key=lambda x: x[1].hour)
    print(f"[info] {target_date_str} has {len(selected)} time-window samples for prediction.")

    pop_sqrt_min = getattr(base_dataset, "pop_sqrt_min", 0.0)
    pop_sqrt_max = getattr(base_dataset, "pop_sqrt_max", 1.0)
    mask = getattr(base_dataset, "study_mask_full", None)
    hourly_records: list[dict] = []

    for sample_idx, dt in selected:
        synchronize_device(device)
        hour_started = perf_counter()
        hour = dt.hour
        load_started = perf_counter()
        X_full, _ = base_dataset.get_full_sequence(sample_idx)  # (T, H, W, C)
        input_load_seconds = perf_counter() - load_started

        print(f"Processing sample {sample_idx}, hour {hour:02d}: input shape {X_full.shape}")

        # Predict the large image with patch-based inference.
        synchronize_device(device)
        inference_started = perf_counter()
        pred_sqrt = predict_full_image(
            model=model,
            X_full=X_full,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
        )
        synchronize_device(device)
        model_inference_seconds = perf_counter() - inference_started

        # Convert to population before calibration, then convert back to sqrt(pop) for output.
        output_started = perf_counter()
        pred_pop = np.square(np.maximum(pred_sqrt, 0.0))
        pred_pop = apply_hourly_linear_calibration(pred_pop, hour, hourly_calib)
        pred_sqrt = np.sqrt(pred_pop)

        key_y = base_dataset.samples[sample_idx]["key_y"]
        ref_tif = base_dataset.pop_path_map[key_y]

        out_tif = output_dir / f"PredPop_{target_date.strftime('%Y%m%d')}_{hour:02d}_real.tif"
        save_real_population_map(
            sqrt_map=pred_sqrt,
            reference_tif=ref_tif,
            output_tif=str(out_tif),
            mask=mask,
        )
        postprocess_and_write_seconds = perf_counter() - output_started
        synchronize_device(device)
        hour_end_to_end_seconds = perf_counter() - hour_started
        patch_count = (
            len(_patch_start_indices(X_full.shape[1], patch_size, overlap))
            * len(_patch_start_indices(X_full.shape[2], patch_size, overlap))
            if patch_size
            else 1
        )
        hourly_records.append(
            {
                "hour": int(hour),
                "sample_index": int(sample_idx),
                "input_time_steps": int(X_full.shape[0]),
                "height": int(X_full.shape[1]),
                "width": int(X_full.shape[2]),
                "channels": int(X_full.shape[3]),
                "patch_count": int(patch_count),
                "input_load_seconds": float(input_load_seconds),
                "model_inference_seconds": float(model_inference_seconds),
                "postprocess_and_write_seconds": float(postprocess_and_write_seconds),
                "end_to_end_seconds": float(hour_end_to_end_seconds),
            }
        )
        print(f"[ok] Saved predicted population map for {target_date_str} {hour:02d}:00 -> {out_tif}")

    synchronize_device(device)
    return {
        "target_date": target_date_str,
        "status": "ok",
        "predicted_hours": len(hourly_records),
        "end_to_end_seconds": float(perf_counter() - day_started),
        "model_inference_seconds": float(sum(row["model_inference_seconds"] for row in hourly_records)),
        "input_load_seconds": float(sum(row["input_load_seconds"] for row in hourly_records)),
        "postprocess_and_write_seconds": float(
            sum(row["postprocess_and_write_seconds"] for row in hourly_records)
        ),
        "hourly_records": hourly_records,
    }

def predict_dates_to_root(
        model: nn.Module,
        base_dataset: HourlyPopulationDataset,
        device: torch.device,
        target_dates: Sequence[str],
        output_root: Path,
        patch_size: int = DEFAULT_PREDICTION_PATCH_SIZE,
        overlap: int = DEFAULT_PREDICTION_OVERLAP,
        hourly_calib: Optional[dict[int, tuple[float, float]]] = None,
) -> list[dict]:
    """Predict multiple dates and return one runtime record per date."""
    runtime_records: list[dict] = []
    for target_date_str in target_dates:
        target_date_tag = target_date_str.replace("-", "")
        pred_dir = output_root / f"Pred_{target_date_tag}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        runtime_records.append(predict_one_day(
            model=model,
            base_dataset=base_dataset,
            device=device,
            target_date_str=target_date_str,
            output_dir=pred_dir,
            patch_size=patch_size,
            overlap=overlap,
            hourly_calib=hourly_calib,
        ))
    return runtime_records
