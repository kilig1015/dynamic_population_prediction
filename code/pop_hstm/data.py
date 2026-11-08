from __future__ import annotations

import glob
import json
import os
import random
import re
from collections import OrderedDict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Sequence

import fiona
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import rotate
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ChronologicalSplit, DEFAULT_RANDOM_SEED, DEFAULT_STATIC_FILES
from .utils import (
    _patch_start_indices,
    min_max_scaler,
    numpy_sequence_to_tensor,
    numpy_target_to_tensor,
)

def build_hourly_population_dataset(
        project_root: Path,
        window_size: int,
        patch_size: int,
        overlap: int,
        static_files: Sequence[str] = DEFAULT_STATIC_FILES,
        random_seed: int = DEFAULT_RANDOM_SEED,
) -> "HourlyPopulationDataset":
    """Create the project's default hourly dataset."""
    return HourlyPopulationDataset(
        pop_dir=str(project_root / "data" / "mobile_phone_data"),
        static_dir=str(project_root / "data" / "static_variables"),
        static_files=list(static_files),
        window_size=window_size,
        patch_size=patch_size,
        overlap=overlap,
        rotation_angle=90,
        do_flip=True,
        random_seed=random_seed,
        target_weekdays=None,
        shape_path=str(project_root / "data" / "shapefiles" / "core_urban_area_Beijing.shp"),
    )

def chronological_split_indices(
        dataset: "HourlyPopulationDataset",
        included_months: Sequence[int] = (9,),
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
) -> ChronologicalSplit:
    """Split samples by target date using chronological boundaries."""
    combined_indices: List[int] = []
    combined_dates: List[date] = []

    for idx, sample in enumerate(dataset.samples):
        key_y = sample["key_y"]
        dt = dataset._meta_by_key[key_y]["dt"]
        if dt.month in included_months:
            combined_indices.append(idx)
            combined_dates.append(dt.date())

    if not combined_dates:
        raise RuntimeError(f"No samples found for months={tuple(included_months)}.")

    unique_dates = sorted(set(combined_dates))
    total_days = len(unique_dates)
    train_days = int(total_days * train_ratio)
    val_days = int(total_days * val_ratio)
    train_end_day = unique_dates[train_days - 1] if train_days > 0 else unique_dates[0]
    val_end_day = unique_dates[train_days + val_days - 1] if val_days > 0 else unique_dates[-1]

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    for idx, dt in zip(combined_indices, combined_dates):
        if dt <= train_end_day:
            train_indices.append(idx)
        elif dt <= val_end_day:
            val_indices.append(idx)
        else:
            test_indices.append(idx)

    return ChronologicalSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        unique_dates=list(unique_dates),
        train_end_day=train_end_day,
        val_end_day=val_end_day,
        combined_indices=combined_indices,
        combined_dates=combined_dates,
    )

def print_split_summary(split: ChronologicalSplit) -> None:
    """Print chronological split ranges for quick inspection."""
    unique_dates = split.unique_dates
    if not unique_dates:
        return

    train_days = len({d for d in split.combined_dates if d <= split.train_end_day})
    val_days = len({d for d in split.combined_dates if split.train_end_day < d <= split.val_end_day})

    print(f"Dates used for split: {[d.strftime('%Y-%m-%d') for d in unique_dates]}")
    print(f"Total samples (time windows): {len(split.combined_indices)}")
    print(f"Train samples (60%): {len(split.train_indices)}")
    print(f"Val   samples (20%): {len(split.val_indices)}")
    print(f"Test  samples (20%): {len(split.test_indices)}")
    print(f"Train date range: {unique_dates[0].strftime('%Y-%m-%d')} to {split.train_end_day.strftime('%Y-%m-%d')}")
    if train_days < len(unique_dates):
        print(f"Val date range: {unique_dates[train_days].strftime('%Y-%m-%d')} to {split.val_end_day.strftime('%Y-%m-%d')}")
    if train_days + val_days < len(unique_dates):
        print(
            f"Test date range: {unique_dates[train_days + val_days].strftime('%Y-%m-%d')} "
            f"to {unique_dates[-1].strftime('%Y-%m-%d')}"
        )

def collect_keys_from_indices(dataset: "HourlyPopulationDataset", sample_indices: Sequence[int]) -> set[str]:
    """Collect all sequence and target keys touched by a set of sample indices."""
    keys: set[str] = set()
    for idx in sample_indices:
        sample = dataset.samples[idx]
        keys.update(sample["key_seq"])
        keys.add(sample["key_y"])
    return keys

def recompute_norm_with_indices(dataset: "HourlyPopulationDataset", sample_indices: Sequence[int]) -> None:
    """Recompute normalization statistics using only the provided sample subset."""
    dataset.recompute_pop_norm(keys=collect_keys_from_indices(dataset, sample_indices))

class HourlyPopulationDataset:
    """
    Hourly-resolution population dataset.
    - Temporal dimension: consecutive hourly steps parsed from mobile-signal raster filenames.
    - Per-time-step channels:
        [pop_t, static_factors..., hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, pop_t_minus_1week]
    - Samples: use the past `window_size` hours to predict next-hour population (regression).
    """

    def __init__(
            self,
            pop_dir: str,
            static_dir: str,
            static_files: Sequence[str],
            window_size: int = 24,
            patch_size: Optional[int] = 30,
            overlap: int = 10,
            rotation_angle: int = 0,
            do_flip: bool = True,
            random_seed: int = 42,
            target_weekdays: Optional[Sequence[int]] = None,
            shape_path: Optional[str] = None,
            pop_norm_method: str = "zscore",
    ) -> None:
        """
        Parameters:
        - pop_dir: directory containing mobile-signal population rasters, expected like pointMMDDHH_krigin_result.tif
        - static_dir: directory containing static-factor rasters
        - static_files: static-factor file names relative to `static_dir`
        - window_size: input sequence length in hours, e.g. 24 / 72 / 168
        - patch_size/overlap/rotation_angle/do_flip: same role as in the original CityGrowthDataset
        - target_weekdays: optionally keep only samples for selected weekdays (0=Mon,...,6=Sun),
                           or use all time steps when set to None
        """
        self.pop_dir = pop_dir
        self.static_dir = static_dir
        self.static_files = list(static_files)
        self.window_size = window_size
        self.patch_size = patch_size
        self.overlap = overlap
        self.rotation_angle = rotation_angle
        self.do_flip = do_flip
        self.target_weekdays = set(target_weekdays) if target_weekdays is not None else None
        self.shape_path = shape_path
        self.study_mask_full: Optional[np.ndarray] = None
        self.pop_norm_method = pop_norm_method.lower()
        if self.pop_norm_method not in {"zscore", "minmax"}:
            raise ValueError(f"Unsupported pop_norm_method: {pop_norm_method}")
        self.pop_sqrt_min = 0.0
        self.pop_sqrt_max = 1.0
        self.pop_sqrt_mean = 0.0
        self.pop_sqrt_std = 1.0
        self._stats_filename = f"pop_norm_stats_{self.pop_norm_method}.json"

        np.random.seed(random_seed)

        # Prepare all time-step metadata first (timestamps, path mapping, etc.).
        self.time_meta, self.pop_path_map = self._load_all_data()
        self._meta_by_key = {m["key"]: m for m in self.time_meta}
        # Keep caches disabled/minimal and load on demand to reduce memory usage.
        self._pop_cache_limit = 0
        self._feat_cache_limit = 0
        self._pop_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._feat_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._patch_origins: Optional[list[tuple[int, int]]] = None
        self._patch_shape: Optional[tuple[int, int]] = None

        # Build sliding-window samples from the time series after metadata is ready.
        self.samples = self._create_time_window_samples()

    # ---------- Internal utilities ----------

    @staticmethod
    def _parse_pop_filename(path: str) -> datetime:
        """
        Parse time from a filename such as point090400_krigin_result.tif -> 2017-09-04 00:00.
        Here 09=month, 04=day, 00=hour, and the year is fixed to 2017.
        """
        basename = os.path.basename(path)
        # Filenames may have different suffixes; only the pointMMDDHH token is required.
        match = re.search(r"point(\d{6})", basename)
        if not match:
            raise ValueError(f"Unable to resolve timestamp from filename: {basename}")
        mmddhh = match.group(1)  # "MMDDHH"
        ts_str = "2017" + mmddhh
        return datetime.strptime(ts_str, "%Y%m%d%H")

    def _load_static_rasters(self) -> np.ndarray:
        """
        Load all static-factor rasters, apply min-max normalization, and stack them as (H, W, C_static).
        All static rasters must share the same spatial extent and resolution.
        """
        static_arrays: list[np.ndarray] = []
        ref_shape = None

        for fname in self.static_files:
            tif_path = os.path.join(self.static_dir, fname)
            if not os.path.exists(tif_path):
                raise FileNotFoundError(f"The static factor file does not exist: {tif_path}")
            with rasterio.open(tif_path) as src:
                arr = src.read(1).astype(np.float32)
            arr = min_max_scaler(arr)
            if ref_shape is None:
                ref_shape = arr.shape
            else:
                if arr.shape != ref_shape:
                    raise ValueError(
                        f"The static factor rasters have inconsistent shapes: {fname}, shape={arr.shape}, ref={ref_shape}"
                    )
            static_arrays.append(arr)

        if not static_arrays:
            raise ValueError("No static factor rasters provided.")

        static_stack = np.stack(static_arrays, axis=-1)  # (H, W, C_static)
        return static_stack

    def _load_all_data(self) -> tuple[list[dict], dict[str, str]]:
        """
        Load all hourly population rasters and create ordered time metadata plus a key->path map.
        Returns:
        - time_meta: a time-ordered metadata list; each item contains timestamp-related fields
            {"dt": datetime, "hour": int, "weekday": int, "is_weekend": int, "key": str}
        - pop_path_map: key -> tif path
        """
        pop_tifs = glob.glob(os.path.join(self.pop_dir, "*.tif"))
        if not pop_tifs:
            raise RuntimeError(f"No .tif population rasters found in {self.pop_dir}.")

        stats_path = os.path.join(self.pop_dir, self._stats_filename)
        loaded_stats = False
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                method = stats.get("method", "minmax")
                if method == self.pop_norm_method:
                    self.pop_sqrt_min = float(stats.get("pop_sqrt_min", 0.0))
                    self.pop_sqrt_max = float(stats.get("pop_sqrt_max", 1.0))
                    self.pop_sqrt_mean = float(stats.get("pop_sqrt_mean", 0.0))
                    self.pop_sqrt_std = float(stats.get("pop_sqrt_std", 1.0))
                    loaded_stats = True
                    print(f"[info] Loaded pop norm stats from {stats_path}: "
                          f"method={method}, min={self.pop_sqrt_min}, max={self.pop_sqrt_max}, "
                          f"mean={self.pop_sqrt_mean}, std={self.pop_sqrt_std}")
                else:
                    print(f"[warn] Stats file method={method} != expected {self.pop_norm_method}, recomputing...")
            except Exception:
                print(f"[warn] Failed to read normalization stats, switching to scanning all population rasters: {stats_path}")

        if not loaded_stats:
            global_sqrt_min = float("inf")
            global_sqrt_max = -float("inf")
            total_pixels = 0.0
            sum_val = 0.0
            sum_sq = 0.0
            for p in pop_tifs:
                with rasterio.open(p) as src:
                    arr = src.read(1).astype(np.float32)
                if arr.size == 0:
                    continue
                arr = np.where(arr < 0, 0.0, arr)
                sqrt_arr = np.sqrt(arr)
                vmin = float(np.min(sqrt_arr))
                vmax = float(np.max(sqrt_arr))
                if vmin < global_sqrt_min:
                    global_sqrt_min = vmin
                if vmax > global_sqrt_max:
                    global_sqrt_max = vmax
                sum_val += float(np.sum(sqrt_arr))
                sum_sq += float(np.sum(sqrt_arr ** 2))
                total_pixels += float(sqrt_arr.size)
            if not np.isfinite(global_sqrt_min) or not np.isfinite(global_sqrt_max):
                raise RuntimeError("Unable to statistics valid global sqrt value range from population rasters.")
            if total_pixels <= 0:
                raise RuntimeError("Population raster pixel count is 0, unable to calculate mean/variance.")
            mean_val = sum_val / total_pixels
            var_val = max(sum_sq / total_pixels - mean_val ** 2, 0.0)
            std_val = float(np.sqrt(var_val))
            if std_val <= 0:
                std_val = 1.0
            self.pop_sqrt_min = global_sqrt_min
            self.pop_sqrt_max = global_sqrt_max
            self.pop_sqrt_mean = mean_val
            self.pop_sqrt_std = std_val

            try:
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "method": self.pop_norm_method,
                            "pop_sqrt_min": self.pop_sqrt_min,
                            "pop_sqrt_max": self.pop_sqrt_max,
                            "pop_sqrt_mean": self.pop_sqrt_mean,
                            "pop_sqrt_std": self.pop_sqrt_std,
                            "source_files": len(pop_tifs),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                print(f"[warn] Failed to write normalization stats file: {stats_path}")

        tmp_records = []
        for path in pop_tifs:
            dt = self._parse_pop_filename(path)
            key = dt.strftime("%Y%m%d%H")
            tmp_records.append({"dt": dt, "key": key, "path": path})

        tmp_records.sort(key=lambda x: x["dt"])
        pop_path_map = {item["key"]: item["path"] for item in tmp_records}

        # Static-factor rasters are loaded once and reused later on demand.
        self.static_stack = self._load_static_rasters()
        self._grid_shape = self.static_stack.shape[:2]

        # If a study-area shapefile is provided, rasterize it to a mask and clip static factors.
        if self.shape_path is not None:
            self.study_mask_full = self._build_study_mask_from_shapefile()
            # Set static factors outside the study area to 0.
            self.static_stack = self.static_stack * self.study_mask_full[..., np.newaxis]
        else:
            self.study_mask_full = None

        time_meta: list[dict] = []
        for item in tmp_records:
            dt = item["dt"]
            hour = dt.hour
            dow = dt.weekday()
            is_weekend = int(dow >= 5)
            time_meta.append(
                {
                    "dt": dt,
                    "hour": hour,
                    "weekday": dow,
                    "is_weekend": is_weekend,
                    "key": item["key"],
                }
            )

        return time_meta, pop_path_map

    def _build_study_mask_from_shapefile(self) -> np.ndarray:
        """Rasterize the study-area shapefile to a mask (H, W), with 1 inside and 0 outside."""
        if self.shape_path is None:
            raise RuntimeError("shape_path is not set, but _build_study_mask_from_shapefile was called.")
        if not self.static_files:
            raise RuntimeError("static_files is empty, unable to determine reference raster.")

        ref_tif = os.path.join(self.static_dir, self.static_files[0])
        if not os.path.exists(ref_tif):
            raise FileNotFoundError(f"Reference static factor does not exist: {ref_tif}")

        with rasterio.open(ref_tif) as src:
            height, width = src.height, src.width
            transform = src.transform

        with fiona.open(self.shape_path, "r") as shp:
            shapes = [(feat["geometry"], 1) for feat in shp]

        mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        mask_f = mask.astype(np.float32)
        print(f"[info] Study-area mask generated from {self.shape_path}, shape={mask_f.shape}")
        return mask_f

    def recompute_pop_norm(
            self, keys: Sequence[str], save_path: Optional[str] = None
    ) -> tuple[float, float]:
        """
        Recompute sqrt(pop) min/max/mean/std using the provided keys, and optionally save to JSON.
        This is intended for train-only statistics to avoid validation/test leakage.
        """
        if not keys:
            raise ValueError("recompute_pop_norm requires at least one key.")
        sqrt_min = float("inf")
        sqrt_max = -float("inf")
        sum_val = 0.0
        sum_sq = 0.0
        total_pixels = 0.0
        for key in keys:
            if key not in self.pop_path_map:
                raise KeyError(f"Population raster not found for key={key}")
            arr = self._get_pop_array(key)
            arr = np.where(arr < 0, 0.0, arr)
            sqrt_arr = np.sqrt(arr)
            vmin = float(np.min(sqrt_arr))
            vmax = float(np.max(sqrt_arr))
            if vmin < sqrt_min:
                sqrt_min = vmin
            if vmax > sqrt_max:
                sqrt_max = vmax
            sum_val += float(np.sum(sqrt_arr))
            sum_sq += float(np.sum(sqrt_arr ** 2))
            total_pixels += float(sqrt_arr.size)
        if not np.isfinite(sqrt_min) or not np.isfinite(sqrt_max):
            raise RuntimeError("Unable to statistics valid global sqrt value range from provided keys.")
        if total_pixels <= 0:
            raise RuntimeError("Population raster pixel count is 0, unable to calculate mean/variance.")
        mean_val = sum_val / total_pixels
        var_val = max(sum_sq / total_pixels - mean_val ** 2, 0.0)
        std_val = float(np.sqrt(var_val))
        if std_val <= 0:
            std_val = 1.0

        self.pop_sqrt_min = sqrt_min
        self.pop_sqrt_max = sqrt_max
        self.pop_sqrt_mean = mean_val
        self.pop_sqrt_std = std_val
        self._feat_cache.clear()

        save_path = save_path or os.path.join(self.pop_dir, self._stats_filename)
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "method": self.pop_norm_method,
                        "pop_sqrt_min": self.pop_sqrt_min,
                        "pop_sqrt_max": self.pop_sqrt_max,
                        "pop_sqrt_mean": self.pop_sqrt_mean,
                        "pop_sqrt_std": self.pop_sqrt_std,
                        "source_keys": len(keys),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(
                f"[info] Recomputed pop norm stats from {len(keys)} keys -> min={sqrt_min}, max={sqrt_max}, mean={mean_val}, std={std_val}, saved to {save_path}")
        except Exception:
            print(f"[warn] Failed to write normalization stats file: {save_path}")
        return sqrt_min, sqrt_max

    def _evict_cache_if_needed(
            self, cache: OrderedDict[str, np.ndarray], limit: int
    ) -> None:
        while len(cache) > limit:
            cache.popitem(last=False)

    def _normalize_pop(self, sqrt_arr: np.ndarray) -> np.ndarray:
        """Normalize sqrt(pop) using either z-score or min-max scaling."""
        if self.pop_norm_method == "zscore":
            std = self.pop_sqrt_std if self.pop_sqrt_std != 0 else 1.0
            return (sqrt_arr - self.pop_sqrt_mean) / std
        denom = (self.pop_sqrt_max - self.pop_sqrt_min) if (self.pop_sqrt_max - self.pop_sqrt_min) != 0 else 1.0
        return (sqrt_arr - self.pop_sqrt_min) / denom

    def _get_pop_array(self, key: str) -> np.ndarray:
        """Load a population raster on demand with a small LRU cache to reduce IO and memory."""
        if key in self._pop_cache:
            arr = self._pop_cache.pop(key)
            self._pop_cache[key] = arr  # move to end 
            return arr
        if key not in self.pop_path_map:
            raise KeyError(f"Population raster not found: {key}")
        tif_path = self.pop_path_map[key]
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
        self._pop_cache[key] = arr
        self._evict_cache_if_needed(self._pop_cache, self._pop_cache_limit)
        return arr

    def _get_feature_cube(self, key: str) -> np.ndarray:
        """Build the feature raster (H, W, C) for a specific time step."""
        if key in self._feat_cache:
            feat = self._feat_cache.pop(key)
            self._feat_cache[key] = feat
            return feat

        meta = self._meta_by_key.get(key)
        if meta is None:
            raise KeyError(f"Time metadata not found: {key}")
        arr_pop = self._get_pop_array(key)
        # If a study-area mask exists, clip the population raster to the study area as well.
        if self.study_mask_full is not None:
            arr_pop = arr_pop * self.study_mask_full
        H, W = self._grid_shape
        if arr_pop.shape != (H, W):
            raise ValueError(
                f"Population raster size is inconsistent with static factors: key={key}, pop_shape={arr_pop.shape}, static_shape={(H, W)}"
            )
        # Apply sqrt transform and normalization.
        arr_pop = np.where(arr_pop < 0, 0.0, arr_pop)
        sqrt_pop = np.sqrt(arr_pop)
        pop_norm = self._normalize_pop(sqrt_pop)

        hour = meta["hour"]
        dow = meta["weekday"]
        is_weekend = float(meta["is_weekend"])

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin = np.sin(2 * np.pi * dow / 7.0)
        dow_cos = np.cos(2 * np.pi * dow / 7.0)
        time_chs = [
            np.full((H, W), hour_sin, dtype=np.float32),
            np.full((H, W), hour_cos, dtype=np.float32),
            np.full((H, W), dow_sin, dtype=np.float32),
            np.full((H, W), dow_cos, dtype=np.float32),
            np.full((H, W), is_weekend, dtype=np.float32),
        ]
        time_stack = np.stack(time_chs, axis=-1)
        if self.study_mask_full is not None:
            time_stack = time_stack * self.study_mask_full[..., np.newaxis]

        dt_prev_week = meta["dt"] - timedelta(days=7)
        prev_key = dt_prev_week.strftime("%Y%m%d%H")
        if prev_key in self.pop_path_map:
            prev_pop_arr = self._get_pop_array(prev_key)
            if self.study_mask_full is not None:
                prev_pop_arr = prev_pop_arr * self.study_mask_full
            if prev_pop_arr.shape != (H, W):
                raise ValueError(
                    f"Previous week's population raster size is inconsistent: key={prev_key}, pop_shape={prev_pop_arr.shape}, static_shape={(H, W)}"
                )
            prev_pop_arr = np.where(prev_pop_arr < 0, 0.0, prev_pop_arr)
            sqrt_prev = np.sqrt(prev_pop_arr)
            prev_pop = self._normalize_pop(sqrt_prev)
        else:
            prev_pop = np.zeros((H, W), dtype=np.float32)
        prev_pop = prev_pop[..., np.newaxis]

        feat = np.concatenate(
            [
                pop_norm[..., np.newaxis],
                self.static_stack,
                time_stack,
                prev_pop,
            ],
            axis=-1,
        )
        self._feat_cache[key] = feat
        self._evict_cache_if_needed(self._feat_cache, self._feat_cache_limit)
        return feat

    def _create_time_window_samples(self) -> list[dict[str, Any]]:
        """
        Build time-series samples from consecutive hourly steps while storing only keys rather than full rasters,
        which keeps memory usage manageable before raster materialization.
        """
        samples: list[dict[str, Any]] = []

        keys = [m["key"] for m in self.time_meta]
        weekdays = [m["weekday"] for m in self.time_meta]

        for idx in range(self.window_size, len(keys)):
            target_weekday = weekdays[idx]
            if self.target_weekdays is not None and target_weekday not in self.target_weekdays:
                continue

            key_y = keys[idx]
            key_x_seq = tuple(keys[idx - self.window_size: idx])

            samples.append({"key_seq": key_x_seq, "key_y": key_y})

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_full_sequence(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Build the full-frame (X, Y) lazily; only used when patch_size is None."""
        sample = self.samples[idx]
        key_seq: Sequence[str] = sample["key_seq"]
        key_y: str = sample["key_y"]
        X = np.stack([self._get_feature_cube(k) for k in key_seq], axis=0).astype(np.float32, copy=False)
        y_pop = self._get_pop_array(key_y)
        if self.study_mask_full is not None:
            y_pop = y_pop * self.study_mask_full
        y_pop = np.where(y_pop < 0, 0.0, y_pop)
        Y = np.sqrt(y_pop)[..., np.newaxis]
        return X, Y

    def get_patch_origins(self) -> list[tuple[int, int]]:
        """Return all patch top-left coordinates for the patch dataset."""
        if self._patch_origins is not None:
            return self._patch_origins
        H, W = self._grid_shape
        if not self.patch_size:
            self._patch_shape = (H, W)
            self._patch_origins = [(0, 0)]
            return self._patch_origins
        if self.patch_size > H or self.patch_size > W:
            raise ValueError(
                f"patch_size={self.patch_size} extend the grid range: {(H, W)}"
            )
        origins: list[tuple[int, int]] = []
        for i in _patch_start_indices(H, self.patch_size, self.overlap):
            for j in _patch_start_indices(W, self.patch_size, self.overlap):
                origins.append((i, j))
        if not origins:
            origins = [(0, 0)]
        self._patch_shape = (self.patch_size, self.patch_size)
        self._patch_origins = origins
        return self._patch_origins

    def num_patches_per_sample(self) -> int:
        return len(self.get_patch_origins())

    def get_patch_by_index(self, sample_idx: int, patch_pos: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_patch, Y_patch) for a given patch index."""
        if sample_idx < 0 or sample_idx >= len(self.samples):
            raise IndexError("sample_idx out of range.")
        origins = self.get_patch_origins()
        if not origins:
            raise RuntimeError("Patch origins not defined.")
        if patch_pos < 0 or patch_pos >= len(origins):
            raise IndexError("patch_pos out of range.")

        sample = self.samples[sample_idx]
        top, left = origins[patch_pos]

        if not self.patch_size:
            return self.get_full_sequence(sample_idx)

        patch_size = self.patch_size
        X_seq: list[np.ndarray] = []
        for key in sample["key_seq"]:
            feat = self._get_feature_cube(key)
            patch = feat[top: top + patch_size, left: left + patch_size, :]
            X_seq.append(np.ascontiguousarray(patch))
        X_patch = np.stack(X_seq, axis=0)
        y_pop = self._get_pop_array(sample["key_y"])
        if self.study_mask_full is not None:
            y_pop = y_pop * self.study_mask_full
        y_pop = np.where(y_pop < 0, 0.0, y_pop)
        y_sqrt = np.sqrt(y_pop)
        Y_patch = np.ascontiguousarray(
            y_sqrt[top: top + patch_size, left: left + patch_size][..., np.newaxis]
        )
        return X_patch, Y_patch

    def _augment(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply random flip/rotation augmentation to the full temporal block."""
        X_aug = X.copy()
        Y_aug = Y.copy()
        if self.do_flip:
            if np.random.rand() < 0.5:
                X_aug = np.flip(X_aug, axis=1)  # flip vertical
                Y_aug = np.flip(Y_aug, axis=0)
            if np.random.rand() < 0.5:
                X_aug = np.flip(X_aug, axis=2)  # flip horizontal
                Y_aug = np.flip(Y_aug, axis=1)

        if self.rotation_angle and self.rotation_angle > 0:
            possible_angles = np.arange(0, 360, self.rotation_angle)
            angle = np.random.choice(possible_angles)

            rotated_X = []
            for t in range(X_aug.shape[0]):
                x_t = X_aug[t]
                mask = np.ones(x_t.shape[:2], dtype=np.float32)
                x_rot = rotate(
                    x_t,
                    angle=angle,
                    axes=(0, 1),
                    reshape=False,
                    order=1,
                    mode="constant",
                    cval=0,
                )
                mask_rot = rotate(
                    mask,
                    angle=angle,
                    axes=(0, 1),
                    reshape=False,
                    order=0,
                    mode="constant",
                    cval=0,
                )
                rotated_X.append(x_rot * mask_rot[..., np.newaxis])
            X_aug = np.stack(rotated_X, axis=0)

            mask_y = np.ones(Y_aug.shape[:2], dtype=np.float32)
            y_rot = rotate(
                Y_aug,
                angle=angle,
                axes=(0, 1),
                reshape=False,
                order=1,
                mode="constant",
                cval=0,
            )
            mask_y_rot = rotate(
                mask_y,
                angle=angle,
                axes=(0, 1),
                reshape=False,
                order=0,
                mode="constant",
                cval=0,
            )
            Y_aug = y_rot * mask_y_rot[..., np.newaxis]
        return X_aug, Y_aug

    def _patch_split(
            self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Split the full frame into spatial patches, consistent with the original patch logic."""
        if not self.patch_size:
            return [X], [Y]

        X_patches: list[np.ndarray] = []
        Y_patches: list[np.ndarray] = []
        _, H, W, _ = X.shape
        for i in _patch_start_indices(H, self.patch_size, self.overlap):
            for j in _patch_start_indices(W, self.patch_size, self.overlap):
                X_patch = X[:, i: i + self.patch_size, j: j + self.patch_size, :]
                Y_patch = Y[i: i + self.patch_size, j: j + self.patch_size, :]
                X_patches.append(X_patch)
                Y_patches.append(Y_patch)
        return X_patches, Y_patches

class HourlyPopulationPatchDataset(Dataset):
    """Flattened patch dataset derived from HourlyPopulationDataset.

    To prevent the sample count from exploding, each time window draws only a limited number of random patches:
    - dataset length = number of time-window samples (instead of time-window count * patch count)
    - each __getitem__ call randomly chooses one patch position
    This greatly reduces the number of batches per epoch and speeds up training.
    """

    def __init__(
            self,
            base_dataset: HourlyPopulationDataset,
            augment: bool = False,
            sample_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.augment = augment
        if sample_indices is None:
            self.sample_indices = list(range(len(self.base_dataset)))
        else:
            self.sample_indices = list(sample_indices)

        # All possible patch top-left coordinates.
        self.patch_origins = list(self.base_dataset.get_patch_origins())
        if len(self.patch_origins) == 0:
            raise ValueError("Failed to generate patch coordinates, please check patch_size/overlap settings.")

        # Key design: draw only one random patch for each sample per access.
        # Tunable parameter 1: patch-sample budget for each time window.
        self.patches_per_sample=14
        self.total_samples = len(self.sample_indices)

    def __len__(self) -> int:
        return self.total_samples * self.patches_per_sample

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Patch index out of bounds.")
        # Each idx corresponds to one time-window sample.
        sample_idx = self.sample_indices[idx // self.patches_per_sample]
        # Randomly select one location among all candidate patches for that sample.
        patch_offset = random.randrange(len(self.patch_origins))
        X_patch, Y_patch = self.base_dataset.get_patch_by_index(sample_idx, patch_offset)
        if self.augment:
            X_patch, Y_patch = self.base_dataset._augment(X_patch, Y_patch)
        X_tensor = numpy_sequence_to_tensor(X_patch)
        Y_tensor = numpy_target_to_tensor(Y_patch)
        return X_tensor, Y_tensor

def create_dataloader(
        base_dataset: HourlyPopulationDataset,
        batch_size: int,
        shuffle: bool,
        augment: bool,
        sample_indices: Optional[Sequence[int]] = None,
        patches_per_sample: int = 14,
        num_workers: int = 32,
) -> DataLoader:
    """Create a DataLoader from HourlyPopulationDataset."""
    patch_dataset = HourlyPopulationPatchDataset(
        base_dataset, augment=augment, sample_indices=sample_indices
    )
    patch_dataset.patches_per_sample = int(patches_per_sample)
    return DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
