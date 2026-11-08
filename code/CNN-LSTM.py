import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import glob
import random
import re
import time
import json
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal  # pylint: disable=unused-import
import rasterio
from rasterio.features import rasterize
import fiona
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import rotate
from torch.utils.data import DataLoader, Dataset
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('Agg')

os.environ["USE_PATH_FOR_GDAL_PYTHON"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def set_global_determinism(seed: int = 42) -> None:
    """Fix all RNG seeds (Python, NumPy, PyTorch) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def min_max_scaler(x: np.ndarray) -> np.ndarray:
    """Scale an array into [0, 1] range."""
    x = x.astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min + 1e-10)


def _patch_start_indices(length: int, patch_size: int, overlap: int) -> list[int]:
    """
    Compute start indices for sliding windows that fully cover the axis.
    Ensures the last patch is aligned to the end even when (length - patch_size)
    is not divisible by (patch_size - overlap).
    """
    if patch_size > length:
        raise ValueError(f"patch_size {patch_size} exceeds dimension {length}")
    step = max(1, patch_size - overlap)
    starts = list(range(0, max(1, length - patch_size + 1), step))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def numpy_sequence_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert (time, H, W, C) numpy array to torch tensor (time, C, H, W)."""
    permuted = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(permuted)).float()


def numpy_target_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert (H, W, 1) numpy array to torch tensor (1, H, W)."""
    permuted = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(permuted)).float()


# (a, b)
DEFAULT_HOURLY_CALIB: dict[int, tuple[float, float]] = {h: (1.0, 0.0) for h in range(24)}


class HourlyPopulationDataset:
    """-
-
[pop_t, ..., hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, pop_t_minus_1week]
- window_size"""

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
        """- pop_dir: pointMMDDHH_krigin_result.tif
- static_dir:
- static_files: static_dir
- window_size: 24 / 72 / 168
- patch_size/overlap/rotation_angle/do_flip: previous dataset implementation
- target_weekdays: weekday 0=,...,6=
None"""
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


        self.time_meta, self.pop_path_map = self._load_all_data()
        self._meta_by_key = {m["key"]: m for m in self.time_meta}

        self._pop_cache_limit = 0
        self._feat_cache_limit = 0
        self._pop_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._feat_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._patch_origins: Optional[list[tuple[int, int]]] = None
        self._patch_shape: Optional[tuple[int, int]] = None


        self.samples = self._create_time_window_samples()



    @staticmethod
    def _parse_pop_filename(path: str) -> datetime:
        """Parse pointMMDDHH token and return datetime."""
        basename = os.path.basename(path)
        # pointMMDDHH
        match = re.search(r"point(\d{6})", basename)
        if not match:
            raise ValueError(f"Failed to parse timestamp from filename: {basename}")
        mmddhh = match.group(1)  # "MMDDHH"
        ts_str = "2017" + mmddhh
        return datetime.strptime(ts_str, "%Y%m%d%H")

    def _load_static_rasters(self) -> np.ndarray:
        """min-max (H, W, C_static)"""
        static_arrays: list[np.ndarray] = []
        ref_shape = None

        for fname in self.static_files:
            tif_path = os.path.join(self.static_dir, fname)
            if not os.path.exists(tif_path):
                raise FileNotFoundError(f"Static factor file does not exist: {tif_path}")
            with rasterio.open(tif_path) as src:
                arr = src.read(1).astype(np.float32)
            arr = min_max_scaler(arr)
            if ref_shape is None:
                ref_shape = arr.shape
            else:
                if arr.shape != ref_shape:
                    raise ValueError(
                        f"Static-factor raster shape mismatch: {fname}, shape={arr.shape}, ref={ref_shape}"
                    )
            static_arrays.append(arr)

        if not static_arrays:
            raise ValueError("No static factor rasters were provided.")

        static_stack = np.stack(static_arrays, axis=-1)  # (H, W, C_static)
        return static_stack

    def _load_all_data(self) -> tuple[list[dict], dict[str, str]]:
        """key->

- time_meta: :
{"dt": datetime, "hour": int, "weekday": int, "is_weekend": int, "key": str}
- pop_path_map: key -> tif"""
        pop_tifs = glob.glob(os.path.join(self.pop_dir, "*.tif"))
        if not pop_tifs:
            raise RuntimeError("Runtime error in population forecasting pipeline.")

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
                print(f"[warn] Failed to load normalization stats, rescanning rasters: {stats_path}")

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
                raise RuntimeError("Runtime error in population forecasting pipeline.")
            if total_pixels <= 0:
                raise RuntimeError("Runtime error in population forecasting pipeline.")
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
                print(f"[warn] Failed to load normalization stats, rescanning rasters: {stats_path}")

        tmp_records = []
        for path in pop_tifs:
            dt = self._parse_pop_filename(path)
            key = dt.strftime("%Y%m%d%H")
            tmp_records.append({"dt": dt, "key": key, "path": path})

        tmp_records.sort(key=lambda x: x["dt"])
        pop_path_map = {item["key"]: item["path"] for item in tmp_records}


        self.static_stack = self._load_static_rasters()
        self._grid_shape = self.static_stack.shape[:2]

        # shapefile mask
        if self.shape_path is not None:
            self.study_mask_full = self._build_study_mask_from_shapefile()

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
        """shapefile mask (H, W)=1=0"""
        if self.shape_path is None:
            raise RuntimeError("Runtime error in population forecasting pipeline.")
        if not self.static_files:
            raise RuntimeError("Runtime error in population forecasting pipeline.")

        ref_tif = os.path.join(self.static_dir, self.static_files[0])
        if not os.path.exists(ref_tif):
            raise FileNotFoundError(f"Reference static factor raster does not exist: {ref_tif}")

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
        """key sqrt(pop) min/max/mean/std json"""
        if not keys:
            raise ValueError("recompute_pop_norm requires at least one key.")
        sqrt_min = float("inf")
        sqrt_max = -float("inf")
        sum_val = 0.0
        sum_sq = 0.0
        total_pixels = 0.0
        for key in keys:
            if key not in self.pop_path_map:
                raise KeyError(f"Missing population raster key={key}")
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
            raise RuntimeError("Runtime error in population forecasting pipeline.")
        if total_pixels <= 0:
            raise RuntimeError("Runtime error in population forecasting pipeline.")
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
            print(f"[warn] Failed to save normalization stats: {save_path}")
        return sqrt_min, sqrt_max

    def _evict_cache_if_needed(
            self, cache: OrderedDict[str, np.ndarray], limit: int
    ) -> None:
        while len(cache) > limit:
            cache.popitem(last=False)

    def _normalize_pop(self, sqrt_arr: np.ndarray) -> np.ndarray:
        """sqrt(pop) zscore minmax"""
        if self.pop_norm_method == "zscore":
            std = self.pop_sqrt_std if self.pop_sqrt_std != 0 else 1.0
            return (sqrt_arr - self.pop_sqrt_mean) / std
        denom = (self.pop_sqrt_max - self.pop_sqrt_min) if (self.pop_sqrt_max - self.pop_sqrt_min) != 0 else 1.0
        return (sqrt_arr - self.pop_sqrt_min) / denom

    def _get_pop_array(self, key: str) -> np.ndarray:
        """LRU IO"""
        if key in self._pop_cache:
            arr = self._pop_cache.pop(key)
            self._pop_cache[key] = arr  # move to end ()
            return arr
        if key not in self.pop_path_map:
            raise KeyError(f"Missing key: {key}")
        tif_path = self.pop_path_map[key]
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
        self._pop_cache[key] = arr
        self._evict_cache_if_needed(self._pop_cache, self._pop_cache_limit)
        return arr

    def _get_feature_cube(self, key: str) -> np.ndarray:
        """(H, W, C)"""
        if key in self._feat_cache:
            feat = self._feat_cache.pop(key)
            self._feat_cache[key] = feat
            return feat

        meta = self._meta_by_key.get(key)
        if meta is None:
            raise KeyError(f": {key}")
        arr_pop = self._get_pop_array(key)
        # mask
        if self.study_mask_full is not None:
            arr_pop = arr_pop * self.study_mask_full
        H, W = self._grid_shape
        if arr_pop.shape != (H, W):
            raise ValueError(
                f"Population/static shape mismatch: key={key}, pop_shape={arr_pop.shape}, static_shape={(H, W)}"
            )
        # sqrt
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
                    f"Previous-week population/static shape mismatch: key={prev_key}, pop_shape={prev_pop_arr.shape}, static_shape={(H, W)}"
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
        """- key key"""
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
        """(X, Y) patch_size=None"""
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
        """patch patch"""
        if self._patch_origins is not None:
            return self._patch_origins
        H, W = self._grid_shape
        if not self.patch_size:
            self._patch_shape = (H, W)
            self._patch_origins = [(0, 0)]
            return self._patch_origins
        if self.patch_size > H or self.patch_size > W:
            raise ValueError(
                f"patch_size={self.patch_size} exceeds raster size {(H, W)}"
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
        """patch (X_patch, Y_patch)"""
        if sample_idx < 0 or sample_idx >= len(self.samples):
            raise IndexError("sample_idx out of range.")
        origins = self.get_patch_origins()
        if not origins:
            raise RuntimeError("Runtime error in population forecasting pipeline.")
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
        """Population forecasting documentation."""
        X_aug = X.copy()
        Y_aug = Y.copy()
        if self.do_flip:
            if np.random.rand() < 0.5:
                X_aug = np.flip(X_aug, axis=1)
                Y_aug = np.flip(Y_aug, axis=0)
            if np.random.rand() < 0.5:
                X_aug = np.flip(X_aug, axis=2)
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
        """patch previous dataset implementation patch"""
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
    """HourlyPopulationDataset patch

** 8 patch**
- = patch
- __getitem__ patch
epoch batch"""

    def __init__(
            self,
            base_dataset: HourlyPopulationDataset,
            augment: bool = False,
            sample_indices: Optional[Sequence[int]] = None,
            **kwargs: Any,
    ) -> None:
        self.base_dataset = base_dataset
        self.augment = augment
        if sample_indices is None:
            self.sample_indices = list(range(len(self.base_dataset)))
        else:
            self.sample_indices = list(sample_indices)

        # patch
        self.patch_origins = list(self.base_dataset.get_patch_origins())
        if len(self.patch_origins) == 0:
            raise ValueError("Unable to generate patch coordinates. Check patch_size/overlap settings.")

        # 1 patch

        # self.patches_per_sample = 16
        # self.patches_per_sample = 12
        self.patches_per_sample = 6
        self.total_samples = len(self.sample_indices)

    def __len__(self) -> int:
        return self.total_samples * self.patches_per_sample

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range in population forecasting pipeline.")
        # idx
        sample_idx = self.sample_indices[idx // self.patches_per_sample]
        # patch
        patch_offset = random.randrange(len(self.patch_origins))
        X_patch, Y_patch = self.base_dataset.get_patch_by_index(sample_idx, patch_offset)
        if self.augment:
            X_patch, Y_patch = self.base_dataset._augment(X_patch, Y_patch)
        X_tensor = numpy_sequence_to_tensor(X_patch)
        Y_tensor = numpy_target_to_tensor(Y_patch)
        return X_tensor, Y_tensor

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution used by the DS blocks."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ReflectionPad2dTime(nn.Module):
    """Reflection padding that accepts 4D or 5D tensors."""

    def __init__(self, pad: tuple[int, int]) -> None:
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ph, pw = self.pad
        pad = (pw, pw, ph, ph)
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            reshaped = x.reshape(b * t, c, h, w)
            padded = F.pad(reshaped, pad, mode="reflect")
            new_h, new_w = padded.shape[-2:]
            return padded.view(b, t, c, new_h, new_w)
        if x.dim() == 4:
            return F.pad(x, pad, mode="reflect")
        raise ValueError("ReflectionPad2dTime expects 4D or 5D tensors.")


class SinePositionalEncoding(nn.Module):
    """Standard sine/cosine positional encoding."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x shape: (seq_len, batch, d_model)."""
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class ResidualRefiner(nn.Module):
    """2D CNN
Project-Unicom ResNetTHU 3x3 +"""

    def __init__(self, in_channels: int = 1, hidden: int = 32) -> None:
        super().__init__()
        k = 3
        p = k // 2
        self.conv1 = nn.Conv2d(in_channels, hidden, k, padding=p)
        self.conv2 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv3 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv4 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv5 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv6 = nn.Conv2d(hidden, hidden, k, padding=p)
        self.conv_out = nn.Conv2d(hidden, 1, k, padding=p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        c1 = self.relu(self.conv1(x))
        c2 = self.relu(self.conv2(c1))
        r1 = c2 + c1

        c3 = self.relu(self.conv3(r1))
        c4 = self.relu(self.conv4(c3))
        r2 = c4 + r1

        c5 = self.relu(self.conv5(r2))
        c6 = self.relu(self.conv6(c5))
        r3 = c6 + r2

        return self.conv_out(self.relu(r3))


class TemporalTransformer(nn.Module):
    """Applies Transformer encoder over temporal dimension per spatial location."""

    def __init__(
            self,
            feature_dim: int,
            hidden_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float = 0.1,
            max_seq_len: int = 168,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(feature_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, feature_dim)
        # hidden_dimnum_heads
        assert hidden_dim % num_heads == 0, f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = SinePositionalEncoding(hidden_dim, max_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        seq = x.permute(0, 3, 4, 1, 2).contiguous().view(b * h * w, t, c)
        seq = self.proj_in(seq)
        seq = seq.permute(1, 0, 2)
        seq = self.pos_encoding(seq)
        encoded = self.encoder(seq)
        encoded = encoded.permute(1, 0, 2)
        encoded = self.proj_out(encoded).view(b, h, w, t, c)
        return encoded.permute(0, 3, 4, 1, 2)


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell without builtin padding (padding handled outside)."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = 0
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
            self,
            input_tensor: torch.Tensor,
            cur_state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = cur_state
        if input_tensor.shape[-2:] != h_cur.shape[-2:]:
            diff_h = input_tensor.shape[-2] - h_cur.shape[-2]
            diff_w = input_tensor.shape[-1] - h_cur.shape[-1]
            if diff_h < 0 or diff_w < 0:
                raise ValueError(
                    "Hidden state spatial dims larger than input; cannot align tensors."
                )
            pad_top = diff_h // 2
            pad_bottom = diff_h - pad_top
            pad_left = diff_w // 2
            pad_right = diff_w - pad_left
            h_cur = F.pad(h_cur, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(combined_conv, chunks=4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
            self, batch_size: int, spatial_size: tuple[int, int], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTMLayer(nn.Module):
    """Single layer ConvLSTM that can optionally return sequences."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: int,
            return_sequences: bool,
    ) -> None:
        super().__init__()
        self.return_sequences = return_sequences
        self.kernel_size = kernel_size
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        batch, seq_len, _, height, width = x.shape
        out_height = height - self.kernel_size + 1
        out_width = width - self.kernel_size + 1
        h, c = self.cell.init_hidden(batch, (out_height, out_width), x.device)

        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t], (h, c))
            if self.return_sequences:
                outputs.append(h)
        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        return h


class DSConvLSTMNet(nn.Module):
    """CNN-LSTM 2D CNN LSTM sqrt(pop)"""

    def __init__(self, in_channels: int, cnn_channels: int = 32, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        # CNN
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
        )
        # LSTM over per-pixel sequences
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # CNN
        x_flat = x.view(B * T, C, H, W)
        feat = self.cnn_encoder(x_flat)              # (B*T, F, H, W)
        Fdim = feat.shape[1]
        feat = feat.view(B, T, Fdim, H, W)

        seq = feat.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, Fdim)  # (BHW, T, F)
        out, _ = self.lstm(seq)                     # (BHW, T, hidden)
        last = out[:, -1, :]
        pred = self.head(last)                      # (BHW, 1)
        pred = pred.view(B, H, W, 1).permute(0, 3, 1, 2)  # (B,1,H,W)
        return pred


def save_real_population_map(
        sqrt_map: np.ndarray,
        reference_tif: str,
        output_tif: str,
        mask: Optional[np.ndarray] = None,
) -> None:
    """sqrt(pop) GeoTIFF"""
    real_map = np.square(np.maximum(sqrt_map, 0.0)).astype(np.float32)
    if mask is not None:
        real_map = real_map * mask.astype(np.float32)
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    profile.update({"count": 1, "dtype": "float32"})
    # nodata/photometric 0 NoData
    profile.pop("photometric", None)
    profile.pop("colormap", None)
    profile.pop("nodata", None)
    profile.pop("NODATA", None)
    out_path = Path(output_tif)
    if out_path.exists():
        try:
            out_path.unlink()
        except PermissionError as exc:
            raise RuntimeError("Failed to overwrite existing file. Close the file and retry.") from exc
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(real_map, 1)
    print(f"Saved population prediction raster: {output_tif}")


def apply_hourly_linear_calibration(
        pop_map: np.ndarray,
        hour: int,
        hourly_calib: Optional[dict[int, tuple[float, float]]] = None,
) -> np.ndarray:
    """pop_corr = a_h * pop_map + b_h
- hourly_calib: {hour: (a, b)}
-"""
    if not hourly_calib:
        return pop_map
    a, b = hourly_calib.get(hour, (1.0, 0.0))
    return np.maximum(a * pop_map + b, 0.0)


def tem_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error over batch + ."""
    with torch.no_grad():
        mse = F.mse_loss(y_pred, y_true)
        return torch.log1p(mse)


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error over batch + ."""
    with torch.no_grad():
        return F.l1_loss(y_pred, y_true)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """R over batch +"""
    with torch.no_grad():

        y_t = y_true.view(y_true.size(0), -1)
        y_p = y_pred.view(y_pred.size(0), -1)
        mean_t = torch.mean(y_t, dim=1, keepdim=True)
        ss_tot = torch.sum((y_t - mean_t) ** 2, dim=1)
        ss_res = torch.sum((y_t - y_p) ** 2, dim=1)
        r2 = 1.0 - ss_res / (ss_tot + eps)
        # batch
        return torch.mean(r2)


def compute_hourly_calibration(
        model: nn.Module,
        base_dataset: "HourlyPopulationDataset",
        device: torch.device,
        sample_indices: Sequence[int],
        patch_size: int = 96,
        overlap: int = 32,
        eps: float = 1e-6,
) -> dict[int, tuple[float, float]]:
    """a_h, b_h
pop_true a_h * pop_pred + b_h
patch"""
    model.eval()
    dataset_mask = getattr(base_dataset, "study_mask_full", None)

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
            sample = base_dataset.samples[sample_idx]
            key_y = sample["key_y"]
            meta = base_dataset._meta_by_key[key_y]
            hour = meta["hour"]

            X_full, Y_full = base_dataset.get_full_sequence(sample_idx)  # (T,H,W,C), (H,W,1) sqrt(pop)
            pred_sqrt = predict_full_image(
                model=model,
                X_full=X_full,
                device=device,
                patch_size=patch_size,
                overlap=overlap,
            )
            true_pop = np.square(np.maximum(Y_full.squeeze(-1), 0.0))
            pred_pop = np.square(np.maximum(pred_sqrt, 0.0))

            if dataset_mask is not None:
                true_pop = true_pop * dataset_mask
                pred_pop = pred_pop * dataset_mask

            # NaN mask mask
            mask_valid = np.isfinite(true_pop) & np.isfinite(pred_pop)
            if dataset_mask is not None:
                mask_valid &= dataset_mask > 0

            if not np.any(mask_valid):
                continue
            tp = true_pop[mask_valid].astype(np.float64)
            pp = pred_pop[mask_valid].astype(np.float64)

            s_pred = float(pp.sum())
            s_true = float(tp.sum())
            s_pred2 = float(np.sum(pp * pp))
            s_pred_true = float(np.sum(pp * tp))
            cnt = float(pp.size)

            stats[hour]["sum_pred"] += s_pred
            stats[hour]["sum_true"] += s_true
            stats[hour]["sum_pred2"] += s_pred2
            stats[hour]["sum_pred_true"] += s_pred_true
            stats[hour]["count"] += cnt

    calib: dict[int, tuple[float, float]] = {}
    for h, st in stats.items():
        if st["count"] <= 0 or st["sum_pred2"] <= eps:
            calib[h] = (1.0, 0.0)
            continue
        mean_pred = st["sum_pred"] / st["count"]
        mean_true = st["sum_true"] / st["count"]
        # true = a*pred + b
        a = (st["sum_pred_true"] - st["count"] * mean_pred * mean_true) / max(
            st["sum_pred2"] - st["count"] * mean_pred * mean_pred, eps
        )
        b = mean_true - a * mean_pred
        calib[h] = (float(a), float(b))
    return calib


class MixedAbsRelLoss(nn.Module):
    """+
- alpha: alpha
- gamma: gamma>0 1 1
- eps:"""

    def __init__(self, alpha: float = 0.5, gamma: float = 0.3, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_sqrt: torch.Tensor, target_sqrt: torch.Tensor) -> torch.Tensor:
        # sqrt(pop)
        pred_pop = torch.clamp(pred_sqrt, min=0) ** 2
        true_pop = torch.clamp(target_sqrt, min=0) ** 2

        abs_err = torch.abs(pred_pop - true_pop)
        rel_err = abs_err / (true_pop + self.eps)

        mean_pop = torch.mean(true_pop)

        weight = torch.pow((true_pop + self.eps) / (mean_pop + self.eps), self.gamma)

        abs_loss = torch.mean(abs_err * weight)
        rel_loss = torch.mean(rel_err * weight)
        return abs_loss + self.alpha * rel_loss


def run_epoch(
        loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        device: torch.device,
        epoch: int = 0,
        phase: str = "train",
) -> dict:
    """epoch / TEM/MAE"""

    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss = 0.0
    total_tem = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_samples = 0

    num_batches = len(loader)

    for batch_idx, (X_batch, Y_batch) in enumerate(loader):

        # ---- GPU ----
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)


        if train_mode:
            optimizer.zero_grad()

        preds = model(X_batch)  # (B, 1, H, W), sqrt(pop)
        loss = criterion(preds, Y_batch)

        if train_mode:
            loss.backward()
            optimizer.step()


        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_tem += tem_metric(Y_batch, preds).item() * batch_size
        total_mae += mae(Y_batch, preds).item() * batch_size
        total_r2 += r2_score(Y_batch, preds).item() * batch_size
        total_samples += batch_size

        # ---- 50 batch ----
        if str(phase).lower() != "test" and batch_idx % 50 == 0:
            print(f"[{phase}] Epoch {epoch:03d} | Batch {batch_idx + 1}/{num_batches}")

    return {
        "loss": total_loss / max(1, total_samples),
        "TEM": total_tem / max(1, total_samples),
        "mae": total_mae / max(1, total_samples),
        "r2": total_r2 / max(1, total_samples),
    }


def train_model(
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        epochs: int,
        patience: int,
        best_loss_path: str,
        best_tem_path: str,
        checkpoint_map: Optional[dict[int, Path]] = None,
) -> list[dict]:
    """- MSELoss
    - TEM, MAE
- patience best_loss / best_tem"""

    print(f"Starting training for {epochs} epochs...")


    criterion = MixedAbsRelLoss(alpha=0.5, gamma=0.6, eps=1e-6)
    # criterion = MixedAbsRelLoss(alpha=0.5, gamma=0.3, eps=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    history = []
    best_val_loss = float("inf")
    best_val_tem = float("inf")
    patience_counter = 0


    os.makedirs(os.path.dirname(best_loss_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_tem_path), exist_ok=True)

    for ep in range(1, epochs + 1):

        # ---------- Train ----------
        train_metrics = run_epoch(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=ep,
            phase="train",
        )

        # ---------- Validation ----------
        val_metrics = run_epoch(
            loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch=ep,
            phase="val",
        )

        # ---------- Test ----------
        test_metrics = run_epoch(
            loader=test_loader,
            model=model,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch=ep,
            phase="test",
        )


        record = {
            "epoch": ep,
            "train_loss": train_metrics["loss"],
            "train_tem": train_metrics["TEM"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_tem": val_metrics["TEM"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_loss": test_metrics["loss"],
            "test_tem": test_metrics["TEM"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
        }
        history.append(record)

        # ---------- LR Scheduler ----------
        scheduler.step(record["val_loss"])

        print(
            f"Epoch {ep:03d} | "
            f"TrainLoss {record['train_loss']:.4f} "
            f"ValLoss {record['val_loss']:.4f} "
            f"Val TEM {record['val_tem']:.4f} "
            f"ValMAE {record['val_mae']:.4f} "
            f"ValR2 {record['val_r2']:.4f}"
        )

        # ---------- checkpoint ----------
        if checkpoint_map and ep in checkpoint_map:
            ckpt_dir = Path(checkpoint_map[ep])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"model_epoch_{ep}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[checkpoint] Saved epoch {ep} model to {ckpt_path}")

        # ---------- Early Stopping ----------
        improved = False

        # best loss
        if record["val_loss"] < best_val_loss - 1e-6:
            best_val_loss = record["val_loss"]
            torch.save(model.state_dict(), best_loss_path)
            improved = True

        # best TEM
        if record["val_tem"] < best_val_tem - 1e-6:
            best_val_tem = record["val_tem"]
            torch.save(model.state_dict(), best_tem_path)
            improved = True

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # TEM
    if history:
        best_record = min(history, key=lambda h: h["val_tem"])
        best_metrics = {
            "epoch": best_record["epoch"],
            "test_tem": best_record["test_tem"],
            "test_mae": best_record["test_mae"],
            "test_r2": best_record["test_r2"],
        }
        metrics_path = Path(best_tem_path).with_name("best_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, ensure_ascii=False, indent=2)
        print(f"[info] Saved best metrics to {metrics_path}")

    return history


from datetime import datetime as _dt


def predict_one_day(
        model: nn.Module,
        base_dataset: "HourlyPopulationDataset",
        device: torch.device,
        target_date_str: str,
        output_dir: Path,
        # patch_size: int = 128, # patch
        patch_size: int = 96,
        overlap: int = 32,
        hourly_calib: Optional[dict[int, tuple[float, float]]] = None,
) -> None:
    """patch"""
    target_date = _dt.strptime(target_date_str, "%Y-%m-%d").date()
    model.eval()


    selected: list[tuple[int, _dt]] = []
    for idx, sample in enumerate(base_dataset.samples):
        key_y = sample["key_y"]
        meta = base_dataset._meta_by_key[key_y]
        dt = meta["dt"]
        if dt.date() == target_date:
            selected.append((idx, dt))

    if not selected:
        print("Population forecasting status message.")
        return

    selected.sort(key=lambda x: x[1].hour)
    print("Population forecasting status message.")

    pop_sqrt_min = getattr(base_dataset, "pop_sqrt_min", 0.0)
    pop_sqrt_max = getattr(base_dataset, "pop_sqrt_max", 1.0)
    mask = getattr(base_dataset, "study_mask_full", None)

    for sample_idx, dt in selected:
        hour = dt.hour
        X_full, _ = base_dataset.get_full_sequence(sample_idx)  # (T, H, W, C)

        print(f"Processing sample {sample_idx}, hour {hour:02d}: input shape {X_full.shape}")

        # patch
        pred_sqrt = predict_full_image(
            model=model,
            X_full=X_full,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
        )

        # sqrt
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
        print(f"[ok] Exported prediction for {target_date_str} {hour:02d}:00 -> {out_tif}")


def visualize_patch_prediction(model: nn.Module, dataset: HourlyPopulationDataset, device: torch.device) -> None:
    """Visualize prediction vs GT for the first patch generated from dataset."""
    patch_dataset = HourlyPopulationPatchDataset(dataset, augment=False)
    if len(patch_dataset) == 0:
        print("No patches available for visualization.")
        return
    X_patch, Y_patch = patch_dataset[0]
    model.eval()
    with torch.no_grad():
        pred = model(X_patch.unsqueeze(0).to(device)).cpu().squeeze(0, 1).numpy()
    gt = Y_patch.squeeze(0).numpy()
    inputs = X_patch.numpy()
    im_0 = inputs[0, 0]
    im_1 = inputs[min(1, inputs.shape[0] - 1), 0]

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(im_0, cmap="gray")
    plt.title("t0 lc1")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(im_1, cmap="gray")
    plt.title("t1 lc1")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(pred, cmap="gray", vmin=0)
    plt.title("Prediction")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(gt, cmap="gray", vmin=0)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def split_into_patches(
        X: np.ndarray, patch_size: int = 30, overlap: int = 10
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Split (time, H, W, C) into overlapping spatial patches."""
    t, H, W, _ = X.shape
    if not patch_size:
        return [X], [(0, 0)]
    patches: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for i in _patch_start_indices(H, patch_size, overlap):
        for j in _patch_start_indices(W, patch_size, overlap):
            patch = X[:, i: i + patch_size, j: j + patch_size, :]
            patches.append(patch)
            origins.append((i, j))
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


def create_dataloader(
        base_dataset: HourlyPopulationDataset,
        batch_size: int,
        shuffle: bool,
        augment: bool,
        sample_indices: Optional[Sequence[int]] = None,
) -> DataLoader:
    """HourlyPopulationDataset DataLoader"""
    patch_dataset = HourlyPopulationPatchDataset(
        base_dataset, augment=augment, sample_indices=sample_indices
    )
    return DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=48,
        drop_last=False,
    )


def main() -> None:
    set_global_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================

    # =============================
    project_root = Path(__file__).resolve().parent.parent

    # =============================
    # 2. pop_dir
    # =============================
    # *.tif .aux.xml.ovr
    pop_dir = project_root / "data" / "mobile_phone_data"
    # weather_csv_path = project_root / "Datas_Weather" / "bjweather_with_workday.csv"

    # =============================
    # 3. static_dir
    # =============================
    static_dir = project_root / "data" / "static_variables"
    static_files = [
        "beijingNTL_500.tif",
        "building_height_500.tif",
        "beijing_pops_500.tif",
        "shopping_den_500.tif",
        "residence_den_500.tif",
        "recreat_den_500.tif",
        "medical_den_500.tif",
        "life_den_500.tif",
        "edu_den_500.tif",
        "canyin_den_500.tif",
        "transport_den_500.tif",
        "road_density_500.tif",
        "FABDEM_500.tif",
        "fab_slope_500m.tif",
        "work_den_500m.tif",
    ]

    # =============================

    # =============================
    results_dir = project_root / "output_results" / "dynamic_population_distribution" / "baselines" / "CNN-LSTM"
    results_dir.mkdir(parents=True, exist_ok=True)

    # =============================

    # =============================
    window_size = 24
    patch_size = 32
    overlap = 10

    shape_path = project_root / "data" / "shapefiles" / "core_urban_area_Beijing.shp"

    # target_weekdays=None
    base_dataset = HourlyPopulationDataset(
        pop_dir=str(pop_dir),
        static_dir=str(static_dir),
        static_files=static_files,
        window_size=window_size,
        patch_size=patch_size,
        overlap=overlap,
        rotation_angle=90,
        do_flip=True,
        random_seed=42,
        target_weekdays=None,
        shape_path=str(shape_path),
    )

    print(f"Total time-window samples: {len(base_dataset)}")


    # =============================
    # 6. 9 + 10 10-09 6:2:2 train / val / test
    # =============================
    combined_indices = []
    combined_dates = []

    for idx, sample in enumerate(base_dataset.samples):
        key_y = sample["key_y"]
        dt = base_dataset._meta_by_key[key_y]["dt"]
        in_sept = dt.month == 9
        if in_sept:
            combined_indices.append(idx)
            combined_dates.append(dt.date())

    num_samples = len(combined_indices)


    unique_dates = sorted(list(set(combined_dates)))
    print(f"Dates used: {[d.strftime('%Y-%m-%d') for d in unique_dates]}")


    total_days = len(unique_dates)
    train_days = int(total_days * 0.6)
    val_days = int(total_days * 0.2)

    train_end_day = unique_dates[train_days - 1] if train_days > 0 else unique_dates[0]
    val_end_day = unique_dates[train_days + val_days - 1] if val_days > 0 else unique_dates[-1]

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for idx, dt in zip(combined_indices, combined_dates):
        if dt <= train_end_day:
            train_indices.append(idx)
        elif dt <= val_end_day:
            val_indices.append(idx)
        else:
            test_indices.append(idx)

    print(f"Total samples (time windows): {num_samples}")
    print(f"Train samples (60%): {len(train_indices)}")
    print(f"Val   samples (20%): {len(val_indices)}")
    print(f"Test  samples (20%): {len(test_indices)}")
    print(f"Train date range: {unique_dates[0].strftime('%Y-%m-%d')} to {train_end_day.strftime('%Y-%m-%d')}")
    if val_days > 0:
        print(f"Val date range: {unique_dates[train_days].strftime('%Y-%m-%d')} to {val_end_day.strftime('%Y-%m-%d')}")
    if train_days + val_days < total_days:
        print(
            f"Test date range: {unique_dates[train_days + val_days].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}")

    # key
    train_keys: set[str] = set()
    for idx in train_indices:
        sample = base_dataset.samples[idx]
        train_keys.update(sample["key_seq"])
        train_keys.add(sample["key_y"])

    base_dataset.recompute_pop_norm(keys=train_keys)

    batch_size = 4
    train_loader = create_dataloader(
        base_dataset,
        batch_size=batch_size,
        shuffle=True,
        augment=False,
        sample_indices=train_indices,
    )

    val_loader = create_dataloader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        sample_indices=val_indices,
    )

    test_loader = create_dataloader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        sample_indices=test_indices,
    )

    # =============================

    # =============================
    C_static = len(static_files)
    in_channels = 1 + C_static + 5 + 1  # pop + static + time_encoding + prev_week_pop

    model = DSConvLSTMNet(
        in_channels=in_channels,
        cnn_channels=16,
        hidden_size=64,
        num_layers=2,
        dropout=0.25,
    ).to(device)

    best_loss_path = results_dir / "best_loss_CNN_LSTM.pt"
    best_tem_path = results_dir / "best_tem_CNN_LSTM.pt"


    checkpoint_epochs = [60, 70, 80, 90, 100, 110, 120]
    checkpoint_map = {ep: results_dir / f"CNN-LSTM_0925_{ep}" for ep in checkpoint_epochs}

    target_dates: list[str] = [
        "2017-09-25",
        "2017-09-26",
        "2017-09-27",
        "2017-09-28",
    ]

    # =============================

    # =============================
    history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        device=device,
        epochs=30,
        # epochs=120,
        # epochs=1000

        # patience=16,
        patience=10,
        # patience=3
        best_loss_path=str(best_loss_path),
        best_tem_path=str(best_tem_path),
        checkpoint_map=checkpoint_map,
    )

    # =============================
    # 9. (a_h, b_h)
    # =============================
    hourly_calib = compute_hourly_calibration(
        model=model,
        base_dataset=base_dataset,
        device=device,
        sample_indices=val_indices,
        patch_size=96,
        overlap=16,
    )
    print(f"[info] Estimated hourly calibration (a,b): {hourly_calib}")

    # =============================
    # 10. checkpoint
    # =============================
    for ep, ckpt_dir in checkpoint_map.items():
        ckpt_path = ckpt_dir / f"model_epoch_{ep}.pt"
        if not ckpt_path.exists():
            print(f"[warn] Checkpoint {ckpt_path} not found, skip prediction.")
            continue
        temp_model = DSConvLSTMNet(
            in_channels=in_channels,
            cnn_channels=16,
            hidden_size=64,
            num_layers=2,
            dropout=0.25,
        ).to(device)
        temp_model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        for target_date_str in target_dates:
            target_date_tag = target_date_str.replace("-", "")
            pred_dir = ckpt_dir / f"Pred_{target_date_tag}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            predict_one_day(
                model=temp_model,
                base_dataset=base_dataset,
                device=device,
                target_date_str=target_date_str,
                output_dir=pred_dir,
                hourly_calib=hourly_calib,
            )

    # =============================
    # 11. TEM
    # =============================
    model.load_state_dict(torch.load(str(best_tem_path), map_location=device))
    model.to(device)
    visualize_patch_prediction(model, base_dataset, device)

    # =============================
    # 12. TEM (0-23)
    # =============================
    for target_date_str in target_dates:
        target_date_tag = target_date_str.replace("-", "")
        day_pred_dir = results_dir / f"Pred_{target_date_tag}"
        day_pred_dir.mkdir(parents=True, exist_ok=True)
        predict_one_day(
            model=model,
            base_dataset=base_dataset,
            device=device,
            target_date_str=target_date_str,
            output_dir=day_pred_dir,
            hourly_calib=hourly_calib,
        )


if __name__ == "__main__":
    main()

