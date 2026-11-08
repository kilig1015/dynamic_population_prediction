from __future__ import annotations
import os

import glob
import random
import re
import time
import json
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, OrderedDict as TypingOrderedDict

from datetime import datetime, timedelta

import numpy as np
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None
try:
    from osgeo import gdal  # pylint: disable=unused-import
except ImportError:
    gdal = None
try:
    import rasterio
    from rasterio.features import rasterize
except ImportError:
    rasterio = None
    rasterize = None
try:
    import fiona
except ImportError:
    fiona = None
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import rotate
from torch.utils.data import DataLoader, Dataset

os.environ["USE_PATH_FOR_GDAL_PYTHON"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEFAULT_RANDOM_SEED = 42


def _require_pyplot():
    global matplotlib, plt
    if plt is None:
        try:
            import matplotlib as _matplotlib
            _matplotlib.use('Agg')
            import matplotlib.pyplot as _plt
        except Exception as exc:
            raise ImportError(
                'matplotlib is unavailable in the current environment. '
                'Install a matplotlib build compatible with the current NumPy version to use plotting features.'
            ) from exc
        matplotlib = _matplotlib
        plt = _plt
    return plt


def _require_geospatial_stack():
    missing = []
    if rasterio is None or rasterize is None:
        missing.append('rasterio')
    if fiona is None:
        missing.append('fiona')
    if missing:
        joined = ', '.join(missing)
        raise ImportError(
            f'Missing required geospatial dependencies: {joined}. '
            'Install them in the active interpreter before running Pop-HSTM_STTest.py.'
        )

DEFAULT_OUTPUT_ROOT_REL = Path("output_results") / "dynamic_population_distribution" / "Spatial_Temporal_Generalization"
LOW_DENSITY_TRAIN_SHAPE_REL = Path("data") / "shapefiles" / "random_shps" / "random_polys.shp"
HIGH_DENSITY_TRAIN_SHAPE_REL = Path("data") / "shapefiles" / "random_shps" / "random_polys_other.shp"
OUTER_SHAPE_REL = Path("data") / "shapefiles" / "core_urban_area_Beijing.shp"
LOW_DENSITY_OUTPUT_REL = DEFAULT_OUTPUT_ROOT_REL / "low_density_ring_generalization"
HIGH_DENSITY_OUTPUT_REL = DEFAULT_OUTPUT_ROOT_REL / "high_density_ring_generalization"
DEFAULT_WINDOW_SEARCH = [6, 8, 12, 24]
DEFAULT_LR_SEARCH = [1e-4, 5e-5, 5e-4]
DEFAULT_PATCHES_PER_SAMPLE_SEARCH = [12, 14, 16, 18]
DEFAULT_LOSS_SEARCH = ["mixed_abs_rel", "mse", "huber", "l1", "log_cosh"]
DEFAULT_TARGET_DATES = [
    "2017-09-25",
    "2017-09-26",
    "2017-09-27",
    "2017-09-28",
]


def resolve_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "mobile_phone_data").exists() and (candidate / "data" / "static_variables").exists():
            return candidate
    raise RuntimeError("Failed to resolve project root from Pop-HSTM_STTest.py location.")


def resolve_run_root() -> Path:
    run_root = resolve_project_root() / DEFAULT_OUTPUT_ROOT_REL
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _resolve_project_path(project_root: Path, path_value: Union[str, Path]) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else project_root / path


def resolve_output_dir_for_training_shape(project_root: Path, shape_path: Union[str, Path]) -> Path:
    shape_resolved = _resolve_project_path(project_root, shape_path).resolve()
    low_shape = (project_root / LOW_DENSITY_TRAIN_SHAPE_REL).resolve()
    high_shape = (project_root / HIGH_DENSITY_TRAIN_SHAPE_REL).resolve()
    if shape_resolved == low_shape:
        return project_root / LOW_DENSITY_OUTPUT_REL
    if shape_resolved == high_shape:
        return project_root / HIGH_DENSITY_OUTPUT_REL
    raise ValueError(
        "Unsupported training shapefile for OSIT output routing: "
        f"{shape_resolved}. Expected {low_shape} or {high_shape}."
    )


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def _patch_start_indices(length: int, patch_size: int, overlap: int) -> List[int]:
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


def compute_group_count(num_channels: int, max_groups: int = 8) -> int:
    """Return the largest group count (<= max_groups) that divides num_channels."""
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


# Default hourly linear-calibration coefficients (a, b). Identity mapping means no calibration.
DEFAULT_HOURLY_CALIB: Dict[int, Tuple[float, float]] = {h: (1.0, 0.0) for h in range(24)}


class HourlyPopulationDataset:
    """
    Hourly population dataset built from mobile-phone raster sequences.
    Each timestep stacks dynamic population, static variables, calendar encodings,
    and the population map from the same hour in the previous week.
    Samples are created with a sliding time window of length ``window_size``.
    The target is the next-hour population raster in sqrt(pop) space.
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
        Args:
            pop_dir: Directory containing hourly mobile-phone population rasters.
            static_dir: Directory containing static raster covariates.
            static_files: Static raster filenames relative to ``static_dir``.
            window_size: Number of historical hours used as model input.
            patch_size/overlap/rotation_angle/do_flip: Spatial sampling and augmentation settings.
            target_weekdays: Optional weekday filter (0=Mon, ..., 6=Sun).
            shape_path: Optional shapefile used to define the study-area mask.
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
        self.sampling_mask: Optional[np.ndarray] = None
        self.pop_norm_method = pop_norm_method.lower()
        if self.pop_norm_method not in {"zscore", "minmax"}:
            raise ValueError(f"Unsupported pop_norm_method: {pop_norm_method}")
        self.pop_sqrt_min = 0.0
        self.pop_sqrt_max = 1.0
        self.pop_sqrt_mean = 0.0
        self.pop_sqrt_std = 1.0
        self._stats_filename = f"pop_norm_stats_{self.pop_norm_method}.json"

        np.random.seed(random_seed)

        # Load timestep metadata and the raster-path mapping once at initialization.
        self.time_meta, self.pop_path_map = self._load_all_data()
        self._meta_by_key = {m["key"]: m for m in self.time_meta}
        # Keep caches effectively disabled to minimize memory usage in this project version.
        self._pop_cache_limit = 0
        self._feat_cache_limit = 0
        self._pop_cache: TypingOrderedDict[str, np.ndarray] = OrderedDict()
        self._feat_cache: TypingOrderedDict[str, np.ndarray] = OrderedDict()
        self._patch_origins: Optional[List[Tuple[int, int]]] = None
        self._patch_shape: Optional[Tuple[int, int]] = None

        # Build the sliding-window sample index after metadata has been prepared.
        self.samples = self._create_time_window_samples()

    # ---------- Internal helpers ----------

    @staticmethod
    def _parse_pop_filename(path: str) -> datetime:
        """
        Parse filenames such as ``point090400_krigin_result.tif`` into datetimes.
        The filename encodes month, day, and hour; the year is fixed to 2017.
        """
        basename = os.path.basename(path)
        # Extract the ``pointMMDDHH`` token and ignore any suffix variations.
        match = re.search(r"point(\d{6})", basename)
        if not match:
            raise ValueError(f"Unable to parse timestamp from filename: {basename}")
        mmddhh = match.group(1)  # "MMDDHH"
        ts_str = "2017" + mmddhh
        return datetime.strptime(ts_str, "%Y%m%d%H")

    def _load_static_rasters(self) -> np.ndarray:
        """
        Load all static rasters, apply min-max scaling, and stack them as (H, W, C_static).
        All static rasters are expected to share the same spatial shape and resolution.
        """
        static_arrays: List[np.ndarray] = []
        ref_shape = None

        for fname in self.static_files:
            tif_path = os.path.join(self.static_dir, fname)
            if not os.path.exists(tif_path):
                raise FileNotFoundError(f"Static variable raster not found: {tif_path}")
            with rasterio.open(tif_path) as src:
                arr = src.read(1).astype(np.float32)
            arr = min_max_scaler(arr)
            if ref_shape is None:
                ref_shape = arr.shape
            else:
                if arr.shape != ref_shape:
                    raise ValueError(
                        f"Static raster shape mismatch: {fname}, shape={arr.shape}, ref={ref_shape}"
                    )
            static_arrays.append(arr)

        if not static_arrays:
            raise ValueError("No static variable rasters were provided.")

        static_stack = np.stack(static_arrays, axis=-1)  # (H, W, C_static)
        return static_stack

    def _load_all_data(self) -> Tuple[List[dict], Dict[str, str]]:
        """
        Load all hourly population rasters and build the time-metadata table.
        Returns a metadata list sorted by timestamp and a key-to-raster-path mapping.
        Each metadata item contains dt/hour/weekday/weekend/key fields.
        The mapping uses the same key convention as the metadata list.
        """
        # time_meta items: {"dt", "hour", "weekday", "is_weekend", "key"}
        # pop_path_map maps each key to its corresponding raster path.
        pop_tifs = glob.glob(os.path.join(self.pop_dir, "*.tif"))
        if not pop_tifs:
            raise RuntimeError(f"No .tif mobile-phone rasters found under {self.pop_dir}.")

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
                print(f"[warn] Failed to read normalization stats from {stats_path}; recomputing.")

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
                raise RuntimeError("Failed to compute a valid global sqrt(pop) range from population rasters.")
            if total_pixels <= 0:
                raise RuntimeError("Population rasters contain zero valid pixels; cannot compute mean and variance.")
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
                print(f"[warn] Failed to write normalization stats to {stats_path}")

        tmp_records = []
        for path in pop_tifs:
            dt = self._parse_pop_filename(path)
            key = dt.strftime("%Y%m%d%H")
            tmp_records.append({"dt": dt, "key": key, "path": path})

        tmp_records.sort(key=lambda x: x["dt"])
        pop_path_map = {item["key"]: item["path"] for item in tmp_records}

        # Static rasters are loaded once and reused when features are assembled on demand.
        self.static_stack = self._load_static_rasters()
        self._grid_shape = self.static_stack.shape[:2]

        # If a study-area shapefile is provided, build a mask and zero-out values outside it.
        if self.shape_path is not None:
            self.study_mask_full = self._build_study_mask_from_shapefile()
            # Zero static covariates outside the study area.
            self.static_stack = self.static_stack * self.study_mask_full[..., np.newaxis]
        else:
            self.study_mask_full = None

        time_meta: List[dict] = []
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
        """Rasterize the study-area shapefile into a binary mask of shape (H, W)."""
        if self.shape_path is None:
            raise RuntimeError("shape_path is not set, but _build_study_mask_from_shapefile was called.")
        if not self.static_files:
            raise RuntimeError("static_files is empty; cannot determine the reference raster.")

        ref_tif = os.path.join(self.static_dir, self.static_files[0])
        if not os.path.exists(ref_tif):
            raise FileNotFoundError(f"Reference static variable raster not found: {ref_tif}")

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
    ) -> Tuple[float, float]:
        """
        Recompute sqrt(pop) normalization statistics from the provided keys.
        Optionally save the resulting min/max/mean/std values to JSON.
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
                raise KeyError(f"Population raster key not found: {key}")
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
            raise RuntimeError("Failed to compute a valid sqrt(pop) range from the provided keys.")
        if total_pixels <= 0:
            raise RuntimeError("Population rasters contain zero valid pixels; cannot compute mean and variance.")
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
                print(f"[warn] Failed to write normalization stats to {save_path}")
        return sqrt_min, sqrt_max

    def _evict_cache_if_needed(
            self, cache: TypingOrderedDict[str, np.ndarray], limit: int
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
        """Load a population raster on demand with an LRU-style cache."""
        if key in self._pop_cache:
            arr = self._pop_cache.pop(key)
            self._pop_cache[key] = arr  # Move to the end to mark this entry as recently used.
            return arr
        if key not in self.pop_path_map:
            raise KeyError(f"Population raster key not found: {key}")
        tif_path = self.pop_path_map[key]
        with rasterio.open(tif_path) as src:
            arr = src.read(1).astype(np.float32)
        self._pop_cache[key] = arr
        self._evict_cache_if_needed(self._pop_cache, self._pop_cache_limit)
        return arr

    def _get_feature_cube(self, key: str) -> np.ndarray:
        """Build the feature cube (H, W, C) for the specified timestep."""
        if key in self._feat_cache:
            feat = self._feat_cache.pop(key)
            self._feat_cache[key] = feat
            return feat

        meta = self._meta_by_key.get(key)
        if meta is None:
            raise KeyError(f"Time metadata not found for key: {key}")
        arr_pop = self._get_pop_array(key)
        # Apply the study-area/output mask to the population target when available.
        if self.study_mask_full is not None:
            arr_pop = arr_pop * self.study_mask_full
        H, W = self._grid_shape
        if arr_pop.shape != (H, W):
            raise ValueError(
                f"Population raster shape mismatch with static variables: key={key}, pop_shape={arr_pop.shape}, static_shape={(H, W)}"
            )
        # Convert population counts to sqrt(pop) before normalization.
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
                    f"Previous-week population raster shape mismatch: key={prev_key}, pop_shape={prev_pop_arr.shape}, static_shape={(H, W)}"
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

    def _create_time_window_samples(self) -> List[Dict[str, Any]]:
        """
        Create sliding-window samples using consecutive hourly keys.
        Each sample stores only the input-key sequence and the target key.
        """
        samples: List[Dict[str, Any]] = []

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

    def get_output_mask(self) -> Optional[np.ndarray]:
        """Return the mask used for targets / metrics / exported predictions."""
        if self.sampling_mask is not None:
            return self.sampling_mask
        return self.study_mask_full

    def get_full_sequence(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build the full-frame (X, Y) pair when patching is disabled."""
        sample = self.samples[idx]
        key_seq: Sequence[str] = sample["key_seq"]
        key_y: str = sample["key_y"]
        X = np.stack([self._get_feature_cube(k) for k in key_seq], axis=0).astype(np.float32, copy=False)
        y_pop = self._get_pop_array(key_y)
        target_mask = self.get_output_mask()
        if target_mask is not None:
            y_pop = y_pop * target_mask
        y_pop = np.where(y_pop < 0, 0.0, y_pop)
        Y = np.sqrt(y_pop)[..., np.newaxis]
        return X, Y

    def get_patch_origins(self) -> List[Tuple[int, int]]:
        """Return top-left coordinates for all spatial patches."""
        if self._patch_origins is not None:
            return self._patch_origins
        H, W = self._grid_shape
        if not self.patch_size:
            self._patch_shape = (H, W)
            self._patch_origins = [(0, 0)]
            return self._patch_origins
        if self.patch_size > H or self.patch_size > W:
            raise ValueError(
                f"patch_size={self.patch_size} exceeds grid size {(H, W)}"
            )
        origins: List[Tuple[int, int]] = []
        for i in _patch_start_indices(H, self.patch_size, self.overlap):
            for j in _patch_start_indices(W, self.patch_size, self.overlap):
                # When a sampling mask is provided, keep only patches that intersect it.
                if hasattr(self, "sampling_mask") and self.sampling_mask is not None:
                    mask_patch = self.sampling_mask[i: i + self.patch_size, j: j + self.patch_size]
                    if mask_patch.sum() <= 0:
                        continue
                origins.append((i, j))
        if not origins:
            origins = [(0, 0)]
        self._patch_shape = (self.patch_size, self.patch_size)
        self._patch_origins = origins
        return self._patch_origins

    def num_patches_per_sample(self) -> int:
        return len(self.get_patch_origins())

    def get_patch_by_index(self, sample_idx: int, patch_pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (X_patch, Y_patch) pair for the given patch index."""
        if sample_idx < 0 or sample_idx >= len(self.samples):
            raise IndexError("sample_idx out of range.")
        origins = self.get_patch_origins()
        if not origins:
            raise RuntimeError("Patch origins are not defined.")
        if patch_pos < 0 or patch_pos >= len(origins):
            raise IndexError("patch_pos out of range.")

        sample = self.samples[sample_idx]
        top, left = origins[patch_pos]

        if not self.patch_size:
            return self.get_full_sequence(sample_idx)

        patch_size = self.patch_size
        X_seq: List[np.ndarray] = []
        for key in sample["key_seq"]:
            feat = self._get_feature_cube(key)
            patch = feat[top: top + patch_size, left: left + patch_size, :]
            X_seq.append(np.ascontiguousarray(patch))
        X_patch = np.stack(X_seq, axis=0)
        y_pop = self._get_pop_array(sample["key_y"])
        target_mask = self.get_output_mask()
        if target_mask is not None:
            y_pop = y_pop * target_mask
        y_pop = np.where(y_pop < 0, 0.0, y_pop)
        y_sqrt = np.sqrt(y_pop)
        Y_patch = np.ascontiguousarray(
            y_sqrt[top: top + patch_size, left: left + patch_size][..., np.newaxis]
        )
        return X_patch, Y_patch

    def _augment(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random flip and rotation augmentation to a spatiotemporal patch."""
        X_aug = X.copy()
        Y_aug = Y.copy()
        if self.do_flip:
            if np.random.rand() < 0.5:
                X_aug = np.flip(X_aug, axis=1)  # Vertical flip.
                Y_aug = np.flip(Y_aug, axis=0)
            if np.random.rand() < 0.5:
                X_aug = np.flip(X_aug, axis=2)  # Horizontal flip.
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
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split a full-frame sample into overlapping spatial patches."""
        if not self.patch_size:
            return [X], [Y]

        X_patches: List[np.ndarray] = []
        Y_patches: List[np.ndarray] = []
        _, H, W, _ = X.shape
        for i in _patch_start_indices(H, self.patch_size, self.overlap):
            for j in _patch_start_indices(W, self.patch_size, self.overlap):
                X_patch = X[:, i: i + self.patch_size, j: j + self.patch_size, :]
                Y_patch = Y[i: i + self.patch_size, j: j + self.patch_size, :]
                X_patches.append(X_patch)
                Y_patches.append(Y_patch)
        return X_patches, Y_patches


class HourlyPopulationPatchDataset(Dataset):
    """Patch dataset that samples spatial patches from HourlyPopulationDataset."""


    def __init__(
            self,
            base_dataset: HourlyPopulationDataset,
            augment: bool = False,
            sample_indices: Optional[Sequence[int]] = None,
            patches_per_sample: int = 16,
    ) -> None:
        self.base_dataset = base_dataset
        self.augment = augment
        if sample_indices is None:
            self.sample_indices = list(range(len(self.base_dataset)))
        else:
            self.sample_indices = list(sample_indices)

        # Enumerate all valid patch origins once and reuse them for every sample.
        self.patch_origins = list(self.base_dataset.get_patch_origins())
        if len(self.patch_origins) == 0:
            raise ValueError("Failed to generate patch coordinates. Check patch_size/overlap settings.")

        # Each sample draws a limited number of random patches per epoch.
        # This reduces the number of batches and keeps training practical.
        # self.patches_per_sample = 16
        # self.patches_per_sample = 12
        self.patches_per_sample = int(patches_per_sample)
        if self.patches_per_sample <= 0:
            raise ValueError("patches_per_sample must be positive.")
        self.total_samples = len(self.sample_indices)

    def __len__(self) -> int:
        return self.total_samples * self.patches_per_sample

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Patch index out of range.")
        # Map the flat dataset index back to the underlying time-window sample.
        sample_idx = self.sample_indices[idx // self.patches_per_sample]
        # Randomly choose one patch location for this sample instance.
        patch_offset = random.randrange(len(self.patch_origins))
        X_patch, Y_patch = self.base_dataset.get_patch_by_index(sample_idx, patch_offset)
        key_y = self.base_dataset.samples[sample_idx]["key_y"]
        hour = self.base_dataset._meta_by_key[key_y]["hour"]
        if self.augment:
            X_patch, Y_patch = self.base_dataset._augment(X_patch, Y_patch)
        X_tensor = numpy_sequence_to_tensor(X_patch)
        Y_tensor = numpy_target_to_tensor(Y_patch)
        hour_tensor = torch.tensor(hour, dtype=torch.int64)
        return X_tensor, Y_tensor, hour_tensor


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

    def __init__(self, pad: Tuple[int, int]) -> None:
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
    """
    Lightweight 2D residual CNN used to refine the main prediction output.
    It applies several 3x3 convolutions with local residual connections.
    """

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
            max_seq_len: int = 168,  # Cover up to one week of hourly history.
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(feature_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, feature_dim)
        # The transformer hidden dimension must be divisible by the number of heads.
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
            cur_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            self, batch_size: int, spatial_size: Tuple[int, int], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        outputs: List[torch.Tensor] = []
        for t in range(seq_len):
            h, c = self.cell(x[:, t], (h, c))
            if self.return_sequences:
                outputs.append(h)
        if self.return_sequences:
            return torch.stack(outputs, dim=1)
        return h


class DSConvLSTMNet(nn.Module):
    """PyTorch implementation of the DS-ConvLSTM architecture."""

    def __init__(
            self,
            in_channels: int,
            kn_s: int = 3,
            dsfilter1: int = 8,
            dsfilter2: int = 4,
            filter1: int = 30,
            filter2: int = 20,
            unit1: int = 4,
            use_transformer: bool = True,
            transformer_dim: int = 64,
            transformer_heads: int = 4,
            transformer_layers: int = 2,
            transformer_dropout: float = 0.1,
            use_residual_refiner: bool = True,
            residual_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.kernel = kn_s
        self.pad = ReflectionPad2dTime((kn_s // 2, kn_s // 2))
        self.use_transformer = use_transformer
        self.use_residual_refiner = use_residual_refiner
        self.ds_conv1 = DepthwiseSeparableConv2d(in_channels, dsfilter1, kn_s)
        self.ds_norm1 = nn.GroupNorm(compute_group_count(dsfilter1), dsfilter1)
        self.ds_conv2 = DepthwiseSeparableConv2d(dsfilter1, dsfilter2, kn_s)
        self.ds_norm2 = nn.GroupNorm(compute_group_count(dsfilter2), dsfilter2)
        self.ds_conv3 = nn.Conv2d(dsfilter2, 1, kernel_size=kn_s, bias=True)
        self.skip_reduce = nn.Conv2d(1, unit1, kernel_size=1, bias=False)
        self.skip_norm = nn.GroupNorm(compute_group_count(unit1), unit1)
        self.skip_out = nn.Conv2d(unit1, 1, kernel_size=3, padding=1, bias=True)

        self.convlstm1 = ConvLSTMLayer(1, filter1, kn_s, return_sequences=True)
        self.norm_lstm1 = nn.GroupNorm(compute_group_count(filter1), filter1)
        if self.use_transformer:
            self.temporal_transformer = TemporalTransformer(
                feature_dim=filter1,
                hidden_dim=transformer_dim,
                num_heads=transformer_heads,
                num_layers=transformer_layers,
                dropout=transformer_dropout,
                # max_seq_len=64,
                max_seq_len=168,  # Cover up to one week of hourly history.

            )
        self.convlstm2 = ConvLSTMLayer(filter1, filter2, kn_s, return_sequences=False)
        self.norm_lstm2 = nn.GroupNorm(compute_group_count(filter2), filter2)
        self.pointwise = nn.Conv2d(filter2, unit1, kernel_size=1)
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.final_conv = nn.Conv2d(unit1, 1, kernel_size=kn_s)
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(2, unit1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(compute_group_count(unit1), unit1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unit1, 1, kernel_size=3, padding=1, bias=True),
        )
        self.residual_refiner = ResidualRefiner(in_channels=1,
                                                hidden=residual_hidden) if self.use_residual_refiner else None

    @staticmethod
    def _match_spatial(src: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Align spatial size of src tensor to target height/width via crop/pad."""
        tgt_h, tgt_w = target_hw
        src_h, src_w = src.shape[-2:]
        diff_h = tgt_h - src_h
        diff_w = tgt_w - src_w
        if diff_h == 0 and diff_w == 0:
            return src
        if diff_h < 0:
            top = (-diff_h) // 2
            bottom = top + tgt_h
            src = src[..., top:bottom, :]
        else:
            pad_top = diff_h // 2
            pad_bottom = diff_h - pad_top
            src = F.pad(src, (0, 0, pad_top, pad_bottom), mode="reflect")
        if diff_w < 0:
            left = (-diff_w) // 2
            right = left + tgt_w
            src = src[..., :, left:right]
        else:
            pad_left = diff_w // 2
            pad_right = diff_w - pad_left
            src = F.pad(src, (pad_left, pad_right, 0, 0), mode="reflect")
        return src

    @staticmethod
    def _time_distributed(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
        batch, seq_len, channels, height, width = x.shape
        reshaped = x.view(batch * seq_len, channels, height, width)
        out = module(reshaped)
        _, out_channels, out_h, out_w = out.shape
        return out.view(batch, seq_len, out_channels, out_h, out_w)

    @staticmethod
    def _time_distributed_norm(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        batch, seq_len, channels, height, width = x.shape
        reshaped = x.view(batch * seq_len, channels, height, width)
        normed = norm(reshaped)
        return normed.view(batch, seq_len, channels, height, width)

    @staticmethod
    def _apply_norm3d(x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        x_perm = x.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
        normed = norm(x_perm)
        return normed.permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Time-distributed DS convolutions
        x = self.pad(x)
        x = self._time_distributed(x, self.ds_conv1)
        x = self._time_distributed_norm(x, self.ds_norm1)

        x = self.pad(x)
        x = self._time_distributed(x, self.ds_conv2)
        x = self._time_distributed_norm(x, self.ds_norm2)

        x = self.pad(x)
        x = self._time_distributed(x, self.ds_conv3)
        skip_map = x[:, -1]

        # ConvLSTM stack
        x = self.pad(x)
        x = self.convlstm1(x)
        if self.use_transformer:
            x = self.temporal_transformer(x)
        x = self._apply_norm3d(x, self.norm_lstm1)

        x = self.pad(x)
        x = self.convlstm2(x)
        x = self.norm_lstm2(x)

        detail_feat = self.pointwise(x)
        detail_feat = self.dropout2d(F.relu(detail_feat))
        logits = self.final_conv(self.pad(detail_feat))

        skip_map = self.pad(skip_map)
        skip_feat = self.skip_reduce(skip_map)
        skip_feat = F.relu(self.skip_norm(skip_feat))
        skip_logits = self.skip_out(skip_feat)
        skip_logits = self._match_spatial(skip_logits, logits.shape[-2:])

        fusion = torch.cat([logits, skip_logits], dim=1)
        refined = self.fusion_refine(fusion)
        out = logits + refined
        if self.residual_refiner is not None:
            # Apply an additional residual refinement stage to the fused prediction.
            out = out + self.residual_refiner(out)
        return out


def save_real_population_map(
        sqrt_map: np.ndarray,
        reference_tif: str,
        output_tif: str,
        mask: Optional[np.ndarray] = None,
) -> None:
    """Convert a sqrt(pop) map back to population counts and save it as GeoTIFF."""
    real_map = np.square(np.maximum(sqrt_map, 0.0)).astype(np.float32)
    if mask is not None:
        real_map = real_map * mask.astype(np.float32)
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
    profile.update({"count": 1, "dtype": "float32"})
    # Remove metadata that would otherwise treat zero as nodata and create holes.
    profile.pop("photometric", None)
    profile.pop("colormap", None)
    profile.pop("nodata", None)
    profile.pop("NODATA", None)
    out_path = Path(output_tif)
    if out_path.exists():
        try:
            out_path.unlink()
        except PermissionError as exc:
            raise RuntimeError(f"Cannot remove existing file {output_tif}. Close any application using it and retry.") from exc
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(real_map, 1)
    print(f"[info] Saved predicted population raster to {output_tif}")


def apply_hourly_linear_calibration(
        pop_map: np.ndarray,
        hour: int,
        hourly_calib: Optional[Dict[int, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Apply hour-specific linear calibration in population space.
    The corrected map is ``a_h * pop_map + b_h`` and is clamped to be non-negative.
    Missing hours fall back to the identity mapping ``(1.0, 0.0)``.
    """
    if not hourly_calib:
        return pop_map
    a, b = hourly_calib.get(hour, (1.0, 0.0))
    return np.maximum(a * pop_map + b, 0.0)


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error over the batch and spatial dimensions."""
    with torch.no_grad():
        mse = F.mse_loss(y_pred, y_true)
        return torch.sqrt(mse)


def tem_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """TEM metric: TEM = ln(1 + MSE)."""
    with torch.no_grad():
        mse = F.mse_loss(y_pred, y_true)
        return torch.log1p(mse)


def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error over the batch and spatial dimensions."""
    with torch.no_grad():
        return F.l1_loss(y_pred, y_true)


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """R-squared over the batch and spatial dimensions."""
    with torch.no_grad():
        # Flatten spatial/channel dimensions so the metric is computed per sample.
        y_t = y_true.view(y_true.size(0), -1)
        y_p = y_pred.view(y_pred.size(0), -1)
        mean_t = torch.mean(y_t, dim=1, keepdim=True)
        ss_tot = torch.sum((y_t - mean_t) ** 2, dim=1)
        ss_res = torch.sum((y_t - y_p) ** 2, dim=1)
        r2 = 1.0 - ss_res / (ss_tot + eps)
        # Average R-squared over the batch dimension.
        return torch.mean(r2)


def compute_hourly_calibration(
        model: nn.Module,
        base_dataset: "HourlyPopulationDataset",
        device: torch.device,
        sample_indices: Sequence[int],
        patch_size: int = 96,
        overlap: int = 48,
        eps: float = 1e-6,
        mask_override: Optional[np.ndarray] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Estimate hour-specific linear calibration coefficients ``(a_h, b_h)``.
    The fit is computed in population space using least squares over full-frame predictions.
    Each hour is calibrated independently after patch predictions are stitched together.
    """
    model.eval()
    dataset_mask = mask_override if mask_override is not None else base_dataset.get_output_mask()
    # Accumulate sufficient statistics separately for each hour of day.
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

            # Keep only finite pixels and, when available, pixels inside the output mask.
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

    calib: Dict[int, Tuple[float, float]] = {}
    for h, st in stats.items():
        if st["count"] <= 0 or st["sum_pred2"] <= eps:
            calib[h] = (1.0, 0.0)
            continue
        mean_pred = st["sum_pred"] / st["count"]
        mean_true = st["sum_true"] / st["count"]
        # Solve the least-squares line ``true = a * pred + b`` for this hour.
        a = (st["sum_pred_true"] - st["count"] * mean_pred * mean_true) / max(
            st["sum_pred2"] - st["count"] * mean_pred * mean_pred, eps
        )
        b = mean_true - a * mean_pred
        calib[h] = (float(a), float(b))
    return calib


class MixedAbsRelLoss(nn.Module):
    """
    Combined absolute and relative error with optional high-value weighting.
    ``alpha`` controls the relative-error term and ``gamma`` upweights large targets.
    ``eps`` avoids division by zero in the relative-error component.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 0.3, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_sqrt: torch.Tensor, target_sqrt: torch.Tensor) -> torch.Tensor:
        # Convert sqrt(pop) predictions back to population counts and clamp to non-negative.
        pred_pop = torch.clamp(pred_sqrt, min=0) ** 2
        true_pop = torch.clamp(target_sqrt, min=0) ** 2

        abs_err = torch.abs(pred_pop - true_pop)
        rel_err = abs_err / (true_pop + self.eps)

        mean_pop = torch.mean(true_pop)
        # Weight larger population values more strongly than small values.
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
        min_truth_pop: float = 1.0,
        clip_quantile: Optional[Tuple[float, float]] = None,
        hourly_calib: Optional[Dict[int, Tuple[float, float]]] = None,
) -> dict:
    """Run one training/validation/test epoch and compute population-domain metrics."""

    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss = 0.0
    total_rmse = 0.0
    total_tem = 0.0
    total_mae = 0.0
    total_mape = 0.0
    total_r2 = 0.0
    total_samples = 0  # Number of valid pixels used for metric aggregation.

    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            X_batch, Y_batch, hour_batch = batch
        else:
            X_batch, Y_batch = batch
            hour_batch = None

        # ---- Move data to the selected device ----
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        hour_batch = hour_batch.to(device) if hour_batch is not None else None

        # ---- Forward pass and optional backward pass ----
        if train_mode:
            optimizer.zero_grad()

        preds = model(X_batch)  # Direct regression in sqrt(pop) space.
        loss = criterion(preds, Y_batch)

        if train_mode:
            loss.backward()
            optimizer.step()

        # ---- Aggregate batch metrics ----
        with torch.no_grad():
            y_true_pop = torch.clamp(Y_batch, min=0) ** 2
            y_pred_pop = torch.clamp(preds, min=0) ** 2

            # Optionally apply hour-specific calibration during evaluation.
            if hourly_calib is not None and hour_batch is not None and phase == "test":
                for b in range(y_pred_pop.size(0)):
                    h = int(hour_batch[b].item())
                    a, b_coef = hourly_calib.get(h, (1.0, 0.0))
                    y_pred_pop[b] = torch.clamp(a * y_pred_pop[b] + b_coef, min=0.0)

            valid = y_true_pop > float(min_truth_pop)
            if valid.sum().item() == 0:
                continue
            y_t = y_true_pop[valid]
            y_p = y_pred_pop[valid]

            if clip_quantile is not None:
                q_low, q_high = clip_quantile
                ql = torch.quantile(y_t, q_low)
                qh = torch.quantile(y_t, q_high)
                keep = (y_t >= ql) & (y_t <= qh)
                if keep.sum().item() == 0:
                    continue
                y_t = y_t[keep]
                y_p = y_p[keep]

            valid_count = y_t.numel()
            batch_rmse = rmse(y_t, y_p)
            batch_tem = tem_metric(y_t, y_p)
            batch_mae = F.l1_loss(y_p, y_t)
            batch_mape = torch.mean(torch.abs((y_p - y_t) / (y_t + 1e-6)))
            mean_t = torch.mean(y_t)
            ss_tot = torch.sum((y_t - mean_t) ** 2)
            ss_res = torch.sum((y_t - y_p) ** 2)
            batch_r2 = 0.0 if ss_tot < 1e-6 else (1.0 - ss_res / (ss_tot + 1e-6))

        total_loss += loss.item() * valid_count
        total_rmse += batch_rmse.item() * valid_count
        total_tem += batch_tem.item() * valid_count
        total_mae += batch_mae.item() * valid_count
        total_mape += batch_mape.item() * valid_count
        total_r2 += float(batch_r2) * valid_count
        total_samples += valid_count

        # ---- Batch-level progress logging (every 50 batches) ----
        if str(phase).lower() != "test" and batch_idx % 50 == 0:
            print(f"[{phase}] Epoch {epoch:03d} | Batch {batch_idx + 1}/{num_batches}")

    return {
        "loss": total_loss / max(1, total_samples),
        "rmse": total_rmse / max(1, total_samples),
        "tem": total_tem / max(1, total_samples),
        "mae": total_mae / max(1, total_samples),
        "mape": total_mape / max(1, total_samples),
        "r2": total_r2 / max(1, total_samples),
    }


def train_model(
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        epochs: int,
        patience: int,
        best_loss_path: str,
        best_rmse_path: str,
        best_val_tem_path: Optional[str] = None,
        checkpoint_map: Optional[Dict[int, Path]] = None,
) -> List[dict]:
    """
    Train the regression model with early stopping and checkpointing.
    The optimization objective is ``MixedAbsRelLoss``.
    Early stopping and checkpoint selection use validation metrics only.
    The outer-domain test split is evaluated after training is finished.
    """

    print(f"Starting training for {epochs} epochs...")

    # Use the mixed absolute/relative population loss for the main optimization objective.
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
    best_val_rmse = float("inf")
    best_val_tem = float("inf")
    patience_counter = 0

    # Ensure the checkpoint directories exist before training starts.
    os.makedirs(os.path.dirname(best_loss_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_rmse_path), exist_ok=True)
    if best_val_tem_path is not None:
        os.makedirs(os.path.dirname(best_val_tem_path), exist_ok=True)

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

        # ---------- Save epoch metrics ----------
        record = {
            "epoch": ep,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_tem": train_metrics["tem"],
            "train_mae": train_metrics["mae"],
            "train_mape": train_metrics["mape"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_tem": val_metrics["tem"],
            "val_mae": val_metrics["mae"],
            "val_mape": val_metrics["mape"],
            "val_r2": val_metrics["r2"],
        }
        history.append(record)

        # ---------- LR Scheduler ----------
        scheduler.step(record["val_loss"])

        print(
            f"Epoch {ep:03d} | "
            f"TrainLoss {record['train_loss']:.4f} "
            f"TrainTEM {record['train_tem']:.4f} "
            f"TrainMAE {record['train_mae']:.4f} "
            f"TrainMAPE {record['train_mape']:.4f} "
            f"TrainR2 {record['train_r2']:.4f} || "
            f"ValLoss {record['val_loss']:.4f} "
            f"ValTEM {record['val_tem']:.4f} "
            f"ValMAE {record['val_mae']:.4f} "
            f"ValMAPE {record['val_mape']:.4f} "
            f"ValR2 {record['val_r2']:.4f}"
        )

        # ---------- Save checkpoints for specific epochs ----------
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

        # best RMSE
        if record["val_rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = record["val_rmse"]
            torch.save(model.state_dict(), best_rmse_path)
            improved = True

        # best validation TEM
        if (
                best_val_tem_path is not None
                and record["val_tem"] < best_val_tem - 1e-6
        ):
            best_val_tem = record["val_tem"]
            torch.save(model.state_dict(), best_val_tem_path)
            improved = True

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save metrics corresponding to the best validation-RMSE checkpoint.
    if history:
        best_record = min(history, key=lambda h: h["val_rmse"])
        best_metrics = {
            "epoch": best_record["epoch"],
            "val_tem": best_record["val_tem"],
            "val_rmse": best_record["val_rmse"],
            "val_mae": best_record["val_mae"],
            "val_mape": best_record["val_mape"],
            "val_r2": best_record["val_r2"],
        }
        metrics_path = Path(best_rmse_path).with_name("best_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, ensure_ascii=False, indent=2)
        print(f"[info] Saved best metrics to {metrics_path}")

    if history and best_val_tem_path is not None:
        best_val_record = min(history, key=lambda h: h["val_tem"])
        best_val_metrics = {
            "epoch": best_val_record["epoch"],
            "val_tem": best_val_record["val_tem"],
            "val_rmse": best_val_record["val_rmse"],
            "val_mae": best_val_record["val_mae"],
            "val_mape": best_val_record["val_mape"],
            "val_r2": best_val_record["val_r2"],
        }
        val_metrics_path = Path(best_val_tem_path).with_name("best_val_tem_metrics.json")
        val_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(val_metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_val_metrics, f, ensure_ascii=False, indent=2)
        print(f"[info] Saved best validation TEM metrics to {val_metrics_path}")

    return history


from datetime import datetime as _dt


def predict_one_day(
        model: nn.Module,
        base_dataset: "HourlyPopulationDataset",
        device: torch.device,
        target_date_str: str,
        output_dir: Path,
        # patch_size: int = 128,  # Enable patch-based prediction with larger tiles if needed.
        patch_size: int = 96,
        overlap: int = 48,
        hourly_calib: Optional[Dict[int, Tuple[float, float]]] = None,
) -> None:
    """Predict a large spatial tile using patch-based inference."""
    target_date = _dt.strptime(target_date_str, "%Y-%m-%d").date()
    model.eval()

    # Find all samples whose target timestamp belongs to the requested date.
    selected: List[Tuple[int, _dt]] = []
    for idx, sample in enumerate(base_dataset.samples):
        key_y = sample["key_y"]
        meta = base_dataset._meta_by_key[key_y]
        dt = meta["dt"]
        if dt.date() == target_date:
            selected.append((idx, dt))

    if not selected:
        print(f"[warn] No samples were found for target date {target_date_str}.")
        return

    selected.sort(key=lambda x: x[1].hour)
    print(f"[info] Found {len(selected)} hourly samples for target date {target_date_str}.")

    pop_sqrt_min = getattr(base_dataset, "pop_sqrt_min", 0.0)
    pop_sqrt_max = getattr(base_dataset, "pop_sqrt_max", 1.0)
    # Prefer the sampling mask for output; otherwise fall back to the study-area mask.
    mask = base_dataset.get_output_mask()

    for sample_idx, dt in selected:
        hour = dt.hour
        X_full, _ = base_dataset.get_full_sequence(sample_idx)  # (T, H, W, C)

        print(f"Processing sample {sample_idx}, hour {hour:02d}: input shape {X_full.shape}")

        # Run patch-based inference over the full spatial tile.
        pred_sqrt = predict_full_image(
            model=model,
            X_full=X_full,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
        )

        # Convert to population space before calibration, then back to sqrt(pop) for export.
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
        print(f"[ok] Saved predicted population raster for {target_date_str} {hour:02d}:00 to {out_tif}")


def visualize_patch_prediction(model: nn.Module, dataset: HourlyPopulationDataset, device: torch.device) -> None:
    """Visualize prediction vs GT for the first patch generated from dataset."""
    _require_pyplot()
    patch_dataset = HourlyPopulationPatchDataset(dataset, augment=False)
    if len(patch_dataset) == 0:
        print("No patches available for visualization.")
        return
    X_patch, Y_patch, _ = patch_dataset[0]
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
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """Split (time, H, W, C) into overlapping spatial patches."""
    t, H, W, _ = X.shape
    if not patch_size:
        return [X], [(0, 0)]
    patches: List[np.ndarray] = []
    origins: List[Tuple[int, int]] = []
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
        patches_per_sample: int = 16,
        num_workers: int = 64,
        pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader for HourlyPopulationDataset."""
    patch_dataset = HourlyPopulationPatchDataset(
        base_dataset,
        augment=augment,
        sample_indices=sample_indices,
        patches_per_sample=patches_per_sample,
    )
    return DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def run_standard_mode() -> None:
    _require_geospatial_stack()
    set_global_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =============================
    # =============================
    # 1. Resolve project directories
    project_root = resolve_project_root()
    resolve_run_root()

    # =============================
    # =============================
    # 2. Dynamic population raster directory
    # =============================
    # Only ``*.tif`` rasters are scanned; auxiliary files are ignored safely.
    pop_dir = project_root / "data" / "mobile_phone_data"

    # =============================
    # =============================
    # 3. Static-variable directory and filenames
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
    # 4. Output directory
    shape_path_train = project_root / LOW_DENSITY_TRAIN_SHAPE_REL
    results_dir = resolve_output_dir_for_training_shape(project_root, shape_path_train)
    results_dir.mkdir(parents=True, exist_ok=True)

    # =============================
    # =============================
    # 5. Dataset configuration
    # =============================
    window_size = 24  # Use the previous 24 hours to predict the next hour.
    patch_size = 32
    overlap = 24  # Trade off spatial coverage against inference speed.
    infer_patch_size = 96
    infer_overlap = 48

    # Low-density ring: training mask uses random_polys.shp; outer mask uses the core urban boundary.
    shape_path_outer = project_root / OUTER_SHAPE_REL

    # Build the training dataset over the training-region mask.
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
        shape_path=str(shape_path_train),
    )

    print(f"Total time-window samples: {len(base_dataset)}")

    # =============================
    # 6. Temporal split over September samples
    # =============================
    # =============================
    combined_indices = []
    combined_dates = []

    for idx, sample in enumerate(base_dataset.samples):
        key_y = sample["key_y"]
        dt = base_dataset._meta_by_key[key_y]["dt"]
        in_sept = dt.month == 9
        if in_sept:
            combined_indices.append(idx)
            combined_dates.append(dt.date())  # Keep the date for temporal splitting.

    num_samples = len(combined_indices)

    # Collect the unique dates participating in the experiment.
    unique_dates = sorted(list(set(combined_dates)))
    print(f"Selected dates: {[d.strftime('%Y-%m-%d') for d in unique_dates]}")

    # Split dates in chronological order into 60% train, 20% val, and 20% test.
    total_days = len(unique_dates)
    train_days = int(total_days * 0.6)
    val_days = int(total_days * 0.2)

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

    # =============================
    # =============================
    # 6.5 Build the outer-domain inference dataset
    infer_dataset = HourlyPopulationDataset(
        pop_dir=str(pop_dir),
        static_dir=str(static_dir),
        static_files=static_files,
        window_size=window_size,
        patch_size=patch_size,
        overlap=overlap,
        rotation_angle=0,
        do_flip=False,
        random_seed=42,
        target_weekdays=None,
        shape_path=str(shape_path_outer),
    )
    train_mask = getattr(base_dataset, "study_mask_full", None)
    outer_mask = getattr(infer_dataset, "study_mask_full", None)
    if train_mask is None or outer_mask is None:
        raise RuntimeError("Failed to build masks from shapefiles.")
    if train_mask.shape != outer_mask.shape:
        raise RuntimeError(f"Mask shape mismatch: train={train_mask.shape}, outer={outer_mask.shape}")
    predict_mask = (outer_mask > 0).astype(np.float32) * (train_mask <= 0).astype(np.float32)
    # Configure masks so inference keeps outer-domain context but predicts only the ring area.
    infer_dataset.study_mask_full = outer_mask  # Feature context uses the outer-domain mask.
    infer_dataset.sampling_mask = predict_mask  # Sampling/output is restricted to the ring area.
    # Recompute normalization over the outer-domain dataset and share it with training data.
    all_keys_infer = [m["key"] for m in infer_dataset.time_meta]
    infer_dataset.recompute_pop_norm(keys=all_keys_infer)
    base_dataset.pop_sqrt_min = infer_dataset.pop_sqrt_min
    base_dataset.pop_sqrt_max = infer_dataset.pop_sqrt_max
    base_dataset.pop_sqrt_mean = infer_dataset.pop_sqrt_mean
    base_dataset.pop_sqrt_std = infer_dataset.pop_sqrt_std
    base_dataset._feat_cache.clear()

    batch_size = 4
    train_loader = create_dataloader(
        base_dataset,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
        sample_indices=train_indices,
        num_workers=64,
        pin_memory=False,
    )

    val_loader = create_dataloader(
        base_dataset,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        sample_indices=val_indices,
        num_workers=64,
        pin_memory=False,
    )

    test_loader = create_dataloader(
        infer_dataset,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        sample_indices=test_indices,
        num_workers=64,
        pin_memory=False,
    )
    

    # =============================
    # =============================
    # 7. Determine the number of model input channels
    C_static = len(static_files)
    in_channels = 1 + C_static + 5 + 1  # pop + static + time_encoding + prev_week_pop

    model = DSConvLSTMNet(in_channels=in_channels, use_transformer=True).to(device)

    best_loss_path = results_dir / "best_loss_Pop_HSTM_STTest.pt"
    best_rmse_path = results_dir / "best_rmse_Pop_HSTM_STTest.pt"
    best_val_tem_path = results_dir / "best_val_tem_Pop_HSTM_STTest.pt"

    # Target dates used when exporting predicted population rasters.
    target_dates: List[str] = [
        "2017-09-18",
        "2017-09-19",
        "2017-09-20",
        "2017-09-21",
        "2017-09-22",
        "2017-09-23",
        "2017-09-24",
        "2017-09-25",
        "2017-09-26",
        "2017-09-27",
        "2017-09-28"
    ]

    # =============================
    # =============================
    # 8. Train the model
    history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device,
        epochs=100,
        # epochs=120,
        # epochs=1000

        # patience=16,
        patience=18,
        # patience=3
        best_loss_path=str(best_loss_path),
        best_rmse_path=str(best_rmse_path),
        best_val_tem_path=str(best_val_tem_path),
        checkpoint_map=None,
    )

    # Training curves are intentionally not rendered in this project variant.

    final_ckpt_path = best_val_tem_path if best_val_tem_path.exists() else best_rmse_path
    print("[info] Final prediction checkpoint loaded.")
    model.load_state_dict(torch.load(str(final_ckpt_path), map_location=device))
    model.to(device)

    # =============================
    # =============================
    # 9. Estimate hourly calibration coefficients from the validation subset.
    hourly_calib = compute_hourly_calibration(
        model=model,
        base_dataset=base_dataset,
        device=device,
        sample_indices=val_indices,
        patch_size=infer_patch_size,
        overlap=infer_overlap,
        mask_override=base_dataset.get_output_mask(),
    )
    print(f"[info] Estimated hourly calibration (a,b): {hourly_calib}")

    # Re-evaluate the test loader with calibration for consistency with exported outputs.
    criterion_eval = MixedAbsRelLoss(alpha=0.5, gamma=0.3, eps=1e-6)
    calibrated_test_metrics = run_epoch(
        loader=test_loader,
        model=model,
        criterion=criterion_eval,
        optimizer=None,
        device=device,
        epoch=0,
        phase="test",
        min_truth_pop=1.0,
        clip_quantile=(0.01, 0.99),
        hourly_calib=hourly_calib,
    )
    print("[info] Calibrated outer-domain evaluation finished.")
    # Keep the calibrated test evaluation in the workflow, but do not persist
    # final_outer_test_metrics.json in the output folder.
    _ = calibrated_test_metrics

    # =============================
    # =============================
    # 11. Visualize predictions for the final model
    visualize_patch_prediction(model, base_dataset, device)

    # =============================
    # =============================
    # 12. Export daily predictions for the selected target dates
    for target_date_str in target_dates:
        target_date_tag = target_date_str.replace("-", "")
        day_pred_dir = results_dir / f"Pred_{target_date_tag}"
        day_pred_dir.mkdir(parents=True, exist_ok=True)
        predict_one_day(
            model=model,
            base_dataset=infer_dataset,
            device=device,
            target_date_str=target_date_str,
            output_dir=day_pred_dir,
            patch_size=infer_patch_size,
            overlap=infer_overlap,
            hourly_calib=hourly_calib,
        )


class RSLogCoshLoss(nn.Module):
    """Numerically stable log-cosh loss for search experiments."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = pred - target
        return torch.mean(x + F.softplus(-2.0 * x) - float(np.log(2.0)))


def rs_normalize_loss_name(loss_name: str) -> str:
    name = str(loss_name).strip().lower()
    if name in {"mixed_abs_rel", "mixedloss", "mixed_loss", "mixed"}:
        return "mixed_abs_rel"
    return name


def rs_build_loss(loss_name: str) -> nn.Module:
    name = rs_normalize_loss_name(loss_name)
    if name == "mixed_abs_rel":
        return MixedAbsRelLoss(alpha=0.5, gamma=0.6, eps=1e-6)
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    if name == "log_cosh":
        return RSLogCoshLoss()
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def get_static_files() -> List[str]:
    return [
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


def build_osit_experiment(
        project_root: Path,
        window_size: int,
        patch_size: int,
        overlap: int,
        random_seed: int,
        shape_path_train: Path,
) -> dict:
    pop_dir = project_root / "data" / "mobile_phone_data"
    static_dir = project_root / "data" / "static_variables"
    static_files = get_static_files()
    shape_path_outer = project_root / OUTER_SHAPE_REL

    base_dataset = HourlyPopulationDataset(
        pop_dir=str(pop_dir),
        static_dir=str(static_dir),
        static_files=static_files,
        window_size=window_size,
        patch_size=patch_size,
        overlap=overlap,
        rotation_angle=90,
        do_flip=True,
        random_seed=random_seed,
        target_weekdays=None,
        shape_path=str(shape_path_train),
    )

    combined_indices = []
    combined_dates = []
    for idx, sample in enumerate(base_dataset.samples):
        key_y = sample["key_y"]
        dt = base_dataset._meta_by_key[key_y]["dt"]
        if dt.month == 9:
            combined_indices.append(idx)
            combined_dates.append(dt.date())

    unique_dates = sorted(list(set(combined_dates)))
    total_days = len(unique_dates)
    train_days = int(total_days * 0.6)
    val_days = int(total_days * 0.2)
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

    infer_dataset = HourlyPopulationDataset(
        pop_dir=str(pop_dir),
        static_dir=str(static_dir),
        static_files=static_files,
        window_size=window_size,
        patch_size=patch_size,
        overlap=overlap,
        rotation_angle=0,
        do_flip=False,
        random_seed=random_seed,
        target_weekdays=None,
        shape_path=str(shape_path_outer),
    )
    train_mask = getattr(base_dataset, "study_mask_full", None)
    outer_mask = getattr(infer_dataset, "study_mask_full", None)
    if train_mask is None or outer_mask is None:
        raise RuntimeError("Failed to build masks from shapefiles.")
    if train_mask.shape != outer_mask.shape:
        raise RuntimeError(f"Mask shape mismatch: train={train_mask.shape}, outer={outer_mask.shape}")
    predict_mask = (outer_mask > 0).astype(np.float32) * (train_mask <= 0).astype(np.float32)
    infer_dataset.study_mask_full = outer_mask
    infer_dataset.sampling_mask = predict_mask
    all_keys_infer = [m["key"] for m in infer_dataset.time_meta]
    infer_dataset.recompute_pop_norm(keys=all_keys_infer)
    base_dataset.pop_sqrt_min = infer_dataset.pop_sqrt_min
    base_dataset.pop_sqrt_max = infer_dataset.pop_sqrt_max
    base_dataset.pop_sqrt_mean = infer_dataset.pop_sqrt_mean
    base_dataset.pop_sqrt_std = infer_dataset.pop_sqrt_std
    base_dataset._feat_cache.clear()

    return {
        "base_dataset": base_dataset,
        "infer_dataset": infer_dataset,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "in_channels": 1 + len(static_files) + 5 + 1,
    }


def rs_train_model_trial(
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        epochs: int,
        patience: int,
        lr: float,
        weight_decay: float,
        criterion: nn.Module,
        best_loss_path: Path,
        best_rmse_path: Path,
        best_val_tem_path: Path,
) -> List[dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    history: List[dict] = []
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    best_val_tem = float("inf")
    patience_counter = 0
    for ckpt in [best_loss_path, best_rmse_path, best_val_tem_path]:
        ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        train_metrics = run_epoch(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=ep,
            phase="train",
        )
        val_metrics = run_epoch(
            loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch=ep,
            phase="val",
        )

        record = {
            "epoch": ep,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_metrics["loss"]),
            "train_rmse": float(train_metrics["rmse"]),
            "train_tem": float(train_metrics["tem"]),
            "train_mae": float(train_metrics["mae"]),
            "train_mape": float(train_metrics["mape"]),
            "train_r2": float(train_metrics["r2"]),
            "val_loss": float(val_metrics["loss"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_tem": float(val_metrics["tem"]),
            "val_mae": float(val_metrics["mae"]),
            "val_mape": float(val_metrics["mape"]),
            "val_r2": float(val_metrics["r2"]),
        }
        history.append(record)
        scheduler.step(record["val_loss"])

        improved = False
        if record["val_loss"] < best_val_loss - 1e-6:
            best_val_loss = record["val_loss"]
            torch.save(model.state_dict(), str(best_loss_path))
            improved = True
        if record["val_rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = record["val_rmse"]
            torch.save(model.state_dict(), str(best_rmse_path))
            improved = True
        if record["val_tem"] < best_val_tem - 1e-6:
            best_val_tem = record["val_tem"]
            torch.save(model.state_dict(), str(best_val_tem_path))
            improved = True

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[epoch {ep:03d}] lr={record['lr']:.2e} | "
            f"Train(TEM/MAE/R2)=({record['train_tem']:.4f}/{record['train_mae']:.4f}/{record['train_r2']:.4f}) | "
            f"Val(TEM/MAE/R2)=({record['val_tem']:.4f}/{record['val_mae']:.4f}/{record['val_r2']:.4f}) | "
            f"ES={patience_counter}/{patience}"
        )
        if patience_counter >= patience:
            print(f"[early-stop] Triggered at epoch {ep:03d}.")
            break

    return history


def rs_build_unique_search_pool(
        window_candidates: Sequence[int],
        lr_candidates: Sequence[float],
        patches_candidates: Sequence[int],
        loss_candidates: Sequence[str],
) -> List[dict]:
    pool: List[dict] = []
    for ws in window_candidates:
        for lr in lr_candidates:
            for pps in patches_candidates:
                for loss_name in loss_candidates:
                    pool.append(
                        {
                            "window_size": int(ws),
                            "lr": float(lr),
                            "patches_per_sample": int(pps),
                            "loss_name": rs_normalize_loss_name(loss_name),
                        }
                    )
    return pool


def rs_save_results_csv(rows: List[dict], out_csv: Path) -> None:
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rs_parse_args(argv: Optional[List[str]] = None):
    import argparse

    p = argparse.ArgumentParser(description="Random-search variant of OSIT that selects the best validation TEM for final prediction.")
    general = p.add_argument_group("general")
    general.add_argument("--trials", type=int, default=8)
    general.add_argument("--epochs", type=int, default=100)
    general.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    general.add_argument(
        "--training-shapefile",
        type=str,
        default=str(HIGH_DENSITY_TRAIN_SHAPE_REL),
        help="Training-region shapefile. random_polys_other.shp routes to high-density random-search output.",
    )
    search = p.add_argument_group("search-space")
    search.add_argument("--window-search", nargs="+", type=int, default=list(DEFAULT_WINDOW_SEARCH))
    search.add_argument("--lr-search", nargs="+", type=float, default=list(DEFAULT_LR_SEARCH))
    search.add_argument("--patches-per-sample-search", nargs="+", type=int, default=list(DEFAULT_PATCHES_PER_SAMPLE_SEARCH))
    search.add_argument("--loss-search", nargs="+", type=str, default=list(DEFAULT_LOSS_SEARCH))

    fixed_trial = p.add_argument_group("fixed-first-trial")
    fixed_trial.add_argument("--fixed-window-size", type=int, default=24)
    fixed_trial.add_argument("--fixed-lr", type=float, default=1e-4)
    fixed_trial.add_argument("--fixed-patches-per-sample", type=int, default=16)
    fixed_trial.add_argument("--fixed-loss-name", type=str, default="mixed_abs_rel")

    training = p.add_argument_group("training")
    training.add_argument("--patch-size", type=int, default=32)
    training.add_argument("--overlap", type=int, default=24)
    training.add_argument("--prediction-patch-size", type=int, default=96)
    training.add_argument("--prediction-overlap", type=int, default=48)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--num-workers", type=int, default=8)
    training.add_argument("--patience", type=int, default=18)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--target-dates", nargs="+", default=list(DEFAULT_TARGET_DATES))
    return p.parse_args(argv)


def run_random_search_mode(argv: Optional[List[str]] = None) -> None:
    _require_geospatial_stack()
    args = rs_parse_args(argv)
    project_root = resolve_project_root()
    resolve_run_root()
    shape_path_train = _resolve_project_path(project_root, args.training_shapefile)
    low_shape = (project_root / LOW_DENSITY_TRAIN_SHAPE_REL).resolve()
    if shape_path_train.resolve() == low_shape:
        raise SystemExit(
            "random_polys.shp is configured for non-random-search low-density runs. "
            "Use: python code\\Pop-HSTM_STTest.py --mode-standard"
        )
    out_dir = resolve_output_dir_for_training_shape(project_root, shape_path_train)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    window_candidates = list(dict.fromkeys(args.window_search))
    lr_candidates = list(dict.fromkeys(args.lr_search))
    patches_candidates = list(dict.fromkeys(args.patches_per_sample_search))
    loss_candidates = [rs_normalize_loss_name(v) for v in list(dict.fromkeys(args.loss_search))]

    fixed_hp = {
        "window_size": int(args.fixed_window_size),
        "lr": float(args.fixed_lr),
        "patches_per_sample": int(args.fixed_patches_per_sample),
        "loss_name": rs_normalize_loss_name(args.fixed_loss_name),
    }
    search_pool = rs_build_unique_search_pool(
        window_candidates=window_candidates,
        lr_candidates=lr_candidates,
        patches_candidates=patches_candidates,
        loss_candidates=loss_candidates,
    )
    search_pool = [hp for hp in search_pool if hp != fixed_hp]
    rng.shuffle(search_pool)

    schedule: List[Tuple[str, dict]] = [("fixed", dict(fixed_hp))]
    random_needed = max(int(args.trials) - 1, 0)
    if random_needed > len(search_pool):
        random_needed = len(search_pool)
    schedule.extend(("random", hp) for hp in search_pool[:random_needed])

    data_cache: Dict[int, dict] = {}
    all_trials: List[dict] = []
    best_trial: Optional[dict] = None
    best_val_metric = float("inf")

    for trial_idx, (trial_source, hp) in enumerate(schedule, start=1):
        set_global_determinism(args.seed + trial_idx)
        trial_dir = out_dir / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[trial {trial_idx:03d}][{trial_source}] hparams={hp}")

        ws = int(hp["window_size"])
        if ws not in data_cache:
            data_cache[ws] = build_osit_experiment(
                project_root=project_root,
                window_size=ws,
                patch_size=args.patch_size,
                overlap=args.overlap,
                random_seed=args.seed,
                shape_path_train=shape_path_train,
            )
            print(f"[cache] prepared datasets for window_size={ws}")
        bundle = data_cache[ws]

        train_loader = create_dataloader(
            base_dataset=bundle["base_dataset"],
            batch_size=args.batch_size,
            shuffle=True,
            augment=True,
            sample_indices=bundle["train_indices"],
            patches_per_sample=hp["patches_per_sample"],
            num_workers=args.num_workers,
            pin_memory=False,
        )
        val_loader = create_dataloader(
            base_dataset=bundle["base_dataset"],
            batch_size=args.batch_size,
            shuffle=False,
            augment=False,
            sample_indices=bundle["val_indices"],
            patches_per_sample=hp["patches_per_sample"],
            num_workers=args.num_workers,
            pin_memory=False,
        )
        model = DSConvLSTMNet(in_channels=bundle["in_channels"], use_transformer=True).to(device)
        criterion = rs_build_loss(hp["loss_name"])
        best_loss_path = trial_dir / "best_loss.pt"
        best_rmse_path = trial_dir / "best_rmse.pt"
        best_val_path = trial_dir / "best_val_tem.pt"
        history = rs_train_model_trial(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            lr=hp["lr"],
            weight_decay=args.weight_decay,
            criterion=criterion,
            best_loss_path=best_loss_path,
            best_rmse_path=best_rmse_path,
            best_val_tem_path=best_val_path,
        )
        if not history:
            all_trials.append({"trial": trial_idx, "status": "failed_no_history", **hp})
            continue

        best_record = min(history, key=lambda x: x["val_tem"])
        trial_row = {
            "trial": trial_idx,
            "trial_source": trial_source,
            "status": "ok",
            "selection_metric": "val_tem",
            "best_epoch_by_val_tem": int(best_record["epoch"]),
            "val_tem": float(best_record["val_tem"]),
            "val_rmse": float(best_record["val_rmse"]),
            "val_mae": float(best_record["val_mae"]),
            "val_mape": float(best_record["val_mape"]),
            "val_r2": float(best_record["val_r2"]),
            "best_val_checkpoint": str(best_val_path),
            **hp,
        }
        save_json(history, trial_dir / "trial_history.json")
        save_json(trial_row, trial_dir / "trial_result.json")
        all_trials.append(trial_row)

        if trial_row["val_tem"] < best_val_metric:
            best_val_metric = trial_row["val_tem"]
            best_trial = trial_row
            print(f"[trial {trial_idx:03d}] new best validation TEM={best_val_metric:.6f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ranked_trials = sorted([r for r in all_trials if r.get("status") == "ok"], key=lambda x: x["val_tem"])
    rs_save_results_csv(all_trials, out_dir / "random_search_all_trials.csv")
    rs_save_results_csv(ranked_trials, out_dir / "random_search_ranked_by_val_tem.csv")

    summary: Dict[str, Any] = {
        "mode": "random_search_osit",
        "selection_metric": "val_tem",
        "n_trials_requested": int(args.trials),
        "n_trials_run": len(schedule),
        "seed": int(args.seed),
        "training_shapefile": str(shape_path_train),
        "output_dir": str(out_dir),
        "fixed_trial": fixed_hp,
        "search_space": {
            "window_size": window_candidates,
            "lr": lr_candidates,
            "patches_per_sample": patches_candidates,
            "loss_name": loss_candidates,
        },
        "best_trial": best_trial,
    }

    if best_trial is not None:
        best_ws = int(best_trial["window_size"])
        bundle = data_cache[best_ws]
        best_model = DSConvLSTMNet(in_channels=bundle["in_channels"], use_transformer=True).to(device)
        best_model.load_state_dict(torch.load(best_trial["best_val_checkpoint"], map_location=device))
        hourly_calib = compute_hourly_calibration(
            model=best_model,
            base_dataset=bundle["base_dataset"],
            device=device,
            sample_indices=bundle["val_indices"],
            patch_size=args.prediction_patch_size,
            overlap=args.prediction_overlap,
            mask_override=bundle["base_dataset"].get_output_mask(),
        )
        best_pred_root = out_dir
        best_pred_root.mkdir(parents=True, exist_ok=True)
        save_json({str(k): [float(v[0]), float(v[1])] for k, v in hourly_calib.items()}, best_pred_root / "hourly_calibration.json")
        for target_date_str in args.target_dates:
            target_date_tag = target_date_str.replace("-", "")
            pred_dir = best_pred_root / f"Pred_{target_date_tag}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            predict_one_day(
                model=best_model,
                base_dataset=bundle["infer_dataset"],
                device=device,
                target_date_str=target_date_str,
                output_dir=pred_dir,
                patch_size=args.prediction_patch_size,
                overlap=args.prediction_overlap,
                hourly_calib=hourly_calib,
            )
        summary["best_predictions_root"] = str(best_pred_root)
        summary["best_hourly_calibration_json"] = str(best_pred_root / "hourly_calibration.json")

        test_loader = create_dataloader(
            base_dataset=bundle["infer_dataset"],
            batch_size=args.batch_size,
            shuffle=False,
            augment=False,
            sample_indices=bundle["test_indices"],
            patches_per_sample=int(best_trial["patches_per_sample"]),
            num_workers=args.num_workers,
            pin_memory=False,
        )
        criterion_eval = rs_build_loss(best_trial["loss_name"])
        final_test_metrics = run_epoch(
            loader=test_loader,
            model=best_model,
            criterion=criterion_eval,
            optimizer=None,
            device=device,
            epoch=0,
            phase="test",
            min_truth_pop=1.0,
            clip_quantile=(0.01, 0.99),
            hourly_calib=hourly_calib,
        )
        # Keep the final test evaluation in the workflow, but do not persist
        # final_outer_test_metrics.json in the output folder.
        _ = final_test_metrics

    save_json(summary, out_dir / "random_search_summary.json")
    if best_trial is not None:
        print(f"[best] trial={best_trial['trial']} val_tem={best_trial['val_tem']:.6f}")
        print(f"[best] predictions saved under: {summary.get('best_predictions_root')}")
    else:
        print("[best] no successful trial.")


def main() -> None:
    run_random_search_mode()


if __name__ == "__main__":
    import sys

    if "--mode-standard" in sys.argv:
        std_argv = [a for a in sys.argv[1:] if a != "--mode-standard"]
        if std_argv:
            raise SystemExit("Pop-HSTM_STTest.py standard mode does not accept extra CLI args.")
        run_standard_mode()
    else:
        run_random_search_mode(sys.argv[1:])










