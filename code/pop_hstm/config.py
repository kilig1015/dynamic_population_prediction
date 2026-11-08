from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from .data import HourlyPopulationDataset
    from torch.utils.data import DataLoader

DEFAULT_RANDOM_SEED = 42
DEFAULT_WINDOW_SIZE = 24
DEFAULT_PATCH_SIZE = 32
DEFAULT_OVERLAP = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 140
DEFAULT_PATIENCE = 18
DEFAULT_PREDICTION_PATCH_SIZE = 96
DEFAULT_PREDICTION_OVERLAP = 16
DEFAULT_FINAL_LR = 1e-4
DEFAULT_FINAL_PATCHES_PER_SAMPLE = 14
DEFAULT_FINAL_LOSS_NAME = "mixed_abs_rel"
DEFAULT_FINAL_RESULTS_DIR = "output_results/dynamic_population_distribution/Pop-HSTM"
DEFAULT_FINAL_SEARCH_OUT_DIR = "output_results/dynamic_population_distribution/Pop-HSTM"
DEFAULT_TIME_GENERALIZATION_RESULTS_DIR = "output_results/dynamic_population_distribution/Time_Generalization"
DEFAULT_CHECKPOINT_EPOCHS: Tuple[int, ...] = (60, 70, 80, 90, 100)
# Project-root markers for resolve_project_root().
# Use current repository layout markers so running from code/pop_hstm still resolves to repo root.
PROJECT_REQUIRED_DIRS = ("code", "data")
DEFAULT_STATIC_FILES: Tuple[str, ...] = (
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
)
DEFAULT_TARGET_DATES: Tuple[str, ...] = (
    "2017-09-25",
    "2017-09-26",
    "2017-09-27",
    "2017-09-28",
)

@dataclass(frozen=True)
class ChronologicalSplit:
    """Chronological dataset split and its date boundaries."""

    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]
    unique_dates: List[date]
    train_end_day: date
    val_end_day: date
    combined_indices: List[int]
    combined_dates: List[date]

@dataclass
class StandardRunSetup:
    """Prepared objects required by the standard training workflow."""

    results_dir: Path
    dataset: HourlyPopulationDataset
    split: ChronologicalSplit
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    in_channels: int
    checkpoint_map: Dict[int, Path]
    target_dates: List[str]

# Default hourly linear calibration coefficients (a, b). By default no correction is applied.
DEFAULT_HOURLY_CALIB: dict[int, tuple[float, float]] = {h: (1.0, 0.0) for h in range(24)}
