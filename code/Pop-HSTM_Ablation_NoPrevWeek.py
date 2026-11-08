import argparse
import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["USE_PATH_FOR_GDAL_PYTHON"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pop_hstm.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EPOCHS,
    DEFAULT_EPOCHS,
    DEFAULT_FINAL_PATCHES_PER_SAMPLE,
    DEFAULT_OVERLAP,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_PREDICTION_OVERLAP,
    DEFAULT_PREDICTION_PATCH_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STATIC_FILES,
    DEFAULT_TARGET_DATES,
    DEFAULT_WINDOW_SIZE,
)
from pop_hstm.data import (
    HourlyPopulationDataset,
    chronological_split_indices,
    create_dataloader,
    print_split_summary,
    recompute_norm_with_indices,
)
from pop_hstm.inference import compute_hourly_calibration, predict_dates_to_root
from pop_hstm.model import DSConvLSTMNet
from pop_hstm.training import train_model
from pop_hstm.utils import resolve_project_root, save_json, set_global_determinism


DEFAULT_OUTPUT_DIR = "output_results/dynamic_population_distribution/Pop-HSTM_Ablation_NoPrevWeek"


class NoPrevWeekHourlyPopulationDataset(HourlyPopulationDataset):
    """
    Pop-HSTM ablation dataset that removes the previous-week same-hour
    population channel from each per-time-step feature cube.

    Original channel layout:
        [pop_t, static_factors..., hour_sin, hour_cos, dow_sin, dow_cos,
         is_weekend, pop_t_minus_1week]

    Ablated channel layout:
        [pop_t, static_factors..., hour_sin, hour_cos, dow_sin, dow_cos,
         is_weekend]
    """

    def _get_feature_cube(self, key: str) -> np.ndarray:
        full_feature = super()._get_feature_cube(key)
        expected_full_channels = 1 + len(self.static_files) + 5 + 1
        if full_feature.shape[-1] != expected_full_channels:
            raise ValueError(
                "Unexpected full feature channel count before ablation: "
                f"got {full_feature.shape[-1]}, expected {expected_full_channels}."
            )
        return np.ascontiguousarray(full_feature[..., :-1])


def build_no_prev_week_dataset(
        project_root: Path,
        window_size: int,
        patch_size: int,
        overlap: int,
        static_files: Sequence[str] = DEFAULT_STATIC_FILES,
        random_seed: int = DEFAULT_RANDOM_SEED,
) -> NoPrevWeekHourlyPopulationDataset:
    """Create the Pop-HSTM dataset with the previous-week same-hour channel removed."""
    return NoPrevWeekHourlyPopulationDataset(
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


def build_ablation_input_channels(static_files: Sequence[str] = DEFAULT_STATIC_FILES) -> int:
    """Return input channel count after removing pop_t_minus_1week."""
    return 1 + len(static_files) + 5


def build_checkpoint_map(output_dir: Path, epochs: int) -> Dict[int, Path]:
    """Create fixed-epoch checkpoint directories under the ablation output root."""
    return {
        int(ep): output_dir / f"Pop-HSTM_NoPrevWeek_0925_{int(ep)}"
        for ep in DEFAULT_CHECKPOINT_EPOCHS
        if int(ep) <= int(epochs)
    }


def split_range_summary(split) -> dict:
    """Return a JSON-safe summary of chronological split ranges."""
    train_dates = sorted({d for d in split.combined_dates if d <= split.train_end_day})
    val_dates = sorted({d for d in split.combined_dates if split.train_end_day < d <= split.val_end_day})
    test_dates = sorted({d for d in split.combined_dates if d > split.val_end_day})
    return {
        "method": "chronological_by_target_date",
        "included_months": [9],
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "train_start": train_dates[0].isoformat() if train_dates else None,
        "train_end": train_dates[-1].isoformat() if train_dates else None,
        "val_start": val_dates[0].isoformat() if val_dates else None,
        "val_end": val_dates[-1].isoformat() if val_dates else None,
        "test_start": test_dates[0].isoformat() if test_dates else None,
        "test_end": test_dates[-1].isoformat() if test_dates else None,
        "train_samples": len(split.train_indices),
        "val_samples": len(split.val_indices),
        "test_samples": len(split.test_indices),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablation experiment for Pop-HSTM: remove the previous-week same-hour "
            "population input channel while keeping the remaining architecture and "
            "training protocol unchanged."
        )
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--patches-per-sample", type=int, default=DEFAULT_FINAL_PATCHES_PER_SAMPLE)
    parser.add_argument("--prediction-patch-size", type=int, default=DEFAULT_PREDICTION_PATCH_SIZE)
    parser.add_argument("--prediction-overlap", type=int, default=DEFAULT_PREDICTION_OVERLAP)
    parser.add_argument("--target-dates", nargs="+", default=list(DEFAULT_TARGET_DATES))
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default 0 is safer on Windows; increase for faster training if needed.",
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action="store_true",
        help="Also save fixed-epoch checkpoints at the same epochs as the main Pop-HSTM workflow.",
    )
    parser.add_argument(
        "--predict-epoch-checkpoints",
        action="store_true",
        help="Export raster predictions for fixed-epoch checkpoints. Requires --save-epoch-checkpoints.",
    )
    parser.add_argument(
        "--disable-hourly-calibration",
        action="store_true",
        help="Disable validation-set hourly linear calibration for exported rasters.",
    )
    return parser.parse_args()


def run_no_prev_week_ablation() -> None:
    args = parse_args()
    set_global_determinism(args.seed)

    project_root = resolve_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_files = list(DEFAULT_STATIC_FILES)
    in_channels = build_ablation_input_channels(static_files)

    print("[ablation] Pop-HSTM without previous-week same-hour population channel")
    print(f"[ablation] output_dir={output_dir}")
    print(f"[ablation] device={device}")
    print(f"[ablation] input_channels={in_channels} = 1 current_pop + {len(static_files)} static + 5 temporal")
    print("[ablation] removed channel: pop_t_minus_1week")

    dataset = build_no_prev_week_dataset(
        project_root=project_root,
        window_size=args.window_size,
        patch_size=args.patch_size,
        overlap=args.overlap,
        static_files=static_files,
        random_seed=args.seed,
    )
    split = chronological_split_indices(dataset, included_months=(9,))
    recompute_norm_with_indices(dataset, split.train_indices)
    print_split_summary(split)

    train_loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,
        sample_indices=split.train_indices,
        patches_per_sample=args.patches_per_sample,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
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
        sample_indices=split.test_indices,
        patches_per_sample=args.patches_per_sample,
        num_workers=args.num_workers,
    )

    config = {
        "experiment": "Pop-HSTM_Ablation_NoPrevWeek",
        "ablation": "remove_previous_week_same_hour_population_channel",
        "removed_channel": "pop_t_minus_1week",
        "remaining_channel_layout": [
            "pop_t",
            "static_factors",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "is_weekend",
        ],
        "input_channels": in_channels,
        "original_input_channels": in_channels + 1,
        "static_files": static_files,
        "window_size": args.window_size,
        "patch_size": args.patch_size,
        "overlap": args.overlap,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "patches_per_sample": args.patches_per_sample,
        "prediction_patch_size": args.prediction_patch_size,
        "prediction_overlap": args.prediction_overlap,
        "target_dates": list(args.target_dates),
        "seed": args.seed,
        "num_workers": args.num_workers,
        "split": split_range_summary(split),
        "hourly_calibration": "disabled" if args.disable_hourly_calibration else "validation_set_hour_specific_least_squares",
    }
    save_json(config, output_dir / "ablation_config.json")

    model = DSConvLSTMNet(in_channels=in_channels, use_transformer=True).to(device)
    best_loss_path = output_dir / "best_loss_Pop-HSTM_NoPrevWeek.pt"
    best_tem_path = output_dir / "best_tem_Pop-HSTM_NoPrevWeek.pt"
    checkpoint_map = build_checkpoint_map(output_dir, args.epochs) if args.save_epoch_checkpoints else None

    history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        best_loss_path=str(best_loss_path),
        best_tem_path=str(best_tem_path),
        checkpoint_map=checkpoint_map,
    )
    save_json(history, output_dir / "history.json")

    final_checkpoint = best_tem_path if best_tem_path.exists() else best_loss_path
    if not final_checkpoint.exists():
        raise FileNotFoundError(f"No trained checkpoint found: {final_checkpoint}")
    model.load_state_dict(torch.load(str(final_checkpoint), map_location=device))
    model.to(device)

    hourly_calib = None
    if not args.disable_hourly_calibration:
        hourly_calib = compute_hourly_calibration(
            model=model,
            base_dataset=dataset,
            device=device,
            sample_indices=split.val_indices,
            patch_size=args.prediction_patch_size,
            overlap=args.prediction_overlap,
        )
        save_json(
            {str(k): [float(v[0]), float(v[1])] for k, v in hourly_calib.items()},
            output_dir / "hourly_calibration.json",
        )
        print(f"[ablation] hourly calibration saved -> {output_dir / 'hourly_calibration.json'}")

    if args.predict_epoch_checkpoints:
        if not checkpoint_map:
            raise ValueError("--predict-epoch-checkpoints requires --save-epoch-checkpoints.")
        for ep, ckpt_dir in checkpoint_map.items():
            ckpt_path = ckpt_dir / f"model_epoch_{ep}.pt"
            if not ckpt_path.exists():
                print(f"[warn] checkpoint missing, skip prediction: {ckpt_path}")
                continue
            epoch_model = DSConvLSTMNet(in_channels=in_channels, use_transformer=True).to(device)
            epoch_model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
            predict_dates_to_root(
                model=epoch_model,
                base_dataset=dataset,
                device=device,
                target_dates=args.target_dates,
                output_root=ckpt_dir,
                patch_size=args.prediction_patch_size,
                overlap=args.prediction_overlap,
                hourly_calib=hourly_calib,
            )

    predict_dates_to_root(
        model=model,
        base_dataset=dataset,
        device=device,
        target_dates=args.target_dates,
        output_root=output_dir,
        patch_size=args.prediction_patch_size,
        overlap=args.prediction_overlap,
        hourly_calib=hourly_calib,
    )
    print(f"[ablation] final predictions saved under: {output_dir}")


if __name__ == "__main__":
    run_no_prev_week_ablation()
