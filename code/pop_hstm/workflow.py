from __future__ import annotations

import shutil
from pathlib import Path
from time import perf_counter
from typing import Dict, Sequence

import torch

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_EPOCHS,
    DEFAULT_EPOCHS,
    DEFAULT_FINAL_LOSS_NAME,
    DEFAULT_FINAL_LR,
    DEFAULT_FINAL_PATCHES_PER_SAMPLE,
    DEFAULT_FINAL_RESULTS_DIR,
    DEFAULT_TIME_GENERALIZATION_RESULTS_DIR,
    DEFAULT_OVERLAP,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_PREDICTION_OVERLAP,
    DEFAULT_PREDICTION_PATCH_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET_DATES,
    DEFAULT_WINDOW_SIZE,
    StandardRunSetup,
)
from .data import (
    build_hourly_population_dataset,
    chronological_split_indices,
    create_dataloader,
    print_split_summary,
    recompute_norm_with_indices,
)
from .inference import compute_hourly_calibration, predict_dates_to_root
from .model import DSConvLSTMNet, build_model_input_channels
from .runtime import (
    collect_model_summary,
    collect_runtime_environment,
    flatten_hourly_runtime_records,
    save_runtime_report,
    summarize_values,
    synchronize_device,
    timed_call,
    write_csv_rows,
)
from .training import train_model
from .utils import resolve_project_root, set_global_determinism
from .visualization import visualize_patch_prediction

def build_checkpoint_map(
        project_root: Path,
        checkpoint_epochs: Sequence[int] = DEFAULT_CHECKPOINT_EPOCHS,
) -> Dict[int, Path]:
    """Create output directories for fixed-epoch checkpoints."""
    results_root = project_root / DEFAULT_FINAL_RESULTS_DIR
    return {int(ep): results_root / f"Pop-HSTM_0925_{int(ep)}" for ep in checkpoint_epochs}

def copy_prediction_dirs_to_time_generalization(
        project_root: Path,
        source_root: Path,
        target_dates: Sequence[str],
) -> Dict[str, Path]:
    """Copy final Pop-HSTM prediction folders into the time-generalization result root."""
    target_root = project_root / DEFAULT_TIME_GENERALIZATION_RESULTS_DIR
    target_root.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, Path] = {}
    for target_date in target_dates:
        target_date_tag = str(target_date).replace("-", "")
        src_dir = source_root / f"Pred_{target_date_tag}"
        dst_dir = target_root / f"Pred_{target_date_tag}"
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Pop-HSTM prediction folder not found: {src_dir}")
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        copied[target_date_tag] = dst_dir
    return copied

def build_standard_run_setup(project_root: Path) -> StandardRunSetup:
    """Prepare dataset, splits, loaders, and output paths for the standard workflow."""
    results_dir = project_root / DEFAULT_FINAL_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_hourly_population_dataset(
        project_root=project_root,
        window_size=DEFAULT_WINDOW_SIZE,
        patch_size=DEFAULT_PATCH_SIZE,
        overlap=DEFAULT_OVERLAP,
    )
    split = chronological_split_indices(dataset, included_months=(9,))
    recompute_norm_with_indices(dataset, split.train_indices)

    train_loader = create_dataloader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        augment=True,
        sample_indices=split.train_indices,
        patches_per_sample=DEFAULT_FINAL_PATCHES_PER_SAMPLE,
    )
    val_loader = create_dataloader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        augment=False,
        sample_indices=split.val_indices,
        patches_per_sample=DEFAULT_FINAL_PATCHES_PER_SAMPLE,
    )
    test_loader = create_dataloader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        augment=False,
        sample_indices=split.test_indices,
        patches_per_sample=DEFAULT_FINAL_PATCHES_PER_SAMPLE,
    )

    return StandardRunSetup(
        results_dir=results_dir,
        dataset=dataset,
        split=split,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        in_channels=build_model_input_channels(),
        checkpoint_map=build_checkpoint_map(project_root),
        target_dates=list(DEFAULT_TARGET_DATES),
    )
def run_standard_workflow() -> None:
    """Run the standard training workflow and export daily raster predictions."""
    set_global_determinism(DEFAULT_RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synchronize_device(device)
    workflow_started = perf_counter()
    project_root = resolve_project_root()
    print(
        f"[standard] fixed hparams: window_size={DEFAULT_WINDOW_SIZE}, "
        f"lr={DEFAULT_FINAL_LR}, patches_per_sample={DEFAULT_FINAL_PATCHES_PER_SAMPLE}, "
        f"loss_name={DEFAULT_FINAL_LOSS_NAME}"
    )
    setup, dataset_preparation_seconds, _ = timed_call(device, build_standard_run_setup, project_root)
    results_dir = setup.results_dir
    base_dataset = setup.dataset
    split = setup.split
    train_indices = split.train_indices
    val_indices = split.val_indices
    print(f"Total time-window samples: {len(base_dataset)}")
    print_split_summary(split)

    train_loader = setup.train_loader
    val_loader = setup.val_loader
    test_loader = setup.test_loader
    in_channels = setup.in_channels
    model = DSConvLSTMNet(in_channels=in_channels, use_transformer=True).to(device)

    best_loss_path = results_dir / "best_loss_Pop-HSTM.pt"
    best_tem_path = results_dir / "best_tem_Pop-HSTM.pt"
    checkpoint_map = setup.checkpoint_map
    target_dates = setup.target_dates

    history, training_seconds, training_peak_cuda_memory_mib = timed_call(
        device,
        train_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        device=device,
        epochs=DEFAULT_EPOCHS,
        patience=DEFAULT_PATIENCE,
        best_loss_path=str(best_loss_path),
        best_tem_path=str(best_tem_path),
        checkpoint_map=checkpoint_map,
        track_cuda_memory=True,
    )
    hourly_calib, calibration_seconds, calibration_peak_cuda_memory_mib = timed_call(
        device,
        compute_hourly_calibration,
        model=model,
        base_dataset=base_dataset,
        device=device,
        sample_indices=val_indices,
        patch_size=DEFAULT_PREDICTION_PATCH_SIZE,
        overlap=DEFAULT_PREDICTION_OVERLAP,
        track_cuda_memory=True,
    )
    print(f"[info] Estimated hourly calibration (a,b): {hourly_calib}")

    checkpoint_prediction_seconds = 0.0
    for ep, ckpt_dir in checkpoint_map.items():
        ckpt_path = ckpt_dir / f"model_epoch_{ep}.pt"
        if not ckpt_path.exists():
            print(f"[warn] Checkpoint {ckpt_path} not found, skip prediction.")
            continue
        temp_model = DSConvLSTMNet(in_channels=in_channels, use_transformer=True).to(device)
        temp_model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        _, checkpoint_elapsed_seconds, _ = timed_call(
            device,
            predict_dates_to_root,
            model=temp_model,
            base_dataset=base_dataset,
            device=device,
            target_dates=target_dates,
            output_root=ckpt_dir,
            patch_size=DEFAULT_PREDICTION_PATCH_SIZE,
            overlap=DEFAULT_PREDICTION_OVERLAP,
            hourly_calib=hourly_calib,
        )
        checkpoint_prediction_seconds += checkpoint_elapsed_seconds
        del temp_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model.load_state_dict(torch.load(str(best_tem_path), map_location=device))
    model.to(device)
    visualize_patch_prediction(model, base_dataset, device)
    final_inference_records, final_inference_seconds, inference_peak_cuda_memory_mib = timed_call(
        device,
        predict_dates_to_root,
        model=model,
        base_dataset=base_dataset,
        device=device,
        target_dates=target_dates,
        output_root=results_dir,
        patch_size=DEFAULT_PREDICTION_PATCH_SIZE,
        overlap=DEFAULT_PREDICTION_OVERLAP,
        hourly_calib=hourly_calib,
        track_cuda_memory=True,
    )
    copied_dirs, output_copy_seconds, _ = timed_call(
        device,
        copy_prediction_dirs_to_time_generalization,
        project_root=project_root,
        source_root=results_dir,
        target_dates=target_dates,
    )
    print(f"[info] Copied final Pop-HSTM predictions to {project_root / DEFAULT_TIME_GENERALIZATION_RESULTS_DIR}")
    for date_tag, dst_dir in copied_dirs.items():
        print(f"[info] Time generalization prediction folder {date_tag}: {dst_dir}")

    synchronize_device(device)
    workflow_seconds = perf_counter() - workflow_started
    predicted_hours = sum(int(record.get("predicted_hours", 0)) for record in final_inference_records)
    hourly_rows = flatten_hourly_runtime_records(1, final_inference_records)
    report = {
        "benchmark_name": "Pop-HSTM standard training and inference runtime",
        "timing_definition": {
            "clock": "time.perf_counter wall clock",
            "cuda_synchronization": "before and after every timed accelerator section",
            "training": "all completed train, validation, and test passes plus checkpoint handling",
            "model_inference": "patch construction, host-to-device transfer, forward pass, and device-to-host transfer",
            "end_to_end_inference": "input loading, model inference, calibration/post-processing, and GeoTIFF writing",
        },
        "environment": collect_runtime_environment(device),
        "model": collect_model_summary(model),
        "data": {
            "window_size": DEFAULT_WINDOW_SIZE,
            "training_patch_size": DEFAULT_PATCH_SIZE,
            "training_overlap": DEFAULT_OVERLAP,
            "training_patches_per_sample": DEFAULT_FINAL_PATCHES_PER_SAMPLE,
            "batch_size": DEFAULT_BATCH_SIZE,
            "prediction_patch_size": DEFAULT_PREDICTION_PATCH_SIZE,
            "prediction_overlap": DEFAULT_PREDICTION_OVERLAP,
            "total_time_window_samples": len(base_dataset),
            "train_samples": len(split.train_indices),
            "validation_samples": len(split.val_indices),
            "test_samples": len(split.test_indices),
            "target_dates": list(target_dates),
        },
        "training": {
            "epochs_requested": DEFAULT_EPOCHS,
            "epochs_completed": len(history),
            "total_seconds": training_seconds,
            "train_pass_total_seconds": sum(float(record["train_seconds"]) for record in history),
            "validation_pass_total_seconds": sum(
                float(record["validation_seconds"]) for record in history
            ),
            "test_pass_total_seconds": sum(float(record["test_seconds"]) for record in history),
            "seconds_per_epoch_compute": summarize_values(
                [record["epoch_compute_seconds"] for record in history]
            ),
            "peak_cuda_memory_mib": training_peak_cuda_memory_mib,
        },
        "inference": {
            "predicted_hours": predicted_hours,
            "total_end_to_end_seconds": final_inference_seconds,
            "total_model_inference_seconds": sum(
                float(record.get("model_inference_seconds", 0.0)) for record in final_inference_records
            ),
            "end_to_end_seconds_per_hour": final_inference_seconds / predicted_hours if predicted_hours else None,
            "end_to_end_seconds_per_24_hours": (
                final_inference_seconds * 24.0 / predicted_hours if predicted_hours else None
            ),
            "peak_cuda_memory_mib": inference_peak_cuda_memory_mib,
            "date_records": final_inference_records,
        },
        "workflow_phases_seconds": {
            "dataset_preparation": dataset_preparation_seconds,
            "training": training_seconds,
            "hourly_calibration": calibration_seconds,
            "checkpoint_prediction_exports": checkpoint_prediction_seconds,
            "final_prediction_export": final_inference_seconds,
            "output_copy": output_copy_seconds,
            "full_reproduction_workflow": workflow_seconds,
        },
        "calibration_peak_cuda_memory_mib": calibration_peak_cuda_memory_mib,
    }
    save_runtime_report(report, results_dir / "runtime_report.json")
    write_csv_rows(history, results_dir / "runtime_training_epochs.csv")
    write_csv_rows(hourly_rows, results_dir / "runtime_inference_per_hour.csv")



# -----------------------------------------------------------------------------
# Fixed-first random search workflow
# -----------------------------------------------------------------------------

