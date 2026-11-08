from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from .config import (
    DEFAULT_FINAL_RESULTS_DIR,
    DEFAULT_OVERLAP,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PREDICTION_OVERLAP,
    DEFAULT_PREDICTION_PATCH_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET_DATES,
    DEFAULT_WINDOW_SIZE,
)
from .model import DSConvLSTMNet, build_model_input_channels
from .runtime import (
    collect_model_summary,
    collect_runtime_environment,
    flatten_hourly_runtime_records,
    save_runtime_report,
    summarize_values,
    timed_call,
    write_csv_rows,
)
from .utils import numpy_sequence_to_tensor, resolve_project_root, set_global_determinism


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Pop-HSTM inference with CUDA synchronization, warm-up, repeats, and JSON/CSV output."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path(DEFAULT_FINAL_RESULTS_DIR) / "best_tem_Pop-HSTM.pt"),
        help="Checkpoint path, relative to the project root unless absolute.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_FINAL_RESULTS_DIR) / "runtime_benchmark"),
        help="Benchmark output directory, relative to the project root unless absolute.",
    )
    parser.add_argument("--target-dates", nargs="+", default=[DEFAULT_TARGET_DATES[0]])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--prediction-patch-size", type=int, default=DEFAULT_PREDICTION_PATCH_SIZE)
    parser.add_argument("--prediction-overlap", type=int, default=DEFAULT_PREDICTION_OVERLAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args(list(argv) if argv is not None else None)


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _find_first_sample(dataset, target_dates: Sequence[str]) -> int:
    target_set = set(target_dates)
    for sample_index, sample in enumerate(dataset.samples):
        target_dt = dataset._meta_by_key[sample["key_y"]]["dt"]
        if target_dt.strftime("%Y-%m-%d") in target_set:
            return sample_index
    raise RuntimeError(f"No prediction samples were found for target_dates={list(target_dates)}")


def _warm_up_model(
        model: torch.nn.Module,
        dataset,
        device: torch.device,
        target_dates: Sequence[str],
        patch_size: int,
        iterations: int,
) -> dict:
    sample_index = _find_first_sample(dataset, target_dates)
    full_sequence, _ = dataset.get_full_sequence(sample_index)
    height = min(int(patch_size), int(full_sequence.shape[1])) if patch_size else int(full_sequence.shape[1])
    width = min(int(patch_size), int(full_sequence.shape[2])) if patch_size else int(full_sequence.shape[2])
    patch = full_sequence[:, :height, :width, :]
    input_tensor = numpy_sequence_to_tensor(patch).unsqueeze(0).to(device)
    iteration_seconds: List[float] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(max(0, int(iterations))):
            _, elapsed_seconds, _ = timed_call(device, model, input_tensor)
            iteration_seconds.append(elapsed_seconds)
    return {
        "iterations": len(iteration_seconds),
        "patch_shape_t_h_w_c": [
            int(patch.shape[0]),
            int(patch.shape[1]),
            int(patch.shape[2]),
            int(patch.shape[3]),
        ],
        "iteration_seconds": iteration_seconds,
    }


def run_runtime_benchmark(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.warmup_iterations < 0:
        raise ValueError("--warmup-iterations cannot be negative")

    from .data import build_hourly_population_dataset, chronological_split_indices, recompute_norm_with_indices
    from .inference import predict_dates_to_root

    set_global_determinism(args.seed)
    project_root = resolve_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _project_path(project_root, args.checkpoint)
    output_dir = _project_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pop-HSTM checkpoint not found: {checkpoint_path}")

    def prepare_dataset():
        dataset_obj = build_hourly_population_dataset(
            project_root=project_root,
            window_size=DEFAULT_WINDOW_SIZE,
            patch_size=DEFAULT_PATCH_SIZE,
            overlap=DEFAULT_OVERLAP,
        )
        split = chronological_split_indices(dataset_obj, included_months=(9,))
        recompute_norm_with_indices(dataset_obj, split.train_indices)
        return dataset_obj, split

    (dataset, split), dataset_preparation_seconds, _ = timed_call(device, prepare_dataset)
    model = DSConvLSTMNet(in_channels=build_model_input_channels(), use_transformer=True).to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    warmup = _warm_up_model(
        model=model,
        dataset=dataset,
        device=device,
        target_dates=args.target_dates,
        patch_size=args.prediction_patch_size,
        iterations=args.warmup_iterations,
    )

    repeat_records: list[dict] = []
    per_hour_rows: list[dict] = []
    for repeat_index in range(1, args.repeats + 1):
        repeat_output_dir = output_dir / f"repeat_{repeat_index:02d}"
        date_records, elapsed_seconds, peak_cuda_memory_mib = timed_call(
            device,
            predict_dates_to_root,
            model=model,
            base_dataset=dataset,
            device=device,
            target_dates=args.target_dates,
            output_root=repeat_output_dir,
            patch_size=args.prediction_patch_size,
            overlap=args.prediction_overlap,
            hourly_calib=None,
            track_cuda_memory=True,
        )
        predicted_hours = sum(int(record.get("predicted_hours", 0)) for record in date_records)
        repeat_record = {
            "repeat": repeat_index,
            "predicted_dates": len([record for record in date_records if record.get("status") == "ok"]),
            "predicted_hours": predicted_hours,
            "end_to_end_seconds": elapsed_seconds,
            "model_inference_seconds": sum(
                float(record.get("model_inference_seconds", 0.0)) for record in date_records
            ),
            "input_load_seconds": sum(float(record.get("input_load_seconds", 0.0)) for record in date_records),
            "postprocess_and_write_seconds": sum(
                float(record.get("postprocess_and_write_seconds", 0.0)) for record in date_records
            ),
            "peak_cuda_memory_mib": peak_cuda_memory_mib,
            "date_records": date_records,
        }
        repeat_records.append(repeat_record)
        per_hour_rows.extend(flatten_hourly_runtime_records(repeat_index, date_records))
        print(
            f"[runtime] repeat={repeat_index}/{args.repeats}, hours={predicted_hours}, "
            f"end_to_end={elapsed_seconds:.3f}s, model_inference={repeat_record['model_inference_seconds']:.3f}s"
        )

    end_to_end_values = [record["end_to_end_seconds"] for record in repeat_records]
    inference_values = [record["model_inference_seconds"] for record in repeat_records]
    hours_values = [record["predicted_hours"] for record in repeat_records]
    nonzero_hours = [value for value in hours_values if value > 0]
    representative_hours = nonzero_hours[0] if nonzero_hours else 0
    summary = {
        "repeat_end_to_end_seconds": summarize_values(end_to_end_values),
        "repeat_model_inference_seconds": summarize_values(inference_values),
        "end_to_end_seconds_per_hour": summarize_values(
            [record["end_to_end_seconds"] / record["predicted_hours"] for record in repeat_records if record["predicted_hours"]]
        ),
        "model_inference_seconds_per_hour": summarize_values(
            [record["model_inference_seconds"] / record["predicted_hours"] for record in repeat_records if record["predicted_hours"]]
        ),
        "end_to_end_seconds_per_24_hours": summarize_values(
            [record["end_to_end_seconds"] * 24.0 / record["predicted_hours"] for record in repeat_records if record["predicted_hours"]]
        ),
        "model_inference_seconds_per_24_hours": summarize_values(
            [record["model_inference_seconds"] * 24.0 / record["predicted_hours"] for record in repeat_records if record["predicted_hours"]]
        ),
        "peak_cuda_memory_mib": summarize_values(
            [record["peak_cuda_memory_mib"] for record in repeat_records if record["peak_cuda_memory_mib"] is not None]
        ),
        "hours_per_repeat": representative_hours,
    }
    report = {
        "benchmark_name": "Pop-HSTM inference runtime benchmark",
        "timing_definition": {
            "clock": "time.perf_counter wall clock",
            "cuda_synchronization": "before and after every timed accelerator section",
            "warmup_excluded": True,
            "model_inference": "patch construction, host-to-device transfer, forward pass, and device-to-host transfer; excludes raster loading and GeoTIFF writing",
            "end_to_end": "input loading, model inference, calibration/post-processing, and GeoTIFF writing",
        },
        "environment": collect_runtime_environment(device),
        "model": {**collect_model_summary(model), "checkpoint": str(checkpoint_path)},
        "data": {
            "target_dates": list(args.target_dates),
            "window_size": DEFAULT_WINDOW_SIZE,
            "prediction_patch_size": args.prediction_patch_size,
            "prediction_overlap": args.prediction_overlap,
            "dataset_samples": len(dataset),
            "train_samples_used_for_normalization": len(split.train_indices),
            "dataset_preparation_seconds": dataset_preparation_seconds,
        },
        "protocol": {
            "warmup_iterations": args.warmup_iterations,
            "measured_repeats": args.repeats,
            "random_seed": args.seed,
            "hourly_calibration": "identity for timing; the production elementwise calibration has negligible cost",
        },
        "warmup": warmup,
        "summary": summary,
        "repeats": repeat_records,
    }
    save_runtime_report(report, output_dir / "runtime_report.json")
    write_csv_rows(per_hour_rows, output_dir / "runtime_per_hour.csv")
    write_csv_rows(
        [
            {
                "device": report["environment"].get("gpu_name", report["environment"]["cpu_model"]),
                "target_dates_per_repeat": len(args.target_dates),
                "hours_per_repeat": representative_hours,
                "repeats": args.repeats,
                "end_to_end_seconds_per_24h_mean": summary["end_to_end_seconds_per_24_hours"]["mean"],
                "end_to_end_seconds_per_24h_std": summary["end_to_end_seconds_per_24_hours"]["std"],
                "model_inference_seconds_per_24h_mean": summary["model_inference_seconds_per_24_hours"]["mean"],
                "model_inference_seconds_per_24h_std": summary["model_inference_seconds_per_24_hours"]["std"],
                "peak_cuda_memory_mib_mean": summary["peak_cuda_memory_mib"]["mean"],
            }
        ],
        output_dir / "runtime_summary.csv",
    )
    print(f"[runtime] Manuscript-ready summary saved to {output_dir / 'runtime_summary.csv'}")
