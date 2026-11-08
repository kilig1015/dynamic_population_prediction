from __future__ import annotations

import csv
import os
import platform
import sys
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, TypeVar

import torch

from .utils import save_json


T = TypeVar("T")


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work so wall-clock measurements are valid."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_call(
        timing_device: torch.device,
        function: Callable[..., T],
        *args: Any,
        track_cuda_memory: bool = False,
        **kwargs: Any,
) -> Tuple[T, float, Optional[float]]:
    """Run a callable and return its result, elapsed seconds, and peak CUDA memory in MiB."""
    synchronize_device(timing_device)
    if track_cuda_memory and timing_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(timing_device)
    started = perf_counter()
    result = function(*args, **kwargs)
    synchronize_device(timing_device)
    elapsed_seconds = perf_counter() - started
    peak_memory_mib: Optional[float] = None
    if track_cuda_memory and timing_device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mib = torch.cuda.max_memory_allocated(timing_device) / (1024.0 ** 2)
    return result, float(elapsed_seconds), peak_memory_mib


def collect_runtime_environment(device: torch.device) -> Dict[str, Any]:
    """Collect the hardware/software fields needed to interpret a runtime result."""
    environment: Dict[str, Any] = {
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device_type": device.type,
        "cpu_model": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "not set"),
        "command": " ".join(sys.argv),
    }
    try:
        import psutil  # Optional: the benchmark does not require this package.

        environment["physical_cpu_count"] = psutil.cpu_count(logical=False)
        environment["system_memory_gib"] = psutil.virtual_memory().total / (1024.0 ** 3)
    except ImportError:
        environment["physical_cpu_count"] = None
        environment["system_memory_gib"] = None
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        environment.update(
            {
                "gpu_name": properties.name,
                "gpu_count": torch.cuda.device_count(),
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_total_memory_gib": properties.total_memory / (1024.0 ** 3),
            }
        )
    return environment


def collect_model_summary(model: torch.nn.Module) -> Dict[str, int]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
    }


def summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    clean_values = [float(value) for value in values]
    if not clean_values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(clean_values),
        "std": pstdev(clean_values) if len(clean_values) > 1 else 0.0,
        "min": min(clean_values),
        "max": max(clean_values),
    }


def flatten_hourly_runtime_records(
        repeat_index: int,
        date_records: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for date_record in date_records:
        for hourly_record in date_record.get("hourly_records", []):
            rows.append(
                {
                    "repeat": int(repeat_index),
                    "target_date": date_record.get("target_date"),
                    **hourly_record,
                }
            )
    return rows


def write_csv_rows(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_runtime_report(report: Dict[str, Any], output_path: Path) -> None:
    save_json(report, output_path)
    print(f"[runtime] Saved runtime report to {output_path}")
