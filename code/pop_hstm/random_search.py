from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FINAL_LOSS_NAME,
    DEFAULT_FINAL_LR,
    DEFAULT_FINAL_PATCHES_PER_SAMPLE,
    DEFAULT_FINAL_SEARCH_OUT_DIR,
    DEFAULT_TIME_GENERALIZATION_RESULTS_DIR,
    DEFAULT_OVERLAP,
    DEFAULT_PATCH_SIZE,
    DEFAULT_PATIENCE,
    DEFAULT_PREDICTION_OVERLAP,
    DEFAULT_PREDICTION_PATCH_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET_DATES,
    DEFAULT_WINDOW_SIZE,
)
from .data import build_hourly_population_dataset, chronological_split_indices, create_dataloader, recompute_norm_with_indices
from .inference import compute_hourly_calibration, predict_dates_to_root
from .model import DSConvLSTMNet, build_model_input_channels
from .runtime import collect_model_summary, collect_runtime_environment, synchronize_device, timed_call
from .training import (
    rs_build_loss,
    rs_evaluate_loader_metrics,
    rs_normalize_loss_name,
    rs_train_model_with_trial_hparams,
)
from .utils import resolve_project_root, save_json, set_global_determinism

if TYPE_CHECKING:
    from .data import HourlyPopulationDataset

def rs_save_results_csv(rows: List[dict], out_csv: Path) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def rs_export_best_trial_predictions(
        out_dir: Path,
        best_trial: Optional[dict],
        target_dates: Sequence[str],
        time_generalization_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Copy the selected trial predictions to the final Pop-HSTM prediction folders."""
    if best_trial is None:
        return None

    trial_id = int(best_trial["trial"])
    source_root = out_dir / f"trial_{trial_id:03d}"
    exported_dirs: dict[str, str] = {}
    time_exported_dirs: dict[str, str] = {}

    for target_date in target_dates:
        target_date_tag = str(target_date).replace("-", "")
        src_dir = source_root / f"Pred_{target_date_tag}"
        dst_dir = out_dir / f"Pred_{target_date_tag}"
        if not src_dir.is_dir():
            raise FileNotFoundError(f"Best-trial prediction folder not found: {src_dir}")
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        exported_dirs[target_date_tag] = str(dst_dir)
        if time_generalization_dir is not None:
            time_dst_dir = time_generalization_dir / f"Pred_{target_date_tag}"
            shutil.copytree(src_dir, time_dst_dir, dirs_exist_ok=True)
            time_exported_dirs[target_date_tag] = str(time_dst_dir)

    return {
        "source_trial": trial_id,
        "source_root": str(source_root),
        "target_root": str(out_dir),
        "pred_dirs": exported_dirs,
        "time_generalization_root": str(time_generalization_dir) if time_generalization_dir is not None else None,
        "time_generalization_pred_dirs": time_exported_dirs,
    }

def rs_write_param_table_best(
        rows: List[dict],
        param_name: str,
        label_fn,
        out_csv: Path,
) -> None:
    grouped: dict = {}
    for r in rows:
        grouped.setdefault(r[param_name], []).append(r)

    out_rows: List[dict] = []
    for value, group in sorted(grouped.items(), key=lambda x: x[0]):
        best = min(group, key=lambda x: x["val_tem"])
        out_rows.append(
            {
                "Setting": label_fn(value),
                "R2": best["test_r2_at_best_val"],
                "TEM": best["test_tem_at_best_val"],
                "MAE": best["test_mae_at_best_val"],
                "MAPE": best["test_mape_at_best_val"],
                "SelectedTrial": best["trial"],
            }
        )
    rs_save_results_csv(out_rows, out_csv)

def rs_write_param_table_mean(
        rows: List[dict],
        param_name: str,
        label_fn,
        out_csv: Path,
) -> None:
    grouped: dict = {}
    for r in rows:
        grouped.setdefault(r[param_name], []).append(r)

    out_rows: List[dict] = []
    for value, group in sorted(grouped.items(), key=lambda x: x[0]):
        n = len(group)
        out_rows.append(
            {
                "Setting": label_fn(value),
                "R2": sum(g["test_r2_at_best_val"] for g in group) / n,
                "TEM": sum(g["test_tem_at_best_val"] for g in group) / n,
                "MAE": sum(g["test_mae_at_best_val"] for g in group) / n,
                "MAPE": sum(g["test_mape_at_best_val"] for g in group) / n,
                "NumTrials": n,
            }
        )
    rs_save_results_csv(out_rows, out_csv)

def rs_parse_args(argv: Optional[List[str]] = None):
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Keep the fixed-first random-search shell, but default the search space to the "
            "reported optimal final hyperparameter configuration."
        )
    )
    general = p.add_argument_group("general")
    general.add_argument("--trials", type=int, default=1)
    general.add_argument("--epochs", type=int, default=100)
    general.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    general.add_argument("--out-dir", type=str, default=DEFAULT_FINAL_SEARCH_OUT_DIR)

    search = p.add_argument_group("search-space")
    search.add_argument(
        "--window-search",
        nargs="+",
        type=int,
        default=[DEFAULT_WINDOW_SIZE],
        help=(
            "Compatibility option for documenting candidate time windows. "
            "Released trials always train with window_size=24."
        ),
    )
    search.add_argument("--lr-search", nargs="+", type=float, default=[DEFAULT_FINAL_LR])
    search.add_argument(
        "--patches-per-sample-search",
        nargs="+",
        type=int,
        default=[DEFAULT_FINAL_PATCHES_PER_SAMPLE],
    )
    search.add_argument("--loss-search", nargs="+", type=str, default=[DEFAULT_FINAL_LOSS_NAME])

    fixed_trial = p.add_argument_group("fixed-first-trial")
    fixed_trial.add_argument("--fixed-window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    fixed_trial.add_argument("--fixed-lr", type=float, default=DEFAULT_FINAL_LR)
    fixed_trial.add_argument(
        "--fixed-patches-per-sample",
        type=int,
        default=DEFAULT_FINAL_PATCHES_PER_SAMPLE,
    )
    fixed_trial.add_argument("--fixed-loss-name", type=str, default=DEFAULT_FINAL_LOSS_NAME)

    training = p.add_argument_group("training")
    training.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    training.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    training.add_argument("--num-workers", type=int, default=8)
    training.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    training.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    training.add_argument("--weight-decay", type=float, default=1e-4)

    prediction = p.add_argument_group("prediction")
    prediction.add_argument("--prediction-patch-size", type=int, default=DEFAULT_PREDICTION_PATCH_SIZE)
    prediction.add_argument("--prediction-overlap", type=int, default=DEFAULT_PREDICTION_OVERLAP)
    prediction.add_argument(
        "--target-dates",
        nargs="+",
        default=list(DEFAULT_TARGET_DATES),
    )
    return p.parse_args(argv)

def rs_build_dataset(project_root: Path, window_size: int, patch_size: int, overlap: int) -> HourlyPopulationDataset:
    return build_hourly_population_dataset(
        project_root=project_root,
        window_size=window_size,
        patch_size=patch_size,
        overlap=overlap,
    )

def rs_split_indices_by_date(dataset: HourlyPopulationDataset) -> Tuple[List[int], List[int], List[int]]:
    split = chronological_split_indices(dataset, included_months=(9,))
    return split.train_indices, split.val_indices, split.test_indices

def rs_recompute_norm_with_train_only(dataset: HourlyPopulationDataset, train_indices: Sequence[int]) -> None:
    recompute_norm_with_indices(dataset, train_indices)

def rs_predict_after_trial(
        model: nn.Module,
        dataset: HourlyPopulationDataset,
        device: torch.device,
        val_indices: Sequence[int],
        trial_dir: Path,
        target_dates: Sequence[str],
        prediction_patch_size: int,
        prediction_overlap: int,
) -> dict[int, tuple[float, float]]:
    """After a trial finishes, estimate hourly calibration and save raster predictions."""
    hourly_calib = compute_hourly_calibration(
        model=model,
        base_dataset=dataset,
        device=device,
        sample_indices=val_indices,
        patch_size=prediction_patch_size,
        overlap=prediction_overlap,
    )
    calib_json = trial_dir / "hourly_calibration.json"
    save_json({str(k): [float(v[0]), float(v[1])] for k, v in hourly_calib.items()}, calib_json)
    print(f"[trial] hourly calibration saved -> {calib_json}")

    predict_dates_to_root(
        model=model,
        base_dataset=dataset,
        device=device,
        target_dates=target_dates,
        output_root=trial_dir,
        patch_size=prediction_patch_size,
        overlap=prediction_overlap,
        hourly_calib=hourly_calib,
    )
    return hourly_calib

def rs_create_dataloader_with_patch_budget(
        base_dataset: HourlyPopulationDataset,
        batch_size: int,
        shuffle: bool,
        augment: bool,
        sample_indices: Optional[Sequence[int]],
        patches_per_sample: int,
        num_workers: int,
):
    return create_dataloader(
        base_dataset=base_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        augment=augment,
        sample_indices=sample_indices,
        patches_per_sample=patches_per_sample,
        num_workers=num_workers,
    )

def rs_build_unique_search_pool(
        window_candidates: Sequence[int],
        lr_candidates: Sequence[float],
        patches_candidates: Sequence[int],
        loss_candidates: Sequence[str],
) -> List[dict]:
    """Build all unique hyperparameter combinations (no replacement sampling)."""
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

def run_random_search_mode(argv: Optional[List[str]] = None) -> None:
    args = rs_parse_args(argv)
    project_root = resolve_project_root()
    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_determinism(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synchronize_device(device)
    search_started = perf_counter()
    rng = random.Random(args.seed)

    declared_window_candidates = list(dict.fromkeys(args.window_search))
    # Keep the time-window search option visible, but make released trials train
    # with the reported final setting for deterministic reproduction.
    window_candidates = [DEFAULT_WINDOW_SIZE]
    lr_candidates = list(dict.fromkeys(args.lr_search))
    patches_candidates = list(dict.fromkeys(args.patches_per_sample_search))
    loss_candidates = [rs_normalize_loss_name(v) for v in list(dict.fromkeys(args.loss_search))]
    if not window_candidates or not lr_candidates or not patches_candidates or not loss_candidates:
        raise ValueError("Search spaces cannot be empty.")

    data_cache: dict = {}
    in_channels = build_model_input_channels()
    all_trials: List[dict] = []
    best_trial: Optional[dict] = None
    best_val_tem = float("inf")
    model_summary: Optional[dict] = None

    fixed_hp = {
        "window_size": DEFAULT_WINDOW_SIZE,
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
    if not search_pool:
        print("[info] Default search space already equals the reported optimal configuration; only the fixed trial will run.")

    schedule: List[tuple[str, dict]] = [("fixed", dict(fixed_hp))]
    random_needed = max(int(args.trials) - 1, 0)
    if random_needed > len(search_pool):
        print(
            f"[warn] Requested total trials={args.trials} means random_needed={random_needed}, "
            f"but only {len(search_pool)} unique random combinations remain after excluding the fixed one. "
            f"Will run {1 + len(search_pool)} trials in total."
        )
        random_needed = len(search_pool)
    schedule.extend(("random", hp) for hp in search_pool[:random_needed])
    trials_run = len(schedule)

    print(
        f"[search] Start final-config search shell: total_trials={trials_run}, "
        f"fixed_first={fixed_hp}, random_following={max(trials_run - 1, 0)}, max_epochs={args.epochs}"
    )
    for t, (trial_source, hp) in enumerate(schedule, start=1):
        synchronize_device(device)
        trial_started = perf_counter()
        set_global_determinism(args.seed + t)
        trial_dir = out_dir / f"trial_{t:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[trial {t:03d}][{trial_source}] hparams={hp}")

        ws = int(hp["window_size"])
        if ws not in data_cache:
            dataset_ws = rs_build_dataset(project_root, ws, args.patch_size, args.overlap)
            train_idx_ws, val_idx_ws, test_idx_ws = rs_split_indices_by_date(dataset_ws)
            rs_recompute_norm_with_train_only(dataset_ws, train_idx_ws)
            data_cache[ws] = (dataset_ws, train_idx_ws, val_idx_ws, test_idx_ws)
            print(f"[cache] prepared dataset for window_size={ws}")
        dataset, train_indices, val_indices, test_indices = data_cache[ws]

        train_loader = rs_create_dataloader_with_patch_budget(
            dataset, args.batch_size, True, True, train_indices, hp["patches_per_sample"], args.num_workers
        )
        val_loader = rs_create_dataloader_with_patch_budget(
            dataset, args.batch_size, False, False, val_indices, hp["patches_per_sample"], args.num_workers
        )
        test_loader = rs_create_dataloader_with_patch_budget(
            dataset, args.batch_size, False, False, test_indices, hp["patches_per_sample"], args.num_workers
        )

        model = DSConvLSTMNet(in_channels=in_channels, use_transformer=True).to(device)
        if model_summary is None:
            model_summary = collect_model_summary(model)
        criterion = rs_build_loss(hp["loss_name"])
        best_loss_path = trial_dir / "best_loss.pt"
        best_tem_path = trial_dir / "best_tem.pt"
        history, training_seconds, training_peak_cuda_memory_mib = timed_call(
            device,
            rs_train_model_with_trial_hparams,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            lr=hp["lr"],
            weight_decay=args.weight_decay,
            criterion=criterion,
            best_loss_path=best_loss_path,
            best_tem_path=best_tem_path,
            track_cuda_memory=True,
        )
        if not history:
            all_trials.append({"trial": t, "status": "failed_no_history"})
            continue

        best_record = min(history, key=lambda x: x["val_tem"])
        model.load_state_dict(torch.load(str(best_tem_path), map_location=device))
        val_eval, validation_evaluation_seconds, _ = timed_call(
            device, rs_evaluate_loader_metrics, model, val_loader, criterion, device
        )
        test_eval, test_evaluation_seconds, _ = timed_call(
            device, rs_evaluate_loader_metrics, model, test_loader, criterion, device
        )
        trial_row: dict = {
            "trial": t,
            "trial_source": trial_source,
            "status": "ok",
            "best_epoch": int(best_record["epoch"]),
            "val_tem": float(val_eval["TEM"]),
            "val_mae": float(val_eval["mae"]),
            "val_r2": float(val_eval["r2"]),
            "val_mape": float(val_eval["mape"]),
            "test_tem_at_best_val": float(test_eval["TEM"]),
            "test_mae_at_best_val": float(test_eval["mae"]),
            "test_r2_at_best_val": float(test_eval["r2"]),
            "test_mape_at_best_val": float(test_eval["mape"]),
            "training_seconds": training_seconds,
            "validation_evaluation_seconds": validation_evaluation_seconds,
            "test_evaluation_seconds": test_evaluation_seconds,
            "training_peak_cuda_memory_mib": training_peak_cuda_memory_mib,
            **hp,
        }

        print(f"[trial {t:03d}] start raster prediction and save outputs ...")
        prediction_seconds: Optional[float] = None
        try:
            _, prediction_seconds, prediction_peak_cuda_memory_mib = timed_call(
                device,
                rs_predict_after_trial,
                model=model,
                dataset=dataset,
                device=device,
                val_indices=val_indices,
                trial_dir=trial_dir,
                target_dates=args.target_dates,
                prediction_patch_size=args.prediction_patch_size,
                prediction_overlap=args.prediction_overlap,
                track_cuda_memory=True,
            )
            trial_row["prediction_status"] = "done"
            trial_row["prediction_days"] = len(args.target_dates)
            trial_row["prediction_seconds"] = prediction_seconds
            trial_row["prediction_peak_cuda_memory_mib"] = prediction_peak_cuda_memory_mib
            trial_row["hourly_calibration_json"] = str(trial_dir / "hourly_calibration.json")
        except Exception as exc:
            trial_row["prediction_status"] = f"failed: {exc}"
            trial_row["prediction_days"] = 0
            trial_row["prediction_seconds"] = prediction_seconds
            print(f"[trial {t:03d}] prediction failed, continue to next trial: {exc}")

        synchronize_device(device)
        trial_row["trial_end_to_end_seconds"] = perf_counter() - trial_started

        all_trials.append(trial_row)
        save_json(trial_row, trial_dir / "trial_result.json")

        if trial_row["val_tem"] < best_val_tem:
            best_val_tem = trial_row["val_tem"]
            best_trial = trial_row
            print(f"[trial {t:03d}] new best val_TEM={best_val_tem:.6f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ok_trials = [r for r in all_trials if r.get("status") == "ok"]
    ok_trials_sorted = sorted(ok_trials, key=lambda x: x["val_tem"])
    rs_save_results_csv(all_trials, out_dir / "random_search_all_trials.csv")
    rs_save_results_csv(ok_trials_sorted, out_dir / "random_search_ranked_by_val_tem.csv")
    if ok_trials_sorted:
        rs_write_param_table_best(ok_trials_sorted, "patches_per_sample", lambda v: f"patch{int(v)}", out_dir / "table_patches_per_sample_best.csv")
        rs_write_param_table_mean(ok_trials_sorted, "patches_per_sample", lambda v: f"patch{int(v)}", out_dir / "table_patches_per_sample_mean.csv")

    time_generalization_dir = project_root / DEFAULT_TIME_GENERALIZATION_RESULTS_DIR
    time_generalization_dir.mkdir(parents=True, exist_ok=True)
    best_prediction_export = rs_export_best_trial_predictions(
        out_dir=out_dir,
        best_trial=best_trial,
        target_dates=args.target_dates,
        time_generalization_dir=time_generalization_dir,
    )

    synchronize_device(device)
    search_end_to_end_seconds = perf_counter() - search_started
    summary = {
        "mode": "final_config_with_search_shell",
        "n_trials_requested": args.trials,
        "n_trials_run": trials_run,
        "n_random_trials_run": max(trials_run - 1, 0),
        "n_unique_random_combinations": len(search_pool),
        "seed": args.seed,
        "selection_metric": "val_tem (internal_key: val_tem)",
        "fixed_params": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "prediction_patch_size": args.prediction_patch_size,
            "prediction_overlap": args.prediction_overlap,
        },
        "fixed_hparams": fixed_hp,
        "searched_hparams": ["lr", "patches_per_sample", "loss_name"],
        "declared_window_search_candidates": declared_window_candidates,
        "time_window_behavior": f"window-search is retained for compatibility, but all released trials train with window_size={DEFAULT_WINDOW_SIZE}",
        "default_search_behavior": "search shell retained; released trials search only learning rate, patch budget, and loss function",
        "target_dates": list(args.target_dates),
        "best_trial": best_trial,
        "best_prediction_export": best_prediction_export,
        "runtime": {
            "search_end_to_end_seconds": search_end_to_end_seconds,
            "sum_successful_trial_training_seconds": sum(
                float(row.get("training_seconds", 0.0)) for row in ok_trials
            ),
            "sum_successful_trial_prediction_seconds": sum(
                float(row.get("prediction_seconds", 0.0) or 0.0) for row in ok_trials
            ),
            "environment": collect_runtime_environment(device),
            "model": model_summary,
        },
    }
    save_json(summary, out_dir / "random_search_summary.json")

    print("\n[search] done.")
    if best_trial is not None:
        print(f"[best] trial={best_trial['trial']}, val_TEM={best_trial['val_tem']:.6f}")
        print(f"[best] predictions exported to: {out_dir}")
        print(f"[best] summary saved in: {out_dir / 'random_search_summary.json'}")
    else:
        print("[best] no successful trial.")

