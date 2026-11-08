from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import DEFAULT_FINAL_LR
from .runtime import timed_call

def tem_metric(target_sqrt: torch.Tensor, pred_sqrt: torch.Tensor) -> torch.Tensor:
    """Compute TEM (ln(1+MSE)) in the model's sqrt(population) regression space."""
    with torch.no_grad():
        mse = F.mse_loss(pred_sqrt, target_sqrt)
        return torch.log1p(mse)


def rmse_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute RMSE in the model's sqrt(population) regression space."""
    with torch.no_grad():
        return torch.sqrt(F.mse_loss(y_pred, y_true))

def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error over batch and spatial dimensions."""
    with torch.no_grad():
        return F.l1_loss(y_pred, y_true)

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """R2 over batch and spatial dimensions, guarded against division by zero."""
    with torch.no_grad():
        # Flatten spatial and channel dimensions.
        y_t = y_true.view(y_true.size(0), -1)
        y_p = y_pred.view(y_pred.size(0), -1)
        mean_t = torch.mean(y_t, dim=1, keepdim=True)
        ss_tot = torch.sum((y_t - mean_t) ** 2, dim=1)
        ss_res = torch.sum((y_t - y_p) ** 2, dim=1)
        r2 = 1.0 - ss_res / (ss_tot + eps)
        # Average over the batch dimension.
        return torch.mean(r2)

class MixedAbsRelLoss(nn.Module):
    """
    Combined absolute + relative error with high-value weighting for population peaks.
    - alpha: relative-error weight; larger values emphasize proportional errors more.
    - gamma: exponent for high-value weighting; gamma>0 upweights large-population pixels.
    - eps: numerical stabilizer used to avoid division by zero.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 0.3, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred_sqrt: torch.Tensor, target_sqrt: torch.Tensor) -> torch.Tensor:
        # Convert sqrt(pop) back to the population domain and enforce non-negativity.
        pred_pop = torch.clamp(pred_sqrt, min=0) ** 2
        true_pop = torch.clamp(target_sqrt, min=0) ** 2

        abs_err = torch.abs(pred_pop - true_pop)
        rel_err = abs_err / (true_pop + self.eps)

        mean_pop = torch.mean(true_pop)
        # High-value weighting relative to the mean, avoiding unbounded amplification.
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
    """Single train/validation epoch loop for regression in the sqrt(population) prediction space."""

    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss = 0.0
    total_tem = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_samples = 0

    num_batches = len(loader)

    for batch_idx, (input_sequence_batch, target_sqrt_batch) in enumerate(loader):

        # ---- Move data to GPU ----
        input_sequence_batch = input_sequence_batch.to(device)
        target_sqrt_batch = target_sqrt_batch.to(device)

        # ---- Forward + backward pass ----
        if train_mode:
            optimizer.zero_grad()

        pred_sqrt_batch = model(input_sequence_batch)  # (B, 1, H, W), directly regress sqrt(population)
        loss = criterion(pred_sqrt_batch, target_sqrt_batch)

        if train_mode:
            loss.backward()
            optimizer.step()

        # ---- Aggregate metrics ----
        batch_size = input_sequence_batch.size(0)
        total_loss += loss.item() * batch_size
        total_tem += tem_metric(target_sqrt_batch, pred_sqrt_batch).item() * batch_size
        total_mae += mae(target_sqrt_batch, pred_sqrt_batch).item() * batch_size
        total_r2 += r2_score(target_sqrt_batch, pred_sqrt_batch).item() * batch_size
        total_samples += batch_size

        # ---- Batch-level progress print every 50 batches ----
        if str(phase).lower() != "test" and batch_idx % 50 == 0:
            print(f"[{phase}] Epoch {epoch:03d} | Batch {batch_idx + 1}/{num_batches}")

    tem_value = total_tem / max(1, total_samples)
    return {
        "loss": total_loss / max(1, total_samples),
        "TEM": tem_value,
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
    """
    Regression training routine:
    - loss function: MixedAbsRelLoss
    - metrics: TEM and MAE
    - includes early stopping plus best_loss / best_tem checkpoint saving
    """

    print(f"Starting training for {epochs} epochs...")

    # Use combined absolute/relative error with high-value weighting to better fit core peaks.
    criterion = MixedAbsRelLoss(alpha=0.5, gamma=0.6, eps=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_FINAL_LR, weight_decay=1e-4)
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

    # Ensure output directories exist.
    os.makedirs(os.path.dirname(best_loss_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_tem_path), exist_ok=True)

    for ep in range(1, epochs + 1):

        # ---------- Train ----------
        train_metrics, train_seconds, _ = timed_call(
            device,
            run_epoch,
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=ep,
            phase="train",
        )

        # ---------- Validation ----------
        val_metrics, val_seconds, _ = timed_call(
            device,
            run_epoch,
            loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch=ep,
            phase="val",
        )

        # ---------- Test ----------
        test_metrics, test_seconds, _ = timed_call(
            device,
            run_epoch,
            loader=test_loader,
            model=model,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch=ep,
            phase="test",
        )

        # ---------- Save epoch record ----------
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
            "train_seconds": train_seconds,
            "validation_seconds": val_seconds,
            "test_seconds": test_seconds,
            "epoch_compute_seconds": train_seconds + val_seconds + test_seconds,
        }
        history.append(record)

        # ---------- LR Scheduler ----------
        scheduler.step(record["val_loss"])

        print(
            f"Epoch {ep:03d} | "
            f"TrainLoss {record['train_loss']:.4f} "
            f"ValLoss {record['val_loss']:.4f} "
            f"ValTEM {record['val_tem']:.4f} "
            f"ValMAE {record['val_mae']:.4f} "
            f"ValR2 {record['val_r2']:.4f} "
            f"Time {record['epoch_compute_seconds']:.1f}s"
        )

        # ---------- Fixed-epoch checkpoint ----------
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

    # Save metrics corresponding to the best validation TEM checkpoint.
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


def run_epoch_convlstm(
        loader: DataLoader,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        device: torch.device,
        epoch: int = 0,
        phase: str = "train",
) -> dict:
    """Single epoch loop for ConvLSTM baselines using RMSE as the primary metric."""

    train_mode = optimizer is not None
    model.train(mode=train_mode)

    total_loss = 0.0
    total_rmse = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_samples = 0

    num_batches = len(loader)

    for batch_idx, (input_sequence_batch, target_sqrt_batch) in enumerate(loader):
        input_sequence_batch = input_sequence_batch.to(device)
        target_sqrt_batch = target_sqrt_batch.to(device)

        if train_mode:
            optimizer.zero_grad()

        pred_sqrt_batch = model(input_sequence_batch)
        loss = criterion(pred_sqrt_batch, target_sqrt_batch)

        if train_mode:
            loss.backward()
            optimizer.step()

        batch_size = input_sequence_batch.size(0)
        total_loss += loss.item() * batch_size
        total_rmse += rmse_metric(target_sqrt_batch, pred_sqrt_batch).item() * batch_size
        total_mae += mae(target_sqrt_batch, pred_sqrt_batch).item() * batch_size
        total_r2 += r2_score(target_sqrt_batch, pred_sqrt_batch).item() * batch_size
        total_samples += batch_size

        if str(phase).lower() != "test" and batch_idx % 50 == 0:
            print(f"[{phase}] Epoch {epoch:03d} | Batch {batch_idx + 1}/{num_batches}")

    return {
        "loss": total_loss / max(1, total_samples),
        "rmse": total_rmse / max(1, total_samples),
        "mae": total_mae / max(1, total_samples),
        "r2": total_r2 / max(1, total_samples),
    }


def train_model_convlstm(
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        epochs: int,
        patience: int,
        best_loss_path: str,
        best_rmse_path: str,
        checkpoint_map: Optional[dict[int, Path]] = None,
) -> list[dict]:
    """
    ConvLSTM-oriented training routine:
    - loss function: MixedAbsRelLoss
    - metrics: RMSE / MAE / R2
    - includes early stopping plus best_loss / best_rmse checkpoint saving
    """

    print(f"Starting ConvLSTM training for {epochs} epochs...")

    criterion = MixedAbsRelLoss(alpha=0.5, gamma=0.6, eps=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_FINAL_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    history: list[dict] = []
    best_val_loss = float("inf")
    best_val_rmse = float("inf")
    patience_counter = 0

    os.makedirs(os.path.dirname(best_loss_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_rmse_path), exist_ok=True)

    for ep in range(1, epochs + 1):
        train_metrics = run_epoch_convlstm(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=ep,
            phase="train",
        )
        val_metrics = run_epoch_convlstm(
            loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch=ep,
            phase="val",
        )
        test_metrics = run_epoch_convlstm(
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
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_loss": test_metrics["loss"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
        }
        history.append(record)

        scheduler.step(record["val_loss"])

        print(
            f"Epoch {ep:03d} | "
            f"TrainLoss {record['train_loss']:.4f} "
            f"ValLoss {record['val_loss']:.4f} "
            f"ValRMSE {record['val_rmse']:.4f} "
            f"ValMAE {record['val_mae']:.4f} "
            f"ValR2 {record['val_r2']:.4f}"
        )

        if checkpoint_map and ep in checkpoint_map:
            ckpt_dir = Path(checkpoint_map[ep])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"model_epoch_{ep}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[checkpoint] Saved epoch {ep} model to {ckpt_path}")

        improved = False
        if record["val_loss"] < best_val_loss - 1e-6:
            best_val_loss = record["val_loss"]
            torch.save(model.state_dict(), best_loss_path)
            improved = True

        if record["val_rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = record["val_rmse"]
            torch.save(model.state_dict(), best_rmse_path)
            improved = True

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if history:
        best_record = min(history, key=lambda h: h["val_rmse"])
        best_metrics = {
            "epoch": best_record["epoch"],
            "test_rmse": best_record["test_rmse"],
            "test_mae": best_record["test_mae"],
            "test_r2": best_record["test_r2"],
        }
        metrics_path = Path(best_rmse_path).with_name("best_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, ensure_ascii=False, indent=2)
        print(f"[info] Saved best metrics to {metrics_path}")

    return history


# -----------------------------------------------------------------------------
# Raster prediction and visualization
# -----------------------------------------------------------------------------

class RSLogCoshLoss(nn.Module):
    """Numerically stable log-cosh loss."""

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

def rs_mape_pop(y_true_sqrt: torch.Tensor, y_pred_sqrt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    true_pop = torch.clamp(y_true_sqrt, min=0) ** 2
    pred_pop = torch.clamp(y_pred_sqrt, min=0) ** 2
    ape = torch.abs(pred_pop - true_pop) / (true_pop + eps)
    return torch.mean(ape) * 100.0

def rs_evaluate_loader_metrics(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tem = 0.0
    total_mae = 0.0
    total_r2 = 0.0
    total_mape = 0.0
    total_samples = 0

    with torch.no_grad():
        for input_sequence_batch, target_sqrt_batch in loader:
            input_sequence_batch = input_sequence_batch.to(device)
            target_sqrt_batch = target_sqrt_batch.to(device)
            pred_sqrt_batch = model(input_sequence_batch)
            loss = criterion(pred_sqrt_batch, target_sqrt_batch)
            batch_size = input_sequence_batch.size(0)
            total_loss += float(loss.item()) * batch_size
            total_tem += float(tem_metric(target_sqrt_batch, pred_sqrt_batch).item()) * batch_size
            total_mae += float(mae(target_sqrt_batch, pred_sqrt_batch).item()) * batch_size
            total_r2 += float(r2_score(target_sqrt_batch, pred_sqrt_batch).item()) * batch_size
            total_mape += float(rs_mape_pop(target_sqrt_batch, pred_sqrt_batch).item()) * batch_size
            total_samples += batch_size

    denom = max(1, total_samples)
    tem_value = total_tem / denom
    return {
        "loss": total_loss / denom,
        "TEM": tem_value,
        "mae": total_mae / denom,
        "r2": total_r2 / denom,
        "mape": total_mape / denom,
    }

def rs_train_model_with_trial_hparams(
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        epochs: int,
        patience: int,
        lr: float,
        weight_decay: float,
        criterion: nn.Module,
        best_loss_path: Path,
        best_tem_path: Path,
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
    best_val_tem = float("inf")
    patience_counter = 0
    best_loss_path.parent.mkdir(parents=True, exist_ok=True)
    best_tem_path.parent.mkdir(parents=True, exist_ok=True)

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
        test_metrics = run_epoch(
            loader=test_loader,
            model=model,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch=ep,
            phase="test",
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        record: dict = {
            "epoch": float(ep),
            "lr": current_lr,
            "train_loss": float(train_metrics["loss"]),
            "train_tem": float(train_metrics["TEM"]),
            "train_mae": float(train_metrics["mae"]),
            "train_r2": float(train_metrics["r2"]),
            "val_loss": float(val_metrics["loss"]),
            "val_tem": float(val_metrics["TEM"]),
            "val_mae": float(val_metrics["mae"]),
            "val_r2": float(val_metrics["r2"]),
            "test_loss": float(test_metrics["loss"]),
            "test_tem": float(test_metrics["TEM"]),
            "test_mae": float(test_metrics["mae"]),
            "test_r2": float(test_metrics["r2"]),
        }
        history.append(record)
        scheduler.step(record["val_loss"])

        improved = False
        if record["val_loss"] < best_val_loss - 1e-6:
            best_val_loss = record["val_loss"]
            torch.save(model.state_dict(), str(best_loss_path))
            improved = True
        if record["val_tem"] < best_val_tem - 1e-6:
            best_val_tem = record["val_tem"]
            torch.save(model.state_dict(), str(best_tem_path))
            improved = True
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[epoch {ep:03d}] "
            f"lr={current_lr:.2e} | "
            f"Train(L/TEM/MAE/R2)=({record['train_loss']:.4f}/{record['train_tem']:.4f}/{record['train_mae']:.4f}/{record['train_r2']:.4f}) | "
            f"Val(L/TEM/MAE/R2)=({record['val_loss']:.4f}/{record['val_tem']:.4f}/{record['val_mae']:.4f}/{record['val_r2']:.4f}) | "
            f"BestValTEM={best_val_tem:.4f} | "
            f"ES={patience_counter}/{patience}"
        )

        if patience_counter >= patience:
            print(f"[early-stop] Triggered at epoch {ep:03d}.")
            break

    return history

