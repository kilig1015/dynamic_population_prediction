from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .data import HourlyPopulationDataset, HourlyPopulationPatchDataset

def plot_training_curves(history: list[dict], output_dir: Optional[Path] = None) -> None:
    """
    Plot loss/TEM/MAE:
    1) train + val + test
    2) train + test
    3) train + val
    Generate nine plots in total.
    """
    if not history:
        return

    save_dir = Path(output_dir) if output_dir is not None else Path("training_plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    test_loss = [h["test_loss"] for h in history]
    train_tem = [h["train_tem"] for h in history]
    val_tem = [h["val_tem"] for h in history]
    test_tem = [h["test_tem"] for h in history]
    train_mae = [h["train_mae"] for h in history]
    val_mae = [h["val_mae"] for h in history]
    test_mae = [h["test_mae"] for h in history]

    def _plot(ys: list[tuple[list[float], str]], title: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(8, 4))
        for series, label in ys:
            plt.plot(epochs, series, label=label)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_path = save_dir / filename
        if out_path.suffix.lower() in {".tif", ".tiff"}:
            plt.savefig(out_path, dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[info] Saved training curve: {out_path}")

    # 1) train + val + test
    _plot([(train_loss, "Train Loss"), (val_loss, "Val Loss"), (test_loss, "Test Loss")],
          "Loss (Train/Val/Test)", "Loss", "loss_train_val_test.tif")
    _plot([(train_tem, "Train TEM"), (val_tem, "Val TEM"), (test_tem, "Test TEM")],
          "TEM (Train/Val/Test)", "TEM", "tem_train_val_test.tif")
    _plot([(train_mae, "Train MAE"), (val_mae, "Val MAE"), (test_mae, "Test MAE")],
          "MAE (Train/Val/Test)", "MAE", "mae_train_val_test.tif")

    # 2) train + test
    _plot([(train_loss, "Train Loss"), (test_loss, "Test Loss")],
          "Loss (Train/Test)", "Loss", "loss_train_test.tif")
    _plot([(train_tem, "Train TEM"), (test_tem, "Test TEM")],
          "TEM (Train/Test)", "TEM", "tem_train_test.tif")
    _plot([(train_mae, "Train MAE"), (test_mae, "Test MAE")],
          "MAE (Train/Test)", "MAE", "mae_train_test.tif")

    # 3) train + val
    _plot([(train_loss, "Train Loss"), (val_loss, "Val Loss")],
          "Loss (Train/Val)", "Loss", "loss_train_val.tif")
    _plot([(train_tem, "Train TEM"), (val_tem, "Val TEM")],
          "TEM (Train/Val)", "TEM", "tem_train_val.tif")
    _plot([(train_mae, "Train MAE"), (val_mae, "Val MAE")],
          "MAE (Train/Val)", "MAE", "mae_train_val.tif")

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

