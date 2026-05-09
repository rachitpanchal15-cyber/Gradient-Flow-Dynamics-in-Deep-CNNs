"""
Experiment 1: Per-layer gradient SNR dynamics in deep CNNs.

Trains ResNet-20, ResNet-56, ResNet-110 on CIFAR-10 for 100 epochs each,
attaches per-Conv2d weight-gradient hooks, and logs

    SNR_l(t) = |mean(grad_l(t))| / (std(grad_l(t)) + 1e-8)

per layer per training step. After each model finishes, a per-epoch
heatmap (depth x epoch) is written to results/ and the raw per-step
SNR log is written as a CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from resnet import resnet20, resnet56, resnet110  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(batch_size: int, num_workers: int, train_subset: int | None = None):
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])

    train_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=test_tf
    )

    if train_subset is not None:
        train_set = Subset(train_set, list(range(min(train_subset, len(train_set)))))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


class SNRRecorder:
    """Registers a tensor hook on each Conv2d weight to record per-step SNR."""

    def __init__(self, model: nn.Module):
        self.records: list[tuple[int, int, str, float]] = []  # epoch, step, layer, snr
        self.layer_order: list[str] = []
        self._step = 0
        self._epoch = 0
        self._handles = []

        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_id = f"{idx:03d}_{name}"
                self.layer_order.append(layer_id)

                def make_hook(layer_id=layer_id):
                    def hook(grad: torch.Tensor):
                        if grad is None:
                            return
                        g = grad.detach()
                        # std with unbiased=False so a single-element tensor still works.
                        mean = g.mean()
                        std = g.std(unbiased=False)
                        snr = (mean.abs() / (std + 1e-8)).item()
                        self.records.append((self._epoch, self._step, layer_id, snr))

                    return hook

                self._handles.append(module.weight.register_hook(make_hook()))
                idx += 1

    def set_step(self, epoch: int, step: int) -> None:
        self._epoch = epoch
        self._step = step

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.records, columns=["epoch", "step", "layer", "snr"]
        )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def save_heatmap(df: pd.DataFrame, model_name: str, out_path: Path) -> None:
    """Aggregate per-step SNR into (layer x epoch) means and render a heatmap."""
    if df.empty:
        return
    pivot = (
        df.groupby(["layer", "epoch"], sort=True)["snr"]
        .mean()
        .unstack("epoch")
        .sort_index()
    )
    fig_w = max(8, min(24, 0.18 * pivot.shape[1] + 4))
    fig_h = max(6, min(28, 0.18 * pivot.shape[0] + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="viridis",
        cbar_kws={"label": "mean SNR (per epoch)"},
        ax=ax,
        rasterized=True,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("conv layer (shallow -> deep)")
    ax.set_title(f"{model_name}: gradient SNR vs depth vs training time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_one_model(
    model_name: str,
    model_fn,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    milestones: list[int],
    device: torch.device,
    num_workers: int,
    train_subset: int | None,
    log_every: int,
) -> None:
    print(f"\n=== {model_name}: building model and loaders ===", flush=True)
    train_loader, test_loader = build_loaders(batch_size, num_workers, train_subset)

    model = model_fn().to(device)
    recorder = SNRRecorder(model)
    print(
        f"{model_name}: {sum(p.numel() for p in model.parameters())/1e6:.3f}M params, "
        f"{len(recorder.layer_order)} Conv2d layers",
        flush=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    csv_path = RESULTS_DIR / f"snr_{model_name}.csv"
    heatmap_path = RESULTS_DIR / f"snr_heatmap_{model_name}.png"
    metrics_path = RESULTS_DIR / f"metrics_{model_name}.csv"
    metrics_rows: list[dict] = []

    global_step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            recorder.set_step(epoch, global_step)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            epoch_n += x.size(0)
            global_step += 1
            if log_every and batch_idx % log_every == 0:
                print(
                    f"{model_name} epoch {epoch+1}/{epochs} step {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.4f} "
                    f"elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
        scheduler.step()
        train_loss = epoch_loss / max(epoch_n, 1)
        test_acc = evaluate(model, test_loader, device)
        metrics_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_acc": test_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.time() - t0,
            }
        )
        print(
            f"{model_name} epoch {epoch+1} done: train_loss={train_loss:.4f} "
            f"test_acc={test_acc:.4f} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    # Persist results.
    print(f"{model_name}: writing SNR CSV with {len(recorder.records)} rows -> {csv_path}", flush=True)
    df = recorder.to_dataframe()
    df.to_csv(csv_path, index=False)

    print(f"{model_name}: rendering heatmap -> {heatmap_path}", flush=True)
    save_heatmap(df, model_name, heatmap_path)

    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    recorder.remove()
    print(f"=== {model_name}: done in {time.time()-t0:.1f}s ===", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[50, 75])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--train-subset", type=int, default=None,
                   help="If set, use only the first N training samples (smoke testing).")
    p.add_argument("--models", type=str, nargs="+",
                   default=["resnet20", "resnet56", "resnet110"])
    p.add_argument("--device", type=str, default=None,
                   help="Override device (cuda/mps/cpu). Default: best available.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {device}", flush=True)

    model_fns = {
        "resnet20": resnet20,
        "resnet56": resnet56,
        "resnet110": resnet110,
    }

    for model_name in args.models:
        if model_name not in model_fns:
            raise ValueError(f"unknown model: {model_name}")
        train_one_model(
            model_name=model_name,
            model_fn=model_fns[model_name],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            milestones=args.milestones,
            device=device,
            num_workers=args.num_workers,
            train_subset=args.train_subset,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
