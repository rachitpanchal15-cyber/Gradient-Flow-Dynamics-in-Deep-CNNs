"""
Experiment 1: Per-layer gradient-flow dynamics in deep CNNs (depth sweep).

Trains ResNet-20, ResNet-56 and ResNet-110 on CIFAR-10 and, via a tensor hook on
each Conv2d weight, records three per-layer statistics each training step:

    grad_norm   = ||grad_l(t)||_2
    SNR_l(t)    = |mean(grad_l(t))| / (std(grad_l(t)) + 1e-8)

These are reduced *online* to per-epoch summaries (one row per layer per epoch):
grad-norm mean + variance, SNR mean + std. Designed for an 8GB unified-memory Mac
(MPS):

  * no raw gradient tensors are ever stored,
  * no per-step history is retained in memory (only small running accumulators),
  * each grad is detached and reduced on-device, and only scalar summaries cross
    to the CPU, where the running aggregation happens in plain Python floats,
  * each epoch's rows are flushed straight to CSV (no large buffers).

Outputs per model: snr_<model>.csv (per-epoch per-layer stats), metrics_<model>.csv
(loss/acc), snr_heatmap_<model>.png. After the sweep, depth_comparison.png plots
the three depths against each other.
"""

from __future__ import annotations

import argparse
import csv
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

# Columns streamed to snr_<model>.csv. One row per Conv2d layer per epoch.
SNR_COLUMNS = ["epoch", "layer", "grad_norm_mean", "grad_norm_var", "snr", "snr_std", "n_steps"]

# Per-depth batch sizes. 128 sits inside the safe 8GB range for all three depths
# (ResNet-20: 128-256, ResNet-56: 64-128, ResNet-110: 32-128) and keeping it equal
# across depths keeps the gradient SNR directly comparable. Lower a single entry
# only if that depth OOMs -- reduce batch size before touching the architecture.
DEFAULT_BATCH_SIZES = {"resnet20": 128, "resnet56": 128, "resnet110": 128}

MODEL_FNS = {"resnet20": resnet20, "resnet56": resnet56, "resnet110": resnet110}


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


class GradStatsRecorder:
    """Per-Conv2d weight-gradient hook that accumulates per-epoch summaries.

    The hook detaches each weight gradient, reduces it on-device to a few scalars
    (L2 norm, and |mean|/std as SNR), and pulls only those scalars to the CPU. The
    running aggregation is kept in plain Python floats -- no gradient tensors and
    no per-step history are retained. ``flush_epoch`` emits one summary row per
    layer and resets the accumulators, so memory stays flat regardless of depth or
    number of steps.
    """

    def __init__(self, model: nn.Module, eps: float = 1e-8):
        self.eps = eps
        self.layer_order: list[str] = []
        self._epoch = 0
        self._handles = []
        # Running per-epoch accumulators (CPU floats), keyed by layer id.
        self._n: dict[str, int] = {}
        self._gn_sum: dict[str, float] = {}     # sum of grad-norm
        self._gn_sumsq: dict[str, float] = {}   # sum of grad-norm^2
        self._snr_sum: dict[str, float] = {}    # sum of SNR
        self._snr_sumsq: dict[str, float] = {}  # sum of SNR^2

        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                layer_id = f"{idx:03d}_{name}"
                self.layer_order.append(layer_id)
                self._reset_layer(layer_id)

                def make_hook(layer_id=layer_id):
                    def hook(grad: torch.Tensor):
                        if grad is None:
                            return
                        # Detach immediately; reduce on-device; only scalars -> CPU.
                        g = grad.detach()
                        gnorm = g.norm().item()
                        mean = g.mean()
                        std = g.std(unbiased=False)
                        snr = (mean.abs() / (std + self.eps)).item()
                        self._n[layer_id] += 1
                        self._gn_sum[layer_id] += gnorm
                        self._gn_sumsq[layer_id] += gnorm * gnorm
                        self._snr_sum[layer_id] += snr
                        self._snr_sumsq[layer_id] += snr * snr

                    return hook

                self._handles.append(module.weight.register_hook(make_hook()))
                idx += 1

    def _reset_layer(self, layer_id: str) -> None:
        self._n[layer_id] = 0
        self._gn_sum[layer_id] = 0.0
        self._gn_sumsq[layer_id] = 0.0
        self._snr_sum[layer_id] = 0.0
        self._snr_sumsq[layer_id] = 0.0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def flush_epoch(self) -> list[tuple]:
        """Reduce this epoch's accumulated step stats to one row per layer.

        Returns the rows (matching SNR_COLUMNS) for incremental streaming to disk
        and resets the accumulators. Nothing is retained in memory afterwards.
        """
        rows: list[tuple] = []
        for layer_id in self.layer_order:
            n = self._n[layer_id]
            if n == 0:
                continue
            gn_mean = self._gn_sum[layer_id] / n
            gn_var = max(self._gn_sumsq[layer_id] / n - gn_mean * gn_mean, 0.0)
            snr_mean = self._snr_sum[layer_id] / n
            snr_var = max(self._snr_sumsq[layer_id] / n - snr_mean * snr_mean, 0.0)
            rows.append(
                (self._epoch, layer_id, gn_mean, gn_var, snr_mean, snr_var ** 0.5, n)
            )
            self._reset_layer(layer_id)
        return rows

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


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


def save_heatmap(csv_path: Path, model_name: str, out_path: Path, value_col: str = "snr") -> None:
    """Render a (layer x epoch) heatmap of `value_col`, read back from the CSV."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    pivot = (
        df.pivot_table(index="layer", columns="epoch", values=value_col, aggfunc="mean")
        .sort_index()
    )
    fig_w = max(8, min(24, 0.18 * pivot.shape[1] + 4))
    fig_h = max(6, min(28, 0.18 * pivot.shape[0] + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        pivot,
        cmap="viridis",
        cbar_kws={"label": f"mean {value_col} (per epoch)"},
        ax=ax,
        rasterized=True,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("conv layer (shallow -> deep)")
    ax.set_title(f"{model_name}: gradient {value_col} vs depth vs training time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_depth_comparison(model_names: list[str], out_path: Path) -> None:
    """Compare gradient SNR across depths, reading each model's streamed CSV.

    Left panel: mean SNR over layers vs epoch (training-time dynamics per depth).
    Right panel: mean SNR over training vs normalized depth (shallow -> deep).
    """
    present = [m for m in model_names if (RESULTS_DIR / f"snr_{m}.csv").exists()]
    if len(present) < 2:
        print(f"comparison: need >=2 models with CSVs, have {present}; skipping.", flush=True)
        return

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    for m in present:
        df = pd.read_csv(RESULTS_DIR / f"snr_{m}.csv")
        per_epoch = df.groupby("epoch")["snr"].mean()
        ax0.plot(per_epoch.index, per_epoch.values, label=m, linewidth=1.5)

        per_layer = df.groupby("layer")["snr"].mean().sort_index()
        depth_frac = np.linspace(0.0, 1.0, len(per_layer))
        ax1.plot(depth_frac, per_layer.values, label=m, marker=".", markersize=4, linewidth=1.0)

    ax0.set_xlabel("epoch")
    ax0.set_ylabel("mean SNR over layers")
    ax0.set_title("Gradient SNR vs training time")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1.set_xlabel("normalized depth (shallow -> deep)")
    ax1.set_ylabel("mean SNR over training")
    ax1.set_title("Gradient SNR vs depth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Depth sweep: " + " vs ".join(present))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"comparison: wrote {out_path}", flush=True)


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
    print(f"\n=== {model_name}: building model and loaders (batch_size={batch_size}) ===", flush=True)
    train_loader, test_loader = build_loaders(batch_size, num_workers, train_subset)

    model = model_fn().to(device)
    recorder = GradStatsRecorder(model)
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

    # Open both logs and write headers up front, then append each epoch's rows as we
    # go, so a crash mid-run still leaves valid, partial CSVs on disk.
    snr_file = open(csv_path, "w", newline="")
    snr_writer = csv.writer(snr_file)
    snr_writer.writerow(SNR_COLUMNS)
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["epoch", "train_loss", "test_acc", "lr", "elapsed_s"])

    t0 = time.time()
    try:
        for epoch in range(epochs):
            model.train()
            recorder.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_n = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
                epoch_n += x.size(0)
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
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Stream this epoch's per-layer stats + metrics straight to disk.
            for row in recorder.flush_epoch():
                snr_writer.writerow(row)
            snr_file.flush()
            metrics_writer.writerow([epoch, train_loss, test_acc, lr_now, elapsed])
            metrics_file.flush()

            print(
                f"{model_name} epoch {epoch+1} done: train_loss={train_loss:.4f} "
                f"test_acc={test_acc:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    finally:
        snr_file.close()
        metrics_file.close()
        recorder.remove()

    # Render the per-model SNR heatmap from the streamed CSV (no in-memory history).
    print(f"{model_name}: rendering heatmap -> {heatmap_path}", flush=True)
    save_heatmap(csv_path, model_name, heatmap_path, value_col="snr")
    print(f"=== {model_name}: done in {time.time()-t0:.1f}s ===", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch size for ALL models. Default: per-depth "
                        f"sizes {DEFAULT_BATCH_SIZES}.")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--milestones", type=int, nargs="+", default=[50, 75])
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers. Default 0: no multiprocessing/spawn IPC, "
                        "which is robust across system sleep and fine for CIFAR-on-MPS.")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--train-subset", type=int, default=None,
                   help="If set, use only the first N training samples. Use a 5k-10k "
                        "subset to validate pipeline stability before the full run.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip a model whose metrics CSV already has --epochs rows "
                        "(avoids recomputing finished runs).")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip training; just (re)render heatmaps + comparison from "
                        "existing CSVs.")
    p.add_argument("--models", type=str, nargs="+",
                   default=["resnet20", "resnet56", "resnet110"])
    p.add_argument("--device", type=str, default=None,
                   help="Override device (cuda/mps/cpu). Default: best available.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def batch_size_for(model_name: str, args: argparse.Namespace) -> int:
    if args.batch_size is not None:
        return args.batch_size
    return DEFAULT_BATCH_SIZES.get(model_name, 128)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    for model_name in args.models:
        if model_name not in MODEL_FNS:
            raise ValueError(f"unknown model: {model_name}")

    if args.plot_only:
        for model_name in args.models:
            save_heatmap(
                RESULTS_DIR / f"snr_{model_name}.csv",
                model_name,
                RESULTS_DIR / f"snr_heatmap_{model_name}.png",
            )
        save_depth_comparison(args.models, RESULTS_DIR / "depth_comparison.png")
        return

    device = torch.device(args.device) if args.device else get_device()
    print(f"device: {device}", flush=True)

    for model_name in args.models:
        if args.skip_existing:
            metrics_path = RESULTS_DIR / f"metrics_{model_name}.csv"
            if metrics_path.exists():
                try:
                    n_done = len(pd.read_csv(metrics_path))
                except Exception:
                    n_done = 0
                if n_done >= args.epochs:
                    print(
                        f"skip {model_name}: {metrics_path.name} already has "
                        f"{n_done} epochs (>= {args.epochs})",
                        flush=True,
                    )
                    continue
        train_one_model(
            model_name=model_name,
            model_fn=MODEL_FNS[model_name],
            epochs=args.epochs,
            batch_size=batch_size_for(model_name, args),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            milestones=args.milestones,
            device=device,
            num_workers=args.num_workers,
            train_subset=args.train_subset,
            log_every=args.log_every,
        )

    # Cross-depth comparison once the sweep finishes.
    save_depth_comparison(args.models, RESULTS_DIR / "depth_comparison.png")


if __name__ == "__main__":
    main()
