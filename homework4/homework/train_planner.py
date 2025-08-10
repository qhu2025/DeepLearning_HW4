from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model


def select_transform_pipeline(model_name: str) -> str:
    if model_name in {"mlp_planner", "transformer_planner"}:
        return "state_only"
    elif model_name in {"cnn_planner"}:
        return "default"
    else:
        raise ValueError(f"Unknown model {model_name}")


def masked_l1_loss(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # preds, labels: (B, N, 2); mask: (B, N)
    error = (preds - labels).abs() * mask[..., None]
    denom = mask.sum().clamp_min(1).to(error.dtype)
    return error.sum() / denom


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)

        if "image" in batch:
            preds = model(batch["image"])  # (B, N, 2)
        else:
            preds = model(batch["track_left"], batch["track_right"])  # (B, N, 2)

        loss = masked_l1_loss(preds, batch["waypoints"], batch["waypoints_mask"])
        loss.backward()
        optimizer.step()

        batch_size = preds.shape[0]
        running_loss += float(loss.item()) * batch_size
        total += batch_size

    return running_loss / max(total, 1)


@torch.inference_mode()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    metric = PlannerMetric()
    running_loss = 0.0
    total = 0

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if "image" in batch:
            preds = model(batch["image"])  # (B, N, 2)
        else:
            preds = model(batch["track_left"], batch["track_right"])  # (B, N, 2)

        loss = masked_l1_loss(preds, batch["waypoints"], batch["waypoints_mask"])
        metric.add(preds, batch["waypoints"], batch["waypoints_mask"])

        batch_size = preds.shape[0]
        running_loss += float(loss.item()) * batch_size
        total += batch_size

    results = metric.compute()
    results.update({"val_loss": running_loss / max(total, 1)})
    return results


def _resolve_model_name(name: str) -> str:
    aliases = {
        "linear_planner": "mlp_planner",
        "mlp": "mlp_planner",
        "transformer": "transformer_planner",
        "cnn": "cnn_planner",
    }
    return aliases.get(name, name)


def train(
    model_name: str,
    transform_pipeline: str | None = None,
    num_workers: int = 2,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_epoch: int = 10,
    weight_decay: float = 1e-4,
    train_split: str = "drive_data/train",
    val_split: str = "drive_data/val",
    device: str | None = None,
) -> dict:
    """
    Notebook-friendly training wrapper expected by the assignment cell.
    """
    model_name = _resolve_model_name(model_name)

    # Device
    if device is not None:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    # Data
    pipeline = transform_pipeline or select_transform_pipeline(model_name)
    train_loader = load_data(
        train_split,
        transform_pipeline=pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        val_split,
        transform_pipeline=pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    # Model
    model = load_model(model_name, with_weights=False).to(dev)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_stats: dict | None = None

    for epoch in range(1, num_epoch + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, dev)
        val_stats = evaluate(model, val_loader, dev)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_stats['val_loss']:.4f} | "
            f"lon={val_stats['longitudinal_error']:.4f} | lat={val_stats['lateral_error']:.4f}"
        )

        if val_stats["val_loss"] < best_val:
            best_val = val_stats["val_loss"]
            save_model(model)
            best_stats = val_stats

    return best_stats or {}


def main():
    parser = argparse.ArgumentParser("Train planners for Homework 4")
    parser.add_argument("--model", type=str, required=True, choices=["mlp_planner", "transformer_planner", "cnn_planner"])
    parser.add_argument("--train_split", type=str, default="drive_data/train")
    parser.add_argument("--val_split", type=str, default="drive_data/val")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None, help="cpu/cuda/mps or auto if None")

    args = parser.parse_args()

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Data
    transform_pipeline = select_transform_pipeline(args.model)
    train_loader = load_data(
        args.train_split,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        args.val_split,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Model
    model = load_model(args.model, with_weights=False)
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path: Path | None = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_stats['val_loss']:.4f} | "
            f"lon={val_stats['longitudinal_error']:.4f} | lat={val_stats['lateral_error']:.4f}"
        )

        if val_stats["val_loss"] < best_val:
            best_val = val_stats["val_loss"]
            best_path = save_model(model)
            print(f"Saved best model to {best_path}")

    if best_path is None:
        best_path = save_model(model)
        print(f"Saved final model to {best_path}")


if __name__ == "__main__":
    main()
