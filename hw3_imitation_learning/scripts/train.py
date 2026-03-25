"""Training script for SO-100 action-chunking imitation learning.

Imports a model from hw3.model and trains it on
state -> action-chunk prediction using the processed zarr dataset.

Usage:
    python scripts/train.py --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
        --state-keys ... \
        --action-keys ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import zarr as zarr_lib
from hw3.dataset import (
    Normalizer,
    SO100ChunkDataset,
    load_and_merge_zarrs,
    load_zarr,
)
from hw3.model import BasePolicy, build_policy
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split

EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.1
GRAD_CLIP_NORM = 1.0
DEFAULT_D_MODEL = 256
DEFAULT_DEPTH = 5  # 5


def train_one_epoch(
    model: BasePolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device, non_blocking=True)
        action_chunks = action_chunks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(states, action_chunks)
        loss.backward()
        if grad_clip_norm is not None:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: BasePolicy,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        states, action_chunks = batch
        states = states.to(device, non_blocking=True)
        action_chunks = action_chunks.to(device, non_blocking=True)

        loss = model.compute_loss(states, action_chunks)
        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action-chunking policy.")
    parser.add_argument(
        "--zarr", type=Path, required=True, help="Path to processed .zarr store."
    )
    parser.add_argument(
        "--policy",
        choices=["obstacle", "multitask"],
        default="obstacle",
        help="Policy type: 'obstacle' for single-cube obstacle scene, 'multitask' for multicube (default: obstacle).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Action chunk horizon H (default: 16).",
    )
    parser.add_argument(
        "--state-keys",
        nargs="+",
        default=None,
        help='State array key specs to concatenate, e.g. state_ee_xyz state_gripper "state_cube[:3]". '
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the state_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--action-keys",
        nargs="+",
        default=None,
        help="Action array key specs to concatenate, e.g. action_ee_xyz action_gripper. "
        "Supports column slicing with [:N], [M:], [M:N]. "
        "If omitted, uses the action_key attribute from the zarr metadata.",
    )
    parser.add_argument(
        "--extra-zarr",
        type=Path,
        nargs="*",
        default=None,
        help="Optional additional processed .zarr stores to merge, e.g. DAgger data.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS}).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE}).")
    parser.add_argument("--lr", type=float, default=LR, help=f"Learning rate (default: {LR}).")
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"Number of residual MLP blocks (default: {DEFAULT_DEPTH}).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint file to initialize model weights from.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not 0.0 <= args.val_split < 1.0:
        raise ValueError("--val-split must be in [0, 1)")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────
    zarr_paths = [args.zarr]
    if args.extra_zarr:
        zarr_paths.extend(args.extra_zarr)

    if len(zarr_paths) == 1:
        states, actions, ep_ends = load_zarr(
            args.zarr,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )
    else:
        print(f"Merging {len(zarr_paths)} zarr stores: {[str(p) for p in zarr_paths]}")
        states, actions, ep_ends = load_and_merge_zarrs(
            zarr_paths,
            state_keys=args.state_keys,
            action_keys=args.action_keys,
        )

    ref_zarr = zarr_lib.open_group(str(zarr_paths[0]), mode="r")
    state_keys = args.state_keys or [str(ref_zarr.attrs.get("state_key", "state"))]
    action_keys = args.action_keys or [str(ref_zarr.attrs.get("action_key", "action"))]

    normalizer = Normalizer.from_data(states, actions)

    dataset = SO100ChunkDataset(
        states,
        actions,
        ep_ends,
        chunk_size=args.chunk_size,
        normalizer=normalizer,
    )
    print(f"Dataset: {len(dataset)} samples, chunk_size={args.chunk_size}")
    print(f"  state_dim={states.shape[1]}, action_dim={actions.shape[1]}")
    print(f"  state_keys={state_keys}")
    print(f"  action_keys={action_keys}")

    if len(dataset) < 2:
        raise ValueError("Dataset is too small to split into train/val sets.")

    # ── train / val split ─────────────────────────────────────────────
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  train_samples={n_train}, val_samples={n_val}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    # ── model ─────────────────────────────────────────────────────────
    model = build_policy(
        args.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=args.chunk_size,
        d_model=DEFAULT_D_MODEL,
        depth=args.depth,
    ).to(device)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model weights from checkpoint: {args.checkpoint}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    # ── training loop ─────────────────────────────────────────────────
    best_val = float("inf")

    # Derive action space tag from action keys (e.g. "ee_xyz", "joints")
    action_space = "unknown"
    if args.action_keys:
        for k in args.action_keys:
            base = k.split("[")[0]  # strip column slices
            if base != "action_gripper":
                action_space = base.removeprefix("action_")
                break

    best_save_name = f'best_model_{action_space}_{args.policy}.pt'
    last_save_name = f'last_model_{action_space}_{args.policy}.pt'

    n_dagger_eps = 0
    for zp in zarr_paths:
        z = zarr_lib.open_group(str(zp), mode="r")
        n_dagger_eps += z.attrs.get("num_dagger_episodes", 0)
    if n_dagger_eps > 0:
        best_save_name = f'best_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt'
        last_save_name = f'last_model_{action_space}_{args.policy}_dagger{n_dagger_eps}ep.pt'

    if "multi_cube" in str(args.zarr):
        ckpt_dir = Path("./checkpoints/multi_cube")
    else:
        ckpt_dir = Path("./checkpoints/single_cube")
    best_save_path = ckpt_dir / best_save_name
    last_save_path = ckpt_dir / last_save_name
    best_save_path.parent.mkdir(parents=True, exist_ok=True)

    grad_clip_norm = GRAD_CLIP_NORM if GRAD_CLIP_NORM > 0 else None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss = evaluate(model, val_loader, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer": {
                "state_mean": normalizer.state_mean,
                "state_std": normalizer.state_std,
                "action_mean": normalizer.action_mean,
                "action_std": normalizer.action_std,
            },
            "chunk_size": args.chunk_size,
            "d_model": DEFAULT_D_MODEL,
            "depth": args.depth,
            "policy_type": args.policy,
            "state_keys": state_keys,
            "action_keys": action_keys,
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, last_save_path)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, best_save_path)
            tag = " saved"

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train {train_loss:.6f} | val {val_loss:.6f} | lr {lr:.2e}{tag}"
        )

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Best checkpoint: {best_save_path}")
    print(f"Last checkpoint: {last_save_path}")


if __name__ == "__main__":
    main()
