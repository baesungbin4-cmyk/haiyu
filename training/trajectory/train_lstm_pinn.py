"""Train or dry-run the LSTM causal-attention trajectory predictor."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrajectoryTrainConfig:
    data: str | None
    output_dir: str
    history_len: int
    future_len: int
    feature_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    epochs: int
    batch_size: int
    learning_rate: float
    lambda_data: float
    lambda_physics: float
    lambda_smooth: float
    dt: float
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default="training/runs/trajectory")
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument("--future-len", type=int, default=5)
    parser.add_argument("--feature-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-data", type=float, default=1.0)
    parser.add_argument("--lambda-physics", type=float, default=0.1)
    parser.add_argument("--lambda-smooth", type=float, default=0.1)
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Frame interval in seconds.  Must match the actual capture "
        "frame rate (e.g. 0.067 for 15 fps, 0.033 for 30 fps).  "
        "A mismatch will cause physics and smoothness loss magnitudes "
        "to be off by (actual_dt / dt)^k, biasing optimisation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = build_config(parse_args())
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.dry_run or config.data is None:
        metrics = dry_run_metrics(config)
        _write_json(output_dir / "metrics.json", metrics)
        print(f"dry-run trajectory metrics written to {output_dir}")
        return
    metrics = train(config)
    _write_json(output_dir / "metrics.json", metrics)


def build_config(args: argparse.Namespace) -> TrajectoryTrainConfig:
    positive_ints = (
        "history_len",
        "future_len",
        "feature_dim",
        "hidden_dim",
        "num_layers",
        "num_heads",
        "epochs",
        "batch_size",
    )
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.dt <= 0:
        raise ValueError("dt must be positive")
    return TrajectoryTrainConfig(
        data=args.data,
        output_dir=args.output_dir,
        history_len=args.history_len,
        future_len=args.future_len,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lambda_data=args.lambda_data,
        lambda_physics=args.lambda_physics,
        lambda_smooth=args.lambda_smooth,
        dt=args.dt,
        dry_run=args.dry_run,
    )


def dry_run_metrics(config: TrajectoryTrainConfig) -> dict[str, Any]:
    _build_model(config)
    return {
        "task": "trajectory_train_lstm_pinn",
        "mode": "dry-run",
        "config": asdict(config),
        "metrics": {
            "train_loss": "拟开展/未验证",
            "validation_ADE": "拟开展/未验证",
            "validation_FDE": "拟开展/未验证",
        },
        "notes": [
            "No fake metrics are generated in dry-run mode.",
            "Split by trajectory before windowing to avoid leakage.",
            "PINN loss is used only during training backpropagation.",
        ],
    }


def train(config: TrajectoryTrainConfig) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from edge.predict.pinn_loss import PINNLossWeights, PINNTrajectoryLoss
    from edge.predict.trajectory_dataset import TrajectoryWindowDataset

    model = _build_model(config)
    trajectories = _load_json_trajectories(Path(config.data or ""))

    # Split by trajectory before windowing to avoid leakage.
    # random_split on windows would place overlapping windows from the same
    # trajectory in both train and val when stride < history_len+future_len.
    train_dataset, val_dataset = TrajectoryWindowDataset.split_by_trajectory(
        trajectories,
        history_len=config.history_len,
        future_len=config.future_len,
        train_ratio=0.8,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    weights = PINNLossWeights(
        lambda_data=config.lambda_data,
        lambda_physics=config.lambda_physics,
        lambda_smooth=config.lambda_smooth,
    )
    criterion = PINNTrajectoryLoss(weights=weights, dt=config.dt)
    model.train()
    last_loss: float | None = None
    for _epoch in range(config.epochs):
        for history, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history)
            losses = criterion(prediction, target)
            losses.total.backward()
            optimizer.step()
            last_loss = float(losses.total.detach().cpu())

    metrics: dict[str, Any] = {
        "task": "trajectory_train_lstm_pinn",
        "mode": "local_training",
        "config": asdict(config),
        "metrics": {
            "train_loss": last_loss,
            "validation_ADE": "拟开展/未验证",
            "validation_FDE": "拟开展/未验证",
        },
        "metric_status": "local run only; not real-port validation",
        "split_info": {
            "method": "split_by_trajectory",
            "train_trajectories": len(train_dataset._trajectories),
            "val_trajectories": len(val_dataset._trajectories),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "note": "Trajectories are split before windowing to prevent "
            "overlapping windows from the same trajectory appearing in "
            "both train and validation sets.",
        },
    }
    metrics["metrics"].update(_evaluate_loader(model, val_dataset))
    return metrics


def _build_model(config: TrajectoryTrainConfig) -> Any:
    from edge.predict.lstm_causal_attention import (
        LSTMCausalAttentionPredictor,
    )

    return LSTMCausalAttentionPredictor(
        feature_dim=config.feature_dim,
        future_len=config.future_len,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )


def _evaluate_loader(model: Any, dataset: Any) -> dict[str, float]:
    import torch
    from torch.utils.data import DataLoader

    from training.trajectory.evaluate_prediction import ade_fde

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for history, target in loader:
            predictions.append(model(history))
            targets.append(target)
    pred = torch.cat(predictions, dim=0).detach().cpu().tolist()
    target = torch.cat(targets, dim=0).detach().cpu().tolist()
    ade, fde = ade_fde(pred, target)
    return {"validation_ADE": ade, "validation_FDE": fde}


def _load_json_trajectories(path: Path) -> list[list[list[float]]]:
    if not path.exists():
        raise FileNotFoundError(f"trajectory data not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("trajectories")
    if not isinstance(payload, list):
        raise ValueError("trajectory data must be a list or trajectories dict")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
