"""Structured pruning plan for YOLO detector compression."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PruneConfig:
    weights: str
    output_dir: str
    target_sparsity: float
    fine_tune_epochs: int
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/yolov8n_haiyu.pt")
    parser.add_argument("--output-dir", default="training/runs/prune")
    parser.add_argument("--target-sparsity", type=float, default=0.3)
    parser.add_argument("--fine-tune-epochs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = build_config(parse_args())
    plan = pruning_plan(config)
    output_path = Path(config.output_dir) / "prune_plan.json"
    if config.dry_run:
        _write_json(output_path, plan)
        print(f"dry-run pruning plan written to {output_path}")
        return
    _require_local_weights(config.weights)
    _write_json(output_path, plan)
    raise NotImplementedError(
        "channel pruning is an engineering skeleton; run dry-run for CI"
    )


def build_config(args: argparse.Namespace) -> PruneConfig:
    if args.target_sparsity <= 0 or args.target_sparsity >= 1:
        raise ValueError("target_sparsity must be in (0, 1)")
    if args.fine_tune_epochs < 0:
        raise ValueError("fine_tune_epochs must be non-negative")
    return PruneConfig(
        weights=args.weights,
        output_dir=args.output_dir,
        target_sparsity=args.target_sparsity,
        fine_tune_epochs=args.fine_tune_epochs,
        dry_run=args.dry_run,
    )


def pruning_plan(config: PruneConfig) -> dict[str, Any]:
    return {
        "task": "detection_prune_yolo",
        "config": asdict(config),
        "method": "structured channel pruning followed by fine tuning",
        "pruning_signal": "BatchNorm gamma or module-level saliency",
        "metrics": {
            "mAP50_before": "拟开展/未验证",
            "mAP50_after": "拟开展/未验证",
            "latency_ms": "拟开展/未验证",
            "model_size_mb": "拟开展/未验证",
        },
        "notes": [
            "Do not claim compression or latency until locally measured.",
            "Keep residual attention blocks export-friendly after pruning.",
        ],
    }


def _require_local_weights(weights: str) -> None:
    path = Path(weights)
    if not path.exists():
        raise FileNotFoundError(f"local weights not found: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
