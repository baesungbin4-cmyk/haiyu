"""Configurable YOLO training entry point for the Haiyu detector."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class YoloTrainingConfig:
    data: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    project: str
    name: str
    freeze: int
    use_residual_attention: bool
    attention_reduction: int
    attention_min_hidden: int
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="training/detection/data.yaml")
    parser.add_argument("--model", default="weights/yolov8n-local.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="training/runs/detection")
    parser.add_argument("--name", default="yolov8n_haiyu_baseline")
    parser.add_argument("--freeze", type=int, default=10)
    parser.add_argument("--use-residual-attention", action="store_true")
    parser.add_argument(
        "--attention-reduction",
        type=int,
        default=16,
        help="Channel reduction ratio for SEResidualAttention (contract §3)",
    )
    parser.add_argument(
        "--attention-min-hidden",
        type=int,
        default=4,
        help="Minimum hidden channels for SEResidualAttention",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-out", default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> YoloTrainingConfig:
    _validate_positive(args.epochs, "epochs")
    _validate_positive(args.imgsz, "imgsz")
    _validate_positive(args.batch, "batch")
    _validate_positive(args.attention_reduction, "attention_reduction")
    _validate_positive(args.attention_min_hidden, "attention_min_hidden")
    if args.freeze < 0:
        raise ValueError("freeze must be non-negative")
    return YoloTrainingConfig(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        freeze=args.freeze,
        use_residual_attention=args.use_residual_attention,
        attention_reduction=args.attention_reduction,
        attention_min_hidden=args.attention_min_hidden,
        dry_run=args.dry_run,
    )


def main() -> None:
    config = build_config(parse_args())
    plan = training_plan(config)
    plan_out = _plan_path(config)
    if config.dry_run:
        if config.use_residual_attention:
            _verify_attention_module_available()
            plan["attention_injection"] = {
                "status": "dry_run_verified",
                "note": "SEResidualAttention module is importable; "
                "injection will be applied during actual training.",
            }
        _write_json(plan_out, plan)
        print(f"dry-run training plan written to {plan_out}")
        return
    _run_ultralytics_training(config)


def training_plan(config: YoloTrainingConfig) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "task": "detection_train_yolo",
        "config": asdict(config),
        "dataset_status": "public/synthetic placeholder unless replaced",
        "metrics": {
            "mAP50": "拟开展/未验证",
            "latency_ms": "拟开展/未验证",
            "model_size_mb": "拟开展/未验证",
        },
        "notes": [
            "No datasets or weights are downloaded automatically.",
            "Ultralytics YOLOv8 is AGPL-3.0; review obligations before use.",
            "Use local measured values only; do not copy contract metrics.",
        ],
    }
    if config.use_residual_attention:
        plan["attention_injection"] = {
            "enabled": True,
            "module": "SEResidualAttention",
            "target": "PAN-FPN neck multi-scale feature maps (P3/P4/P5)",
            "source": "edge/detect/attention_injector.py (shared with detector)",
            "reduction_ratio": config.attention_reduction,
            "min_hidden_channels": config.attention_min_hidden,
        }
    return plan


def _verify_attention_module_available() -> None:
    """Ensure the shared attention injector can be imported."""
    try:
        from edge.detect.attention_injector import (  # noqa: F401
            inject_attention_to_yolo,
        )
    except ImportError as exc:
        raise ImportError(
            "attention_injector module is required for "
            "--use-residual-attention.  Ensure edge/detect/ is on "
            "PYTHONPATH."
        ) from exc


def _inject_attention_to_yolo(
    model: object, config: YoloTrainingConfig
) -> dict[str, Any]:
    """Inject SEResidualAttention into the YOLO neck via the shared injector.

    Delegates to ``edge.detect.attention_injector.inject_attention_to_yolo``,
    the single source of truth for attention injection used by both the
    training pipeline and the edge detector.
    """
    from edge.detect.attention_injector import (
        inject_attention_to_yolo as _shared_inject,
    )

    return _shared_inject(
        model,
        reduction=config.attention_reduction,
        min_hidden_channels=config.attention_min_hidden,
    )


def _run_ultralytics_training(config: YoloTrainingConfig) -> None:
    data_path = Path(config.data)
    model_path = Path(config.model)
    if not data_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            "model weights must be local; automatic download is disabled"
        )
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        message = "install ultralytics to run non-dry training"
        raise RuntimeError(message) from exc

    model = YOLO(str(model_path))

    if config.use_residual_attention:
        injection_info = _inject_attention_to_yolo(model, config)
        plan = training_plan(config)
        plan["attention_injection"]["injected_layers"] = injection_info[
            "injected_layers"
        ]
        plan["attention_injection"]["status"] = "injected"
        plan["attention_injection"]["reduction"] = config.attention_reduction
        plan["attention_injection"]["min_hidden"] = config.attention_min_hidden
        print(
            "SEResidualAttention injected into YOLO neck: "
            f"{injection_info['injected_layers']}"
        )
        _write_json(_plan_path(config), plan)

    model.train(
        data=str(data_path),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        project=config.project,
        name=config.name,
        freeze=config.freeze,
    )

    # ------------------------------------------------------------------
    # H1/M1: After training, strip attention from the best checkpoint so
    # it can be loaded with the standard YOLO("best.pt") call.
    # The detector re-injects attention at load time.
    # ------------------------------------------------------------------
    if config.use_residual_attention:
        _strip_and_resave_checkpoint(config)


def _strip_and_resave_checkpoint(config: YoloTrainingConfig) -> None:
    """Strip attention weights from the best checkpoint's state_dict.

    H1/M1: ultralytics saves checkpoints whose state_dict contains
    attention parameter keys.  ``YOLO(path)`` reconstructs a standard
    architecture from its YAML config and will fail when the keys don't
    match.  We load the raw checkpoint, filter out attention keys, and
    re-save so standard loading works.
    """
    import torch

    best_path = Path(config.project) / config.name / "weights" / "best.pt"
    if not best_path.exists():
        print(
            "Warning: best.pt not found at expected path "
            f"{best_path}; skipping attention strip."
        )
        return

    # ultralytics checkpoints are torch.save()'d dicts with a 'model' key
    # containing the state_dict (YOLO saves the full training state).
    try:
        checkpoint = torch.load(str(best_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Warning: could not load checkpoint {best_path}: {exc}")
        return

    # Locate the model state_dict within the checkpoint
    state_dict = None
    if isinstance(checkpoint, dict):
        # Common ultralytics layout: ckpt['model'].state_dict() was saved
        if "model" in checkpoint:
            model_data = checkpoint["model"]
            if hasattr(model_data, "state_dict"):
                state_dict = model_data.state_dict()
            elif isinstance(model_data, dict):
                state_dict = model_data

    if state_dict is None:
        print(
            "Warning: could not extract state_dict from checkpoint; "
            "skipping attention strip."
        )
        return

    # Filter out attention.* keys
    clean_sd = {k: v for k, v in state_dict.items() if "attention" not in k.lower()}
    removed = len(state_dict) - len(clean_sd)
    if removed == 0:
        print("No attention keys found in checkpoint — already clean.")
        return

    # Replace the state_dict in-place
    if "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
        checkpoint["model"].load_state_dict(clean_sd, strict=False)
    elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint["model"] = clean_sd

    clean_path = Path(str(best_path).replace(".pt", "_clean.pt"))
    torch.save(checkpoint, str(clean_path))
    print(
        f"Clean checkpoint saved to {clean_path} "
        f"({removed} attention keys removed).  "
        f"Load with YOLO('{clean_path}') or "
        f"YOLOAttentionDetector('{clean_path}', use_residual_attention=True)."
    )


def _plan_path(config: YoloTrainingConfig) -> Path:
    output_dir = Path(config.project) / config.name
    return output_dir / "training_plan.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


if __name__ == "__main__":
    main()
