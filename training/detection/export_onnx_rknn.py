"""Export local YOLO weights to ONNX and prepare RKNN conversion metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExportConfig:
    weights: str
    output_dir: str
    imgsz: int
    opset: int
    rknn_target: str
    quantize_int8: bool
    calibration_dir: str | None
    no_simplify: bool
    dry_run: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/yolov8n_haiyu.pt")
    parser.add_argument("--output-dir", default="training/runs/export")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--rknn-target", default="rk3588")
    parser.add_argument("--quantize-int8", action="store_true")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable onnx-simplifier (only if simplifier is unavailable "
        "or known to break injected-attention graphs).  "
        "Note: unsimplified graphs may contain redundant ops that cause "
        "RKNN conversion failures (review H2).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    config = build_config(parse_args())
    output_dir = Path(config.output_dir)
    plan_path = output_dir / "export_plan.json"
    plan = export_plan(config)
    if config.dry_run:
        _write_json(plan_path, plan)
        print(f"dry-run export plan written to {plan_path}")
        return
    _export_onnx(config, output_dir)
    _write_json(plan_path, plan)


def build_config(args: argparse.Namespace) -> ExportConfig:
    if args.imgsz <= 0:
        raise ValueError("imgsz must be positive")
    if args.opset <= 0:
        raise ValueError("opset must be positive")
    if args.quantize_int8 and not args.calibration_dir:
        raise ValueError("INT8 export requires --calibration-dir")
    return ExportConfig(
        weights=args.weights,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        opset=args.opset,
        rknn_target=args.rknn_target,
        quantize_int8=args.quantize_int8,
        calibration_dir=args.calibration_dir,
        no_simplify=args.no_simplify,
        dry_run=args.dry_run,
    )


def export_plan(config: ExportConfig) -> dict[str, Any]:
    return {
        "task": "detection_export_onnx_rknn",
        "config": asdict(config),
        "onnx": {
            "status": "拟开展/未验证",
            "opset": config.opset,
            "dynamic": False,
        },
        "rknn": {
            "target": config.rknn_target,
            "status": "拟开展/未验证",
            "quantization": "INT8" if config.quantize_int8 else "disabled",
        },
        "metrics": {
            "latency_ms": "拟开展/未验证",
            "model_size_mb": "拟开展/未验证",
        },
        "notes": [
            "ONNX/RKNN files must be generated from local trained weights.",
            "RKNN operator compatibility must be validated on RK3588.",
        ],
    }


def _export_onnx(config: ExportConfig, output_dir: Path) -> None:
    weights_path = Path(config.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"local weights not found: {weights_path}")
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("install ultralytics to export ONNX") from exc

    # Warn if onnx-simplifier may be missing
    simplify = not config.no_simplify
    if simplify:
        try:
            import onnxsim  # noqa: F401
        except ImportError:
            print(
                "Warning: onnx-simplifier not installed. "
                "Install with: pip install onnx-simplifier. "
                "Falling back to simplify=False — the exported graph may "
                "contain redundant ops that cause RKNN conversion failures "
                "(review H2).  Use --no-simplify to suppress this warning."
            )
            simplify = False

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    model.export(
        format="onnx",
        imgsz=config.imgsz,
        opset=config.opset,
        dynamic=False,
        simplify=simplify,
    )
    if config.quantize_int8:
        _write_rknn_placeholder(config, output_dir)


def _write_rknn_placeholder(config: ExportConfig, output_dir: Path) -> None:
    placeholder = {
        "status": "拟开展/未验证",
        "target": config.rknn_target,
        "calibration_dir": config.calibration_dir,
        "note": "Use RKNN-Toolkit2 offline; no conversion is claimed here.",
    }
    _write_json(output_dir / "rknn_conversion_plan.json", placeholder)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
