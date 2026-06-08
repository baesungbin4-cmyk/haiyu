"""Evaluate trajectory prediction files with ADE/FDE metrics.

Supports global and per-class ADE/FDE statistics (review item S6).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-file", default=None)
    parser.add_argument("--target-file", default=None)
    parser.add_argument(
        "--metrics-out",
        default="training/runs/trajectory/eval.json",
    )
    parser.add_argument(
        "--class-labels",
        default=None,
        help="Path to a JSON array of per-sequence class IDs or names "
        "(same length as the batch dimension).  Enables per-class ADE/FDE.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.metrics_out)
    if args.dry_run:
        payload = dry_run_metrics()
    else:
        if args.prediction_file is None or args.target_file is None:
            raise ValueError("prediction and target files are required")
        prediction = _load_positions(Path(args.prediction_file))
        target = _load_positions(Path(args.target_file))
        class_labels = _load_class_labels(args.class_labels)
        ade, fde = ade_fde(prediction, target)
        payload: dict[str, Any] = {
            "task": "trajectory_evaluate_prediction",
            "metrics": {"ADE": ade, "FDE": fde},
            "metric_status": "local evaluation only; not real-port validation",
        }
        if class_labels is not None:
            per_class = per_class_ade_fde(prediction, target, class_labels)
            payload["per_class_metrics"] = per_class
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"trajectory evaluation metrics written to {output_path}")


def dry_run_metrics() -> dict[str, Any]:
    return {
        "task": "trajectory_evaluate_prediction",
        "mode": "dry-run",
        "metrics": {"ADE": "拟开展/未验证", "FDE": "拟开展/未验证"},
        "notes": ["No predictions were evaluated; no fake numbers emitted."],
    }


def ade_fde(
    prediction: list[Any],
    target: list[Any],
) -> tuple[float, float]:
    pred_batches = _as_batches(prediction)
    target_batches = _as_batches(target)
    if len(pred_batches) != len(target_batches):
        raise ValueError("prediction and target batch sizes must match")

    total_distance = 0.0
    total_points = 0
    final_distance = 0.0
    for pred_seq, target_seq in zip(pred_batches, target_batches):
        if len(pred_seq) != len(target_seq):
            message = "prediction and target sequence lengths must match"
            raise ValueError(message)
        if not pred_seq:
            raise ValueError("sequences must be non-empty")
        for pred_point, target_point in zip(pred_seq, target_seq):
            total_distance += _distance(pred_point, target_point)
            total_points += 1
        final_distance += _distance(pred_seq[-1], target_seq[-1])
    if total_points == 0:
        raise ValueError("no trajectory points to evaluate")
    ade = total_distance / total_points
    fde = final_distance / len(pred_batches)
    return ade, fde


def _load_positions(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"positions file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("prediction", "predictions", "target", "targets"):
            if key in payload:
                return payload[key]
    if isinstance(payload, list):
        return payload
    raise ValueError("positions file must contain a list or known dict key")


def _as_batches(values: list[Any]) -> list[list[list[float]]]:
    if not values:
        raise ValueError("positions must be non-empty")
    first = values[0]
    if _is_point(first):
        return [[_point(first_item) for first_item in values]]
    return [[_point(point) for point in sequence] for sequence in values]


def _is_point(value: Any) -> bool:
    return isinstance(value, list | tuple) and len(value) == 2


def _point(value: Any) -> list[float]:
    if not _is_point(value):
        raise ValueError("each point must be [x, y]")
    return [float(value[0]), float(value[1])]


def per_class_ade_fde(
    prediction: list[Any],
    target: list[Any],
    class_labels: list[str | int],
) -> dict[str, dict[str, float]]:
    """Compute ADE and FDE grouped by class label.

    Parameters
    ----------
    prediction, target:
        Same format as ``ade_fde`` — list of batches, each batch a list
        of sequences of ``[x, y]`` points.
    class_labels:
        One label per batch entry; length must equal the total number of
        sequences across all prediction batches.

    Returns
    -------
    dict
        ``{class_name: {"ADE": float, "FDE": float, "count": int}, ...}``
        plus a ``"global"`` entry with the full-dataset ADE/FDE for
        cross-reference.
    """
    pred_batches = _as_batches(prediction)
    target_batches = _as_batches(target)
    if len(pred_batches) != len(target_batches):
        raise ValueError("prediction and target batch sizes must match")

    # Flatten batches into per-sequence lists
    all_pred_seqs: list[list[list[float]]] = []
    all_target_seqs: list[list[list[float]]] = []
    for pred_seq_list, target_seq_list in zip(pred_batches, target_batches):
        if len(pred_seq_list) != len(target_seq_list):
            raise ValueError("prediction and target sequences per batch must match")
        all_pred_seqs.extend(pred_seq_list)
        all_target_seqs.extend(target_seq_list)

    total_seqs = len(all_pred_seqs)
    if len(class_labels) != total_seqs:
        raise ValueError(
            f"class_labels length ({len(class_labels)}) must match "
            f"total sequences ({total_seqs})"
        )

    # Group by class
    groups: dict[str | int, dict[str, float | int]] = defaultdict(
        lambda: {
            "total_distance": 0.0,
            "total_points": 0,
            "final_distance": 0.0,
            "count": 0,
        }
    )
    for label, pred_seq, target_seq in zip(
        class_labels, all_pred_seqs, all_target_seqs
    ):
        if len(pred_seq) != len(target_seq):
            raise ValueError("sequence lengths must match")
        group = groups[label]
        for pred_point, target_point in zip(pred_seq, target_seq):
            group["total_distance"] += _distance(pred_point, target_point)
            group["total_points"] += 1  # type: ignore[operator]
        group["final_distance"] += _distance(pred_seq[-1], target_seq[-1])
        group["count"] += 1  # type: ignore[operator]

    result: dict[str, dict[str, float]] = {}
    for label, group in sorted(groups.items(), key=lambda x: str(x[0])):
        n = int(group["count"])
        result[str(label)] = {
            "ADE": round(group["total_distance"] / int(group["total_points"]), 6),
            "FDE": round(group["final_distance"] / n, 6),
            "count": n,
        }

    # Add global reference
    global_ade, global_fde = ade_fde(prediction, target)
    result["global"] = {"ADE": global_ade, "FDE": global_fde, "count": total_seqs}
    return result


def _load_class_labels(path: str | None) -> list[str | int] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"class labels file not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("class labels file must contain a JSON array")
    return payload


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


if __name__ == "__main__":
    main()
