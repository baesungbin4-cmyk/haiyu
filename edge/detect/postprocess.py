"""YOLO-style detector postprocessing utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from edge.common.schemas import BBox, Detection


@dataclass(frozen=True)
class RawDetection:
    """Detection row after confidence filtering and NMS."""

    bbox: BBox
    confidence: float
    class_id: int


def calculate_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate IoU for two ``(x1, y1, x2, y2)`` boxes."""

    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("boxes must contain four coordinates")

    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def non_max_suppression(
    predictions: torch.Tensor | Sequence[Sequence[float]],
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
) -> list[RawDetection]:
    """Filter YOLO rows ``[x1, y1, x2, y2, conf, class_id]`` with NMS."""

    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if iou_threshold < 0 or iou_threshold > 1:
        raise ValueError("iou_threshold must be between 0 and 1")

    rows = _to_prediction_rows(predictions)
    rows = [row for row in rows if row.confidence >= confidence_threshold]
    rows.sort(key=lambda row: row.confidence, reverse=True)

    kept: list[RawDetection] = []
    for row in rows:
        should_keep = True
        for existing in kept:
            same_class = row.class_id == existing.class_id
            overlaps = calculate_iou(row.bbox, existing.bbox) > iou_threshold
            if same_class and overlaps:
                should_keep = False
                break
        if should_keep:
            kept.append(row)
    return kept


def detections_from_predictions(
    predictions: torch.Tensor | Sequence[Sequence[float]],
    *,
    device_id: str,
    timestamp: float,
    class_names: Mapping[int, str] | Sequence[str] | None = None,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Convert raw YOLO rows to validated ``Detection`` schema objects."""

    rows = non_max_suppression(
        predictions,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    return [
        Detection(
            device_id=device_id,
            timestamp=timestamp,
            bbox=row.bbox,
            confidence=row.confidence,
            class_name=_class_name(row.class_id, class_names),
            pixel_coordinate=_bbox_center(row.bbox),
        )
        for row in rows
    ]


def _to_prediction_rows(
    predictions: torch.Tensor | Sequence[Sequence[float]],
) -> list[RawDetection]:
    tensor = torch.as_tensor(predictions, dtype=torch.float32).detach().cpu()
    if tensor.numel() == 0:
        return []
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[1] != 6:
        raise ValueError("predictions must have shape [N, 6] or [1, N, 6]")

    rows: list[RawDetection] = []
    for values in tensor.tolist():
        x1, y1, x2, y2, confidence, class_id = values
        rows.append(
            RawDetection(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                class_id=int(class_id),
            )
        )
    return rows


def _class_name(
    class_id: int,
    class_names: Mapping[int, str] | Sequence[str] | None,
) -> str:
    if class_names is None:
        return f"class_{class_id}"
    if isinstance(class_names, Mapping):
        return class_names.get(class_id, f"class_{class_id}")
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


__all__ = [
    "RawDetection",
    "calculate_iou",
    "detections_from_predictions",
    "non_max_suppression",
]
