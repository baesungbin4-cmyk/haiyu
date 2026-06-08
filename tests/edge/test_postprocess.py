import math

import torch

from edge.common.schemas import Detection
from edge.detect.postprocess import (
    calculate_iou,
    detections_from_predictions,
    non_max_suppression,
)


def test_calculate_iou_for_overlap_and_no_overlap() -> None:
    iou = calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))

    assert math.isclose(iou, 25 / 175)
    assert calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_nms_suppresses_overlapping_same_class_boxes() -> None:
    predictions = torch.tensor(
        [
            [0, 0, 10, 10, 0.90, 0],
            [1, 1, 11, 11, 0.80, 0],
            [20, 20, 30, 30, 0.70, 0],
        ]
    )

    kept = non_max_suppression(predictions, iou_threshold=0.45)

    assert [row.class_id for row in kept] == [0, 0]
    assert [row.bbox for row in kept] == [
        (0.0, 0.0, 10.0, 10.0),
        (20.0, 20.0, 30.0, 30.0),
    ]


def test_nms_suppresses_all_but_highest_conf_in_cluster() -> None:
    predictions = torch.tensor(
        [
            [0, 0, 10, 10, 0.60, 0],
            [1, 1, 11, 11, 0.90, 0],
            [2, 2, 12, 12, 0.75, 0],
        ]
    )

    kept = non_max_suppression(predictions, iou_threshold=0.45)

    assert len(kept) == 1
    assert math.isclose(kept[0].confidence, 0.90, rel_tol=1e-6)
    assert kept[0].bbox == (1.0, 1.0, 11.0, 11.0)


def test_nms_keeps_overlapping_different_class_boxes() -> None:
    predictions = [
        [0, 0, 10, 10, 0.90, 0],
        [1, 1, 11, 11, 0.80, 1],
    ]

    kept = non_max_suppression(predictions, iou_threshold=0.45)

    assert len(kept) == 2


def test_nms_handles_empty_predictions() -> None:
    assert non_max_suppression(torch.empty(0, 6)) == []
    assert non_max_suppression([]) == []


def test_nms_keeps_single_detection() -> None:
    kept = non_max_suppression([[5, 5, 15, 15, 0.80, 2]])

    assert len(kept) == 1
    assert kept[0].class_id == 2
    assert kept[0].bbox == (5.0, 5.0, 15.0, 15.0)


def test_nms_filters_by_confidence_threshold() -> None:
    predictions = [
        [0, 0, 10, 10, 0.49, 0],
        [20, 20, 30, 30, 0.50, 0],
    ]

    kept = non_max_suppression(predictions, confidence_threshold=0.5)

    assert len(kept) == 1
    assert kept[0].bbox == (20.0, 20.0, 30.0, 30.0)


def test_predictions_convert_to_detection_schema_objects() -> None:
    predictions = [[10, 20, 30, 60, 0.88, 0]]

    detections = detections_from_predictions(
        predictions,
        device_id="edge-01",
        timestamp=1_718_000_000.0,
        class_names=["vessel"],
    )

    assert len(detections) == 1
    detection = detections[0]
    assert isinstance(detection, Detection)
    assert detection.class_name == "vessel"
    assert detection.pixel_coordinate == (20.0, 40.0)
