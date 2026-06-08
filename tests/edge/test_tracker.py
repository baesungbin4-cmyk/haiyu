from edge.common.schemas import Detection
from edge.track.tracker import ConstantVelocityFallback, IoUTracker


def _detection(
    timestamp: float,
    bbox: tuple[float, float, float, float],
    pixel_coordinate: tuple[float, float],
    world_coordinate: tuple[float, float] | None = None,
    class_name: str = "vessel",
) -> Detection:
    return Detection(
        device_id="edge-01",
        timestamp=timestamp,
        bbox=bbox,
        confidence=0.9,
        class_name=class_name,
        pixel_coordinate=pixel_coordinate,
        world_coordinate=world_coordinate,
    )


def test_same_object_across_adjacent_frames_keeps_track_id() -> None:
    tracker = IoUTracker(iou_threshold=0.2)

    first = tracker.update(
        [
            _detection(
                timestamp=10.0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                pixel_coordinate=(5.0, 5.0),
                world_coordinate=(100.0, 200.0),
            )
        ]
    )
    second = tracker.update(
        [
            _detection(
                timestamp=11.0,
                bbox=(1.0, 0.0, 11.0, 10.0),
                pixel_coordinate=(6.0, 5.0),
                world_coordinate=(102.0, 201.0),
            )
        ]
    )

    assert second[0].track_id == first[0].track_id
    assert second[0].velocity == (2.0, 1.0)


def test_missing_frame_does_not_crash_or_drop_recent_track() -> None:
    tracker = IoUTracker(iou_threshold=0.2, max_missed=1)

    first = tracker.update(
        [
            _detection(
                timestamp=10.0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                pixel_coordinate=(5.0, 5.0),
            )
        ]
    )
    missing = tracker.update([], timestamp=11.0)
    second = tracker.update(
        [
            _detection(
                timestamp=12.0,
                bbox=(2.0, 0.0, 12.0, 10.0),
                pixel_coordinate=(7.0, 5.0),
            )
        ]
    )

    assert missing == []
    assert second[0].track_id == first[0].track_id
    assert second[0].velocity == (1.0, 0.0)


def test_unmatched_detection_gets_new_track_id() -> None:
    tracker = IoUTracker(iou_threshold=0.5)

    first = tracker.update(
        [
            _detection(
                timestamp=1.0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                pixel_coordinate=(5.0, 5.0),
            )
        ]
    )
    second = tracker.update(
        [
            _detection(
                timestamp=2.0,
                bbox=(100.0, 100.0, 110.0, 110.0),
                pixel_coordinate=(105.0, 105.0),
            )
        ]
    )

    assert second[0].track_id != first[0].track_id


def test_constant_velocity_fallback_can_emit_missing_prediction() -> None:
    tracker = IoUTracker(
        iou_threshold=0.2,
        fallback=ConstantVelocityFallback(),
        emit_predictions_on_miss=True,
    )

    tracker.update(
        [
            _detection(
                timestamp=1.0,
                bbox=(0.0, 0.0, 10.0, 10.0),
                pixel_coordinate=(5.0, 5.0),
            )
        ]
    )
    tracked = tracker.update(
        [
            _detection(
                timestamp=2.0,
                bbox=(2.0, 0.0, 12.0, 10.0),
                pixel_coordinate=(7.0, 5.0),
            )
        ]
    )
    predicted = tracker.update([], timestamp=3.0)

    assert predicted[0].track_id == tracked[0].track_id
    assert predicted[0].pixel_coordinate == (9.0, 5.0)
    assert predicted[0].bbox == (4.0, 0.0, 14.0, 10.0)
