import numpy as np

from edge.common.schemas import Detection
from edge.edge_processor_v2 import (
    EdgeProcessor,
    EdgeProcessorConfig,
    MockDetector,
    MockTrajectoryPredictor,
    RecordingMqttPublisher,
)


def _frame() -> np.ndarray:
    return np.full((12, 16, 3), 24, dtype=np.uint8)


def _danger_detection() -> Detection:
    return Detection(
        device_id="edge-01",
        timestamp=0.0,
        bbox=(35.0, 0.0, 45.0, 10.0),
        confidence=0.9,
        class_name="vessel",
        pixel_coordinate=(40.0, 0.0),
        world_coordinate=(40.0, 0.0),
    )


def _safe_detection() -> Detection:
    return Detection(
        device_id="edge-01",
        timestamp=0.0,
        bbox=(195.0, 195.0, 205.0, 205.0),
        confidence=0.9,
        class_name="vessel",
        pixel_coordinate=(200.0, 200.0),
        world_coordinate=(200.0, 200.0),
    )


def test_mock_frame_pipeline_runs_end_to_end() -> None:
    publisher = RecordingMqttPublisher()
    processor = EdgeProcessor(
        detector=MockDetector([_danger_detection()]),
        predictor=MockTrajectoryPredictor(),
        mqtt_publisher=publisher,
        config=EdgeProcessorConfig(history_len=1, fps=10.0),
        clock=lambda: 100.0,
    )

    result = processor.process_frame(_frame())

    assert result.errors == ()
    assert len(result.detections) == 1
    assert len(result.tracks) == 1
    assert len(result.predictions) == 1
    assert len(result.risks) == 1
    assert len(result.events) == 1
    assert publisher.events == list(result.events)
    assert result.risks[0].timestamp == 100.0
    assert result.risks[0].risk_level in ("medium", "high", "critical")


def test_detector_failure_does_not_crash_loop() -> None:
    processor = EdgeProcessor(
        detector=MockDetector(error=RuntimeError("detector unavailable")),
        config=EdgeProcessorConfig(history_len=1),
        clock=lambda: 100.0,
    )

    result = processor.process_frame(_frame())

    assert result.detections == ()
    assert result.tracks == ()
    assert result.risks == ()
    assert result.events == ()
    assert result.errors
    assert "detect" in result.errors[0]


def test_insufficient_history_skips_predictor_safely() -> None:
    processor = EdgeProcessor(
        detector=MockDetector([_safe_detection()]),
        predictor=MockTrajectoryPredictor(),
        config=EdgeProcessorConfig(history_len=2),
        clock=lambda: 100.0,
    )

    result = processor.process_frame(_frame())

    assert result.errors == ()
    assert len(result.tracks) == 1
    assert result.predictions == ()
    assert len(result.risks) == 1


def test_risk_result_appears_when_mock_trajectory_indicates_danger() -> None:
    future_offsets = [(5.0, -170.0, -200.0)]
    processor = EdgeProcessor(
        detector=MockDetector([_safe_detection()]),
        predictor=MockTrajectoryPredictor(future_offsets=future_offsets),
        config=EdgeProcessorConfig(history_len=1),
        clock=lambda: 100.0,
    )

    result = processor.process_frame(_frame())

    assert len(result.predictions) == 1
    assert len(result.risks) == 1
    assert result.risks[0].should_alert is True
    assert result.risks[0].risk_level != "none"
    assert result.risks[0].cpa_distance_m is not None
    assert result.risks[0].cpa_distance_m < 50.0
