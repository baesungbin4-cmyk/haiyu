"""Edge processing loop skeleton for frame-to-alert flow.

Performance values in project contracts are engineering targets or
``样机级，未经实港验证`` unless measured in a deployment-specific benchmark.
This module does not claim measured latency.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from edge.common.schemas import (
    Detection,
    MqttEventEnvelope,
    RiskResult,
    TrackState,
    TrajectoryPoint,
    TrajectoryPrediction,
)
from edge.enhance.lowlight import LowLightConfig, enhance_low_light
from edge.risk.risk_engine import evaluate_track_risk


class DetectionRunner(Protocol):
    def detect(
        self,
        frame: np.ndarray,
        *,
        timestamp: float,
    ) -> list[Detection]: ...


class Tracker(Protocol):
    def update(
        self,
        detections: Sequence[Detection],
        *,
        timestamp: float,
    ) -> list[TrackState]: ...


class TrajectoryPredictor(Protocol):
    def predict(
        self,
        history: Sequence[TrackState],
        *,
        timestamp: float,
    ) -> TrajectoryPrediction: ...


class RiskEvaluator(Protocol):
    def evaluate(
        self,
        target_state: TrackState,
        prediction: TrajectoryPrediction | None,
        *,
        timestamp: float,
    ) -> RiskResult: ...


class MqttPublisher(Protocol):
    def publish(self, event: MqttEventEnvelope) -> None: ...


@dataclass(frozen=True)
class EdgeProcessorConfig:
    """Runtime configuration for the edge processing loop."""

    device_id: str = "edge-01"
    fps: float = 15.0
    history_len: int = 4
    lowlight_config: LowLightConfig | None = None
    ownship_track_id: str = "ownship"

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        if not self.device_id.strip():
            raise ValueError("device_id must be non-empty")


@dataclass(frozen=True)
class FrameProcessingResult:
    """Structured output for one processed frame."""

    timestamp: float
    detections: tuple[Detection, ...]
    tracks: tuple[TrackState, ...]
    predictions: tuple[TrajectoryPrediction, ...]
    risks: tuple[RiskResult, ...]
    events: tuple[MqttEventEnvelope, ...]
    errors: tuple[str, ...]


class EdgeProcessor:
    """Runs one frame through the edge processing chain."""

    def __init__(
        self,
        *,
        detector: DetectionRunner,
        tracker: Tracker | None = None,
        predictor: TrajectoryPredictor | None = None,
        risk_evaluator: RiskEvaluator | None = None,
        mqtt_publisher: MqttPublisher | None = None,
        config: EdgeProcessorConfig | None = None,
        logger: logging.Logger | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or EdgeProcessorConfig()
        self.detector = detector
        self.tracker = tracker or SimpleTracker(self.config.device_id)
        self.predictor = predictor
        self.risk_evaluator = risk_evaluator or SimpleRiskEvaluator(
            device_id=self.config.device_id,
            ownship_track_id=self.config.ownship_track_id,
        )
        self.mqtt_publisher = mqtt_publisher
        self.logger = logger or logging.getLogger(__name__)
        self.clock = clock or time.time
        self._history: dict[Any, deque[TrackState]] = defaultdict(
            lambda: deque(maxlen=self.config.history_len)
        )
        self._seq = 0

    def process_frame(self, frame: np.ndarray) -> FrameProcessingResult:
        timestamp = self.clock()
        errors: list[str] = []
        events: list[MqttEventEnvelope] = []
        predictions: list[TrajectoryPrediction] = []
        risks: list[RiskResult] = []

        enhanced_frame = self._run_stage(
            "enhance",
            lambda: self._enhance(frame),
            errors,
            fallback=frame,
        )
        detections = self._run_stage(
            "detect",
            lambda: self.detector.detect(enhanced_frame, timestamp=timestamp),
            errors,
            fallback=[],
        )
        tracks = self._run_stage(
            "track",
            lambda: self.tracker.update(detections, timestamp=timestamp),
            errors,
            fallback=[],
        )

        for track in tracks:
            history = self._history[track.track_id]
            history.append(track)
            prediction = self._maybe_predict(track, history, timestamp, errors)
            if prediction is not None:
                predictions.append(prediction)
            risk = self._evaluate_risk(track, prediction, timestamp, errors)
            if risk is None:
                continue
            risks.append(risk)
            event = self._make_alarm_event(risk)
            events.append(event)
            self._publish(event, errors)

        result = FrameProcessingResult(
            timestamp=timestamp,
            detections=tuple(detections),
            tracks=tuple(tracks),
            predictions=tuple(predictions),
            risks=tuple(risks),
            events=tuple(events),
            errors=tuple(errors),
        )
        self.logger.info(
            "edge_frame_processed",
            extra={
                "device_id": self.config.device_id,
                "timestamp": timestamp,
                "detections": len(result.detections),
                "tracks": len(result.tracks),
                "risks": [risk.risk_level for risk in result.risks],
                "errors": list(result.errors),
            },
        )
        return result

    def run(
        self,
        frames: Iterable[np.ndarray],
        *,
        max_frames: int | None = None,
    ) -> list[FrameProcessingResult]:
        results: list[FrameProcessingResult] = []
        min_interval = 1.0 / self.config.fps
        last_start: float | None = None
        for index, frame in enumerate(frames):
            if max_frames is not None and index >= max_frames:
                break
            started = self.clock()
            if last_start is not None:
                elapsed = started - last_start
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            last_start = self.clock()
            results.append(self.process_frame(frame))
        return results

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        config = self.config.lowlight_config
        if config is None or config.mode == "disabled":
            return frame.copy()
        return enhance_low_light(frame, config)

    def _maybe_predict(
        self,
        track: TrackState,
        history: deque[TrackState],
        timestamp: float,
        errors: list[str],
    ) -> TrajectoryPrediction | None:
        if self.predictor is None or len(history) < self.config.history_len:
            return None
        return self._run_stage(
            "predict",
            lambda: self.predictor.predict(
                tuple(history),
                timestamp=timestamp,
            ),
            errors,
            fallback=None,
        )

    def _evaluate_risk(
        self,
        track: TrackState,
        prediction: TrajectoryPrediction | None,
        timestamp: float,
        errors: list[str],
    ) -> RiskResult | None:
        return self._run_stage(
            "risk",
            lambda: self.risk_evaluator.evaluate(
                track,
                prediction,
                timestamp=timestamp,
            ),
            errors,
            fallback=None,
        )

    def _make_alarm_event(self, risk: RiskResult) -> MqttEventEnvelope:
        event = MqttEventEnvelope(
            device_id=self.config.device_id,
            timestamp=risk.timestamp,
            seq=self._seq,
            event_type="alarm",
            topic=f"haiyu/{self.config.device_id}/alarm",
            payload=risk.to_dict(),
            qos=1,
            retain=False,
        )
        self._seq += 1
        return event

    def _publish(
        self,
        event: MqttEventEnvelope,
        errors: list[str],
    ) -> None:
        if self.mqtt_publisher is None:
            return
        self._run_stage(
            "mqtt_publish",
            lambda: self.mqtt_publisher.publish(event),
            errors,
            fallback=None,
        )

    def _run_stage(
        self,
        stage: str,
        operation: Callable[[], Any],
        errors: list[str],
        fallback: Any,
    ) -> Any:
        try:
            return operation()
        except Exception as exc:  # pragma: no cover
            message = f"{stage}: {exc}"
            errors.append(message)
            self.logger.exception(
                "edge_stage_failed",
                extra={"stage": stage, "device_id": self.config.device_id},
            )
            return fallback


class SimpleTracker:
    """Detection-to-track adapter for mock and CPU-only tests."""

    def __init__(self, device_id: str) -> None:
        self.device_id = device_id

    def update(
        self,
        detections: Sequence[Detection],
        *,
        timestamp: float,
    ) -> list[TrackState]:
        tracks: list[TrackState] = []
        for index, detection in enumerate(detections):
            bbox = detection.bbox
            tracks.append(
                TrackState(
                    device_id=self.device_id,
                    track_id=index,
                    timestamp=timestamp,
                    bbox=bbox,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                    pixel_coordinate=detection.pixel_coordinate,
                    velocity=(0.0, 0.0),
                    world_coordinate=detection.world_coordinate,
                )
            )
        return tracks


class SimpleRiskEvaluator:
    """TCPA/CPA evaluator with a static ownship state for mock mode."""

    def __init__(
        self,
        *,
        device_id: str,
        ownship_track_id: str = "ownship",
        own_position_m: tuple[float, float] = (0.0, 0.0),
        own_velocity_mps: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.device_id = device_id
        self.own_state = TrackState(
            device_id=device_id,
            track_id=ownship_track_id,
            timestamp=0.0,
            bbox=(0.0, 0.0, 0.0, 0.0),
            confidence=1.0,
            class_name="ownship",
            pixel_coordinate=own_position_m,
            velocity=own_velocity_mps,
            world_coordinate=own_position_m,
        )

    def evaluate(
        self,
        target_state: TrackState,
        prediction: TrajectoryPrediction | None,
        *,
        timestamp: float,
    ) -> RiskResult:
        own_state = TrackState(
            device_id=self.device_id,
            track_id=self.own_state.track_id,
            timestamp=timestamp,
            bbox=self.own_state.bbox,
            confidence=self.own_state.confidence,
            class_name=self.own_state.class_name,
            pixel_coordinate=self.own_state.pixel_coordinate,
            velocity=self.own_state.velocity,
            world_coordinate=self.own_state.world_coordinate,
        )
        return evaluate_track_risk(own_state, target_state, prediction)


class MockDetector:
    """Detector returning prebuilt detections or a configured failure.

    **For unit tests only.**  The production detector is
    ``YOLOAttentionDetector`` (``edge/detect/yolo_attention_detector.py``),
    which loads real weights and optionally injects ``SEResidualAttention``
    into the YOLO PAN-FPN neck (contract §3).
    """

    def __init__(
        self,
        detections: Sequence[Detection] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.detections = tuple(detections or ())
        self.error = error

    def detect(
        self,
        frame: np.ndarray,
        *,
        timestamp: float,
    ) -> list[Detection]:
        if self.error is not None:
            raise self.error
        output = []
        for detection in self.detections:
            output.append(
                Detection(
                    device_id=detection.device_id,
                    timestamp=timestamp,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                    pixel_coordinate=detection.pixel_coordinate,
                    world_coordinate=detection.world_coordinate,
                )
            )
        return output


class MockTrajectoryPredictor:
    """Predictor returning a simple straight-line future from track history."""

    def __init__(
        self,
        future_offsets: Sequence[tuple[float, float, float]] | None = None,
    ) -> None:
        self.future_offsets = tuple(future_offsets or ((1.0, 0.0, 0.0),))

    def predict(
        self,
        history: Sequence[TrackState],
        *,
        timestamp: float,
    ) -> TrajectoryPrediction:
        latest = history[-1]
        base = latest.world_coordinate or latest.pixel_coordinate
        points = []
        for seconds, dx_value, dy_value in self.future_offsets:
            point = (base[0] + dx_value, base[1] + dy_value)
            points.append(
                TrajectoryPoint(
                    device_id=latest.device_id,
                    track_id=latest.track_id,
                    timestamp=timestamp + seconds,
                    pixel_coordinate=point,
                    velocity=latest.velocity,
                    world_coordinate=point,
                )
            )
        return TrajectoryPrediction(
            device_id=latest.device_id,
            track_id=latest.track_id,
            timestamp=timestamp,
            points=tuple(points),
            horizon_seconds=max(offset[0] for offset in self.future_offsets),
            model_name="mock-trajectory",
        )


class RecordingMqttPublisher:
    """In-memory MQTT hook for tests."""

    def __init__(self) -> None:
        self.events: list[MqttEventEnvelope] = []

    def publish(self, event: MqttEventEnvelope) -> None:
        self.events.append(event)


__all__ = [
    "DetectionRunner",
    "EdgeProcessor",
    "EdgeProcessorConfig",
    "FrameProcessingResult",
    "MockDetector",
    "MockTrajectoryPredictor",
    "RecordingMqttPublisher",
    "SimpleRiskEvaluator",
    "SimpleTracker",
    "Tracker",
    "TrajectoryPredictor",
]
