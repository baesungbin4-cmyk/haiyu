"""Lightweight edge-side multi-object tracker interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol

from edge.common.schemas import (
    BBox,
    Coordinate2D,
    Detection,
    TrackId,
    TrackState,
    Velocity2D,
)
from edge.detect.postprocess import calculate_iou


class Tracker(Protocol):
    """Minimal tracker contract used by the edge processing pipeline."""

    def update(
        self,
        detections: Sequence[Detection],
        timestamp: float | None = None,
    ) -> list[TrackState]:
        """Associate detections and return current tracked states."""


class MotionFallback(Protocol):
    """Small CV-KF-compatible prediction hook for missed detections."""

    def predict(self, state: TrackState, timestamp: float) -> TrackState:
        """Predict a state at ``timestamp`` from the previous state."""


@dataclass(frozen=True)
class ConstantVelocityFallback:
    """Constant-velocity fallback with a Kalman-like public interface."""

    def predict(self, state: TrackState, timestamp: float) -> TrackState:
        if not isinstance(state, TrackState):
            raise TypeError("state must be a TrackState")
        timestamp = _coerce_non_negative_float(timestamp, "timestamp")
        dt = timestamp - state.timestamp
        if dt < 0:
            raise ValueError("timestamp must not move backwards")

        dx = state.velocity[0] * dt
        dy = state.velocity[1] * dt
        pixel_coordinate = _shift_coordinate(state.pixel_coordinate, dx, dy)
        bbox = _shift_bbox(state.bbox, dx, dy)
        world_coordinate = None
        if state.world_coordinate is not None:
            world_coordinate = _shift_coordinate(
                state.world_coordinate,
                dx,
                dy,
            )

        return TrackState(
            device_id=state.device_id,
            track_id=state.track_id,
            timestamp=timestamp,
            bbox=bbox,
            confidence=state.confidence,
            class_name=state.class_name,
            pixel_coordinate=pixel_coordinate,
            velocity=state.velocity,
            world_coordinate=world_coordinate,
        )


@dataclass(frozen=True)
class _ActiveTrack:
    track_id: TrackId
    device_id: str
    class_name: str
    timestamp: float
    bbox: BBox
    confidence: float
    pixel_coordinate: Coordinate2D
    velocity: Velocity2D
    world_coordinate: Coordinate2D | None
    missed: int = 0

    @classmethod
    def from_state(
        cls,
        state: TrackState,
        missed: int = 0,
    ) -> "_ActiveTrack":
        return cls(
            track_id=state.track_id,
            device_id=state.device_id,
            class_name=state.class_name,
            timestamp=state.timestamp,
            bbox=state.bbox,
            confidence=state.confidence,
            pixel_coordinate=state.pixel_coordinate,
            velocity=state.velocity,
            world_coordinate=state.world_coordinate,
            missed=missed,
        )

    def to_state(self) -> TrackState:
        return TrackState(
            device_id=self.device_id,
            track_id=self.track_id,
            timestamp=self.timestamp,
            bbox=self.bbox,
            confidence=self.confidence,
            class_name=self.class_name,
            pixel_coordinate=self.pixel_coordinate,
            velocity=self.velocity,
            world_coordinate=self.world_coordinate,
        )


class IoUTracker:
    """Simple IoU-based tracker suitable for edge fallback operation."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 1,
        fallback: MotionFallback | None = None,
        emit_predictions_on_miss: bool = False,
    ) -> None:
        self.iou_threshold = _coerce_probability(
            iou_threshold,
            "iou_threshold",
        )
        is_bool = isinstance(max_missed, bool)
        is_integer = isinstance(max_missed, Integral)
        if is_bool or not is_integer:
            raise TypeError("max_missed must be a non-negative integer")
        if max_missed < 0:
            raise ValueError("max_missed must be non-negative")

        self.max_missed = int(max_missed)
        self.fallback = fallback
        self.emit_predictions_on_miss = bool(emit_predictions_on_miss)
        self._next_track_id = 1
        self._tracks: dict[TrackId, _ActiveTrack] = {}

    @property
    def active_track_ids(self) -> tuple[TrackId, ...]:
        """Return currently retained track identifiers."""

        return tuple(self._tracks)

    def update(
        self,
        detections: Sequence[Detection],
        timestamp: float | None = None,
    ) -> list[TrackState]:
        """Associate detections with active tracks."""

        detections = _validate_detections(detections)
        if timestamp is not None:
            timestamp = _coerce_non_negative_float(timestamp, "timestamp")
        if not detections:
            return self._handle_missing_frame(timestamp)

        previous_track_ids = set(self._tracks)
        matches = self._match_detections(detections)
        matched_track_ids = set(matches.values())
        states: list[TrackState] = []

        for detection_index, detection in enumerate(detections):
            track_id = matches.get(detection_index)
            if track_id is None:
                track_id = self._allocate_track_id()
                previous = None
            else:
                previous = self._tracks[track_id]

            state = self._state_from_detection(
                detection=detection,
                track_id=track_id,
                previous=previous,
                timestamp=timestamp,
            )
            self._tracks[track_id] = _ActiveTrack.from_state(state)
            states.append(state)

        unmatched_track_ids = previous_track_ids - matched_track_ids
        self._age_unmatched_tracks(unmatched_track_ids)
        return states

    def _handle_missing_frame(
        self,
        timestamp: float | None,
    ) -> list[TrackState]:
        states: list[TrackState] = []
        for track_id, track in list(self._tracks.items()):
            missed = track.missed + 1
            if missed > self.max_missed:
                del self._tracks[track_id]
                continue

            if (
                self.fallback is not None
                and self.emit_predictions_on_miss
                and timestamp is not None
            ):
                state = self.fallback.predict(track.to_state(), timestamp)
                self._tracks[track_id] = _ActiveTrack.from_state(
                    state,
                    missed=missed,
                )
                states.append(state)
            else:
                self._tracks[track_id] = _ActiveTrack(
                    track_id=track.track_id,
                    device_id=track.device_id,
                    class_name=track.class_name,
                    timestamp=track.timestamp,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    pixel_coordinate=track.pixel_coordinate,
                    velocity=track.velocity,
                    world_coordinate=track.world_coordinate,
                    missed=missed,
                )
        return states

    def _match_detections(
        self,
        detections: Sequence[Detection],
    ) -> dict[int, TrackId]:
        candidates: list[tuple[float, str, int, TrackId]] = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self._tracks.items():
                if not _can_match(track, detection):
                    continue
                iou = calculate_iou(track.bbox, detection.bbox)
                if iou >= self.iou_threshold:
                    candidates.append(
                        (
                            iou,
                            str(track_id),
                            detection_index,
                            track_id,
                        )
                    )

        candidates.sort(reverse=True)
        matches: dict[int, TrackId] = {}
        used_tracks: set[TrackId] = set()
        for _, _, detection_index, track_id in candidates:
            if detection_index in matches or track_id in used_tracks:
                continue
            matches[detection_index] = track_id
            used_tracks.add(track_id)
        return matches

    def _state_from_detection(
        self,
        detection: Detection,
        track_id: TrackId,
        previous: _ActiveTrack | None,
        timestamp: float | None,
    ) -> TrackState:
        state_timestamp = detection.timestamp
        if timestamp is not None:
            state_timestamp = timestamp

        velocity = (0.0, 0.0)
        if previous is not None:
            velocity = _velocity_from_detection(
                previous=previous,
                detection=detection,
                timestamp=state_timestamp,
            )

        return TrackState(
            device_id=detection.device_id,
            track_id=track_id,
            timestamp=state_timestamp,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_name=detection.class_name,
            pixel_coordinate=detection.pixel_coordinate,
            velocity=velocity,
            world_coordinate=detection.world_coordinate,
        )

    def _age_unmatched_tracks(self, track_ids: set[TrackId]) -> None:
        for track_id in track_ids:
            track = self._tracks.get(track_id)
            if track is None:
                continue
            missed = track.missed + 1
            if missed > self.max_missed:
                del self._tracks[track_id]
            else:
                self._tracks[track_id] = _ActiveTrack(
                    track_id=track.track_id,
                    device_id=track.device_id,
                    class_name=track.class_name,
                    timestamp=track.timestamp,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    pixel_coordinate=track.pixel_coordinate,
                    velocity=track.velocity,
                    world_coordinate=track.world_coordinate,
                    missed=missed,
                )

    def _allocate_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id


class DeepSortCompatibleTracker(IoUTracker):
    """Non-ReID DeepSORT-compatible shim.

    This class keeps the expected tracker interface but uses IoU association
    plus optional constant-velocity fallback, not appearance embeddings.
    """


SimpleTracker = IoUTracker


def _velocity_from_detection(
    previous: _ActiveTrack,
    detection: Detection,
    timestamp: float,
) -> Velocity2D:
    dt = timestamp - previous.timestamp
    if dt <= 0:
        return (0.0, 0.0)

    has_previous_world = previous.world_coordinate is not None
    has_detection_world = detection.world_coordinate is not None
    if has_previous_world and has_detection_world:
        old_x, old_y = previous.world_coordinate
        new_x, new_y = detection.world_coordinate
    else:
        old_x, old_y = previous.pixel_coordinate
        new_x, new_y = detection.pixel_coordinate
    return ((new_x - old_x) / dt, (new_y - old_y) / dt)


def _can_match(track: _ActiveTrack, detection: Detection) -> bool:
    return (
        track.device_id == detection.device_id
        and track.class_name == detection.class_name
    )


def _validate_detections(
    detections: Sequence[Detection],
) -> list[Detection]:
    if detections is None:
        raise TypeError("detections must be a sequence of Detection objects")
    result = list(detections)
    for detection in result:
        if not isinstance(detection, Detection):
            raise TypeError("detections must contain Detection objects")
    return result


def _coerce_probability(value: float, field_name: str) -> float:
    result = _coerce_non_negative_float(value, field_name)
    if result > 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return result


def _coerce_non_negative_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a non-negative number")
    result = float(value)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _shift_coordinate(
    coordinate: Coordinate2D,
    dx: float,
    dy: float,
) -> Coordinate2D:
    return (coordinate[0] + dx, coordinate[1] + dy)


def _shift_bbox(bbox: BBox, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


__all__ = [
    "ConstantVelocityFallback",
    "DeepSortCompatibleTracker",
    "IoUTracker",
    "MotionFallback",
    "SimpleTracker",
    "Tracker",
]
