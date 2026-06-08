"""Shared edge-side data schemas.

The classes in this module define data exchanged between edge modules. They
only validate structure and basic value ranges; algorithm logic belongs in the
detector, tracker, predictor, and risk modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

Coordinate2D: TypeAlias = tuple[float, float]
BBox: TypeAlias = tuple[float, float, float, float]
Velocity2D: TypeAlias = tuple[float, float]
TrackId: TypeAlias = str | int
RiskLevel: TypeAlias = Literal["none", "low", "medium", "high", "critical"]

_VALID_RISK_LEVELS = frozenset({"none", "low", "medium", "high", "critical"})


def _coerce_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")

    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _coerce_non_negative_float(value: Any, field_name: str) -> float:
    result = _coerce_finite_float(value, field_name)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _coerce_optional_finite_float(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return _coerce_finite_float(value, field_name)


def _coerce_optional_non_negative_float(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None
    return _coerce_non_negative_float(value, field_name)


def _coerce_float_tuple(
    value: Any,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of {length} numbers")

    result = tuple(_coerce_finite_float(item, field_name) for item in value)
    if len(result) != length:
        raise ValueError(f"{field_name} must contain {length} numbers")
    return result


def _coerce_coordinate(value: Any, field_name: str) -> Coordinate2D:
    return _coerce_float_tuple(value, 2, field_name)


def _coerce_optional_coordinate(
    value: Any,
    field_name: str,
) -> Coordinate2D | None:
    if value is None:
        return None
    return _coerce_coordinate(value, field_name)


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_track_id(value: Any) -> None:
    if isinstance(value, bool):
        raise TypeError("track_id must be a string or non-negative integer")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("track_id must be a non-empty string")
        return
    if isinstance(value, Integral):
        if value < 0:
            raise ValueError("track_id must be non-negative")
        return
    raise TypeError("track_id must be a string or non-negative integer")


def _validate_confidence(value: Any) -> float:
    result = _coerce_finite_float(value, "confidence")
    if result < 0 or result > 1:
        raise ValueError("confidence must be between 0 and 1")
    return result


def _validate_bool(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")


def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        result = {}
        for field in fields(value):
            result[field.name] = _to_plain(getattr(value, field.name))
        return result
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class Detection:
    """Single detector output in pixel coordinates."""

    device_id: str
    timestamp: float
    bbox: BBox
    confidence: float
    class_name: str
    pixel_coordinate: Coordinate2D
    world_coordinate: Coordinate2D | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.device_id, "device_id")
        _validate_non_empty_string(self.class_name, "class_name")
        object.__setattr__(
            self,
            "timestamp",
            _coerce_non_negative_float(self.timestamp, "timestamp"),
        )
        object.__setattr__(
            self,
            "bbox",
            _coerce_float_tuple(self.bbox, 4, "bbox"),
        )
        object.__setattr__(
            self,
            "confidence",
            _validate_confidence(self.confidence),
        )
        object.__setattr__(
            self,
            "pixel_coordinate",
            _coerce_coordinate(self.pixel_coordinate, "pixel_coordinate"),
        )
        object.__setattr__(
            self,
            "world_coordinate",
            _coerce_optional_coordinate(
                self.world_coordinate,
                "world_coordinate",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class TrackState:
    """Tracked target state at one timestamp."""

    device_id: str
    track_id: TrackId
    timestamp: float
    bbox: BBox
    confidence: float
    class_name: str
    pixel_coordinate: Coordinate2D
    velocity: Velocity2D
    world_coordinate: Coordinate2D | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.device_id, "device_id")
        _validate_track_id(self.track_id)
        _validate_non_empty_string(self.class_name, "class_name")
        object.__setattr__(
            self,
            "timestamp",
            _coerce_non_negative_float(self.timestamp, "timestamp"),
        )
        object.__setattr__(
            self,
            "bbox",
            _coerce_float_tuple(self.bbox, 4, "bbox"),
        )
        object.__setattr__(
            self,
            "confidence",
            _validate_confidence(self.confidence),
        )
        object.__setattr__(
            self,
            "pixel_coordinate",
            _coerce_coordinate(self.pixel_coordinate, "pixel_coordinate"),
        )
        object.__setattr__(
            self,
            "velocity",
            _coerce_coordinate(self.velocity, "velocity"),
        )
        object.__setattr__(
            self,
            "world_coordinate",
            _coerce_optional_coordinate(
                self.world_coordinate,
                "world_coordinate",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class TrackHistory:
    """Ordered collection of states for one tracked target."""

    device_id: str
    track_id: TrackId
    timestamp: float
    states: tuple[TrackState, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.device_id, "device_id")
        _validate_track_id(self.track_id)
        object.__setattr__(
            self,
            "timestamp",
            _coerce_non_negative_float(self.timestamp, "timestamp"),
        )
        states = tuple(self.states)
        for state in states:
            if not isinstance(state, TrackState):
                raise TypeError("states must contain TrackState instances")
            if state.device_id != self.device_id:
                raise ValueError("state device_id mismatch")
            if state.track_id != self.track_id:
                raise ValueError("state track_id must match history track_id")
        object.__setattr__(self, "states", states)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class TrajectoryPoint:
    """Point in a predicted or observed trajectory."""

    device_id: str
    track_id: TrackId
    timestamp: float
    pixel_coordinate: Coordinate2D
    velocity: Velocity2D
    world_coordinate: Coordinate2D | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.device_id, "device_id")
        _validate_track_id(self.track_id)
        object.__setattr__(
            self,
            "timestamp",
            _coerce_non_negative_float(self.timestamp, "timestamp"),
        )
        object.__setattr__(
            self,
            "pixel_coordinate",
            _coerce_coordinate(self.pixel_coordinate, "pixel_coordinate"),
        )
        object.__setattr__(
            self,
            "velocity",
            _coerce_coordinate(self.velocity, "velocity"),
        )
        object.__setattr__(
            self,
            "world_coordinate",
            _coerce_optional_coordinate(
                self.world_coordinate,
                "world_coordinate",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class TrajectoryPrediction:
    """Predicted trajectory for one tracked target."""

    device_id: str
    track_id: TrackId
    timestamp: float
    points: tuple[TrajectoryPoint, ...]
    horizon_seconds: float | None = None
    model_name: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.device_id, "device_id")
        _validate_track_id(self.track_id)
        if self.model_name is not None:
            _validate_non_empty_string(self.model_name, "model_name")
        object.__setattr__(
            self,
            "timestamp",
            _coerce_non_negative_float(self.timestamp, "timestamp"),
        )
        object.__setattr__(
            self,
            "horizon_seconds",
            _coerce_optional_non_negative_float(
                self.horizon_seconds,
                "horizon_seconds",
            ),
        )
        points = tuple(self.points)
        for point in points:
            if not isinstance(point, TrajectoryPoint):
                raise TypeError("points must contain TrajectoryPoint")
            if point.device_id != self.device_id:
                raise ValueError("point device_id mismatch")
            if point.track_id != self.track_id:
                raise ValueError("point track_id mismatch")
        object.__setattr__(self, "points", points)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class RiskResult:
    """Risk assessment output for one tracked target."""

    device_id: str
    track_id: TrackId
    timestamp: float
    risk_level: RiskLevel
    should_alert: bool
    tcpa_seconds: float | None = None
    cpa_distance_m: float | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.device_id, "device_id")
        _validate_track_id(self.track_id)
        if self.risk_level not in _VALID_RISK_LEVELS:
            raise ValueError(
                "risk_level must be one of " f"{sorted(_VALID_RISK_LEVELS)}"
            )
        _validate_bool(self.should_alert, "should_alert")
        if self.message is not None:
            _validate_non_empty_string(self.message, "message")
        object.__setattr__(
            self,
            "timestamp",
            _coerce_non_negative_float(self.timestamp, "timestamp"),
        )
        object.__setattr__(
            self,
            "tcpa_seconds",
            _coerce_optional_finite_float(
                self.tcpa_seconds,
                "tcpa_seconds",
            ),
        )
        object.__setattr__(
            self,
            "cpa_distance_m",
            _coerce_optional_non_negative_float(
                self.cpa_distance_m,
                "cpa_distance_m",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class MqttEventEnvelope:
    """MQTT event wrapper for edge-to-cloud payloads."""

    device_id: str
    timestamp: float
    seq: int
    event_type: str
    topic: str
    payload: Mapping[str, Any]
    qos: int = 1
    retain: bool = False

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.device_id, "device_id")
        _validate_non_empty_string(self.event_type, "event_type")
        _validate_non_empty_string(self.topic, "topic")
        if not isinstance(self.seq, Integral) or isinstance(self.seq, bool):
            raise TypeError("seq must be a non-negative integer")
        if self.seq < 0:
            raise ValueError("seq must be non-negative")
        if not isinstance(self.qos, Integral) or isinstance(self.qos, bool):
            raise TypeError("qos must be 0, 1, or 2")
        if self.qos not in (0, 1, 2):
            raise ValueError("qos must be 0, 1, or 2")
        _validate_bool(self.retain, "retain")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        object.__setattr__(
            self,
            "timestamp",
            _coerce_non_negative_float(self.timestamp, "timestamp"),
        )
        object.__setattr__(
            self,
            "payload",
            MappingProxyType(dict(self.payload)),
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)
