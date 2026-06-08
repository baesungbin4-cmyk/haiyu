"""TCPA/CPA risk engine that returns shared ``RiskResult`` schemas."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TypeAlias

from edge.common.schemas import (
    RiskResult,
    TrackId,
    TrackState,
    TrajectoryPoint,
    TrajectoryPrediction,
)
from edge.risk.tcpa_cpa import (
    Vector2D,
    calculate_tcpa_cpa,
    coerce_vector2,
    vector_distance_m,
)

DEFAULT_TCPA_ALERT_SECONDS = 30.0
DEFAULT_CPA_ALERT_DISTANCE_M = 50.0

CoordinateSource: TypeAlias = TrackState | TrajectoryPoint


# ---------------------------------------------------------------------------
# Configurable risk band thresholds (contract §6: "阈值可配" / review M4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskThresholds:
    """Configurable TCPA/CPA band thresholds for risk-level classification.

    All values are non-negative floats.  Bands are checked in descending
    severity order so the most urgent level wins.

    The "low" band (contract §6 "低") sits between "none" (no alert)
    and "medium" — it captures targets that are within the overall alert
    envelope but still far enough in both TCPA and CPA dimensions to be
    considered low-priority.
    """

    tcpa_alert: float = DEFAULT_TCPA_ALERT_SECONDS
    cpa_alert: float = DEFAULT_CPA_ALERT_DISTANCE_M

    # Critical: both dimensions are dangerously tight
    tcpa_critical: float = 10.0
    cpa_critical: float = 20.0

    # High: at least one dimension is tight
    tcpa_high: float = 20.0
    cpa_high: float = 35.0

    # Low: both dimensions are comfortably within the alert envelope
    tcpa_low: float = 25.0
    cpa_low: float = 40.0

    def __post_init__(self) -> None:
        for name in (
            "tcpa_alert",
            "cpa_alert",
            "tcpa_critical",
            "cpa_critical",
            "tcpa_high",
            "cpa_high",
            "tcpa_low",
            "cpa_low",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite number")
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FutureDistanceResult:
    """Minimum sampled future distance from a predicted trajectory."""

    time_to_min_seconds: float
    distance_m: float


def _coerce_non_negative_threshold(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _position_from_schema(
    source: CoordinateSource,
    use_world_coordinate: bool,
) -> Vector2D:
    if use_world_coordinate and source.world_coordinate is not None:
        return source.world_coordinate
    return source.pixel_coordinate


def _relative_position(
    own_position_m: Sequence[float],
    target_position_m: Sequence[float],
) -> Vector2D:
    own_x, own_y = coerce_vector2(own_position_m, "own_position_m")
    target_x, target_y = coerce_vector2(target_position_m, "target_position_m")
    return (target_x - own_x, target_y - own_y)


def _relative_velocity(
    own_velocity_mps: Sequence[float],
    target_velocity_mps: Sequence[float],
) -> Vector2D:
    own_x, own_y = coerce_vector2(own_velocity_mps, "own_velocity_mps")
    target_x, target_y = coerce_vector2(
        target_velocity_mps,
        "target_velocity_mps",
    )
    return (target_x - own_x, target_y - own_y)


def calculate_min_future_distance(
    own_position_m: Sequence[float],
    own_velocity_mps: Sequence[float],
    own_timestamp: float,
    prediction: TrajectoryPrediction,
    use_world_coordinate: bool = True,
) -> FutureDistanceResult | None:
    """Return minimum sampled future distance against a target prediction."""

    if not isinstance(prediction, TrajectoryPrediction):
        raise TypeError("prediction must be a TrajectoryPrediction")
    own_position = coerce_vector2(own_position_m, "own_position_m")
    own_velocity = coerce_vector2(own_velocity_mps, "own_velocity_mps")
    reference_timestamp = _coerce_non_negative_threshold(
        own_timestamp,
        "own_timestamp",
    )

    best: FutureDistanceResult | None = None
    for point in prediction.points:
        time_offset = point.timestamp - reference_timestamp
        if time_offset < 0:
            continue
        own_future_position = (
            own_position[0] + own_velocity[0] * time_offset,
            own_position[1] + own_velocity[1] * time_offset,
        )
        target_future_position = _position_from_schema(
            point,
            use_world_coordinate,
        )
        distance = vector_distance_m(
            own_future_position,
            target_future_position,
        )
        if best is None or distance < best.distance_m:
            best = FutureDistanceResult(
                time_to_min_seconds=time_offset,
                distance_m=distance,
            )
    return best


def should_alert(
    tcpa_seconds: float,
    cpa_distance_m: float,
    tcpa_threshold_seconds: float = DEFAULT_TCPA_ALERT_SECONDS,
    cpa_threshold_m: float = DEFAULT_CPA_ALERT_DISTANCE_M,
) -> bool:
    """Return whether TCPA/CPA metrics trigger the configured alert rule."""

    tcpa_threshold = _coerce_non_negative_threshold(
        tcpa_threshold_seconds,
        "tcpa_threshold_seconds",
    )
    cpa_threshold = _coerce_non_negative_threshold(
        cpa_threshold_m,
        "cpa_threshold_m",
    )
    return (
        tcpa_seconds >= 0.0
        and tcpa_seconds < tcpa_threshold
        and cpa_distance_m < cpa_threshold
    )


def classify_risk_level(
    tcpa_seconds: float,
    cpa_distance_m: float,
    tcpa_threshold_seconds: float = DEFAULT_TCPA_ALERT_SECONDS,
    cpa_threshold_m: float = DEFAULT_CPA_ALERT_DISTANCE_M,
    *,
    thresholds: RiskThresholds | None = None,
) -> str:
    """Classify TCPA/CPA metrics into configured severity bands.

    Bands are checked in descending severity so the most urgent level
    wins.  With default thresholds::

        none     — not alerting
        low      — alerting but tcpa≥25s AND cpa≥40m (safe margin)
        medium   — alerting, not low/critical/high
        high     — tcpa<20s OR cpa<35m (but not critical)
        critical — tcpa<10s AND cpa<20m

    Pass a ``RiskThresholds`` instance to customise individual band
    boundaries (contract §6: "阈值可配").
    """
    t = thresholds or RiskThresholds(
        tcpa_alert=tcpa_threshold_seconds,
        cpa_alert=cpa_threshold_m,
    )

    if not should_alert(
        tcpa_seconds,
        cpa_distance_m,
        tcpa_threshold_seconds=t.tcpa_alert,
        cpa_threshold_m=t.cpa_alert,
    ):
        return "none"

    if tcpa_seconds < t.tcpa_critical and cpa_distance_m < t.cpa_critical:
        return "critical"
    if tcpa_seconds < t.tcpa_high or cpa_distance_m < t.cpa_high:
        return "high"
    if tcpa_seconds >= t.tcpa_low and cpa_distance_m >= t.cpa_low:
        return "low"
    return "medium"


def evaluate_risk(
    *,
    device_id: str,
    track_id: TrackId,
    timestamp: float,
    own_position_m: Sequence[float],
    own_velocity_mps: Sequence[float],
    target_position_m: Sequence[float],
    target_velocity_mps: Sequence[float],
    prediction: TrajectoryPrediction | None = None,
    tcpa_threshold_seconds: float = DEFAULT_TCPA_ALERT_SECONDS,
    cpa_threshold_m: float = DEFAULT_CPA_ALERT_DISTANCE_M,
    risk_thresholds: RiskThresholds | None = None,
    use_world_coordinate: bool = True,
) -> RiskResult:
    """Evaluate one target against ownship state and optional prediction."""

    relative_position = _relative_position(own_position_m, target_position_m)
    relative_velocity = _relative_velocity(
        own_velocity_mps,
        target_velocity_mps,
    )
    metrics = calculate_tcpa_cpa(
        relative_position,
        relative_velocity,
    )
    tcpa_seconds = metrics.tcpa_seconds
    cpa_distance_m = metrics.cpa_distance_m

    if prediction is not None:
        if prediction.device_id != device_id:
            raise ValueError("prediction device_id mismatch")
        if prediction.track_id != track_id:
            raise ValueError("prediction track_id mismatch")
        future_metrics = calculate_min_future_distance(
            own_position_m,
            own_velocity_mps,
            timestamp,
            prediction,
            use_world_coordinate=use_world_coordinate,
        )
        if future_metrics is not None:
            future_distance_m = future_metrics.distance_m
            if future_distance_m < cpa_distance_m:
                cpa_distance_m = future_distance_m
                tcpa_seconds = min(
                    tcpa_seconds,
                    future_metrics.time_to_min_seconds,
                )

    thresholds = risk_thresholds or RiskThresholds(
        tcpa_alert=tcpa_threshold_seconds,
        cpa_alert=cpa_threshold_m,
    )
    alert = should_alert(
        tcpa_seconds,
        cpa_distance_m,
        tcpa_threshold_seconds=thresholds.tcpa_alert,
        cpa_threshold_m=thresholds.cpa_alert,
    )
    return RiskResult(
        device_id=device_id,
        track_id=track_id,
        timestamp=timestamp,
        risk_level=classify_risk_level(
            tcpa_seconds,
            cpa_distance_m,
            thresholds=thresholds,
        ),
        should_alert=alert,
        tcpa_seconds=tcpa_seconds,
        cpa_distance_m=cpa_distance_m,
        message="TCPA/CPA alert" if alert else "TCPA/CPA clear",
    )


def evaluate_track_risk(
    own_state: TrackState,
    target_state: TrackState,
    prediction: TrajectoryPrediction | None = None,
    tcpa_threshold_seconds: float = DEFAULT_TCPA_ALERT_SECONDS,
    cpa_threshold_m: float = DEFAULT_CPA_ALERT_DISTANCE_M,
    risk_thresholds: RiskThresholds | None = None,
    use_world_coordinate: bool = True,
) -> RiskResult:
    """Evaluate risk from schema track states."""

    if not isinstance(own_state, TrackState):
        raise TypeError("own_state must be a TrackState")
    if not isinstance(target_state, TrackState):
        raise TypeError("target_state must be a TrackState")
    if prediction is not None:
        if prediction.device_id != target_state.device_id:
            raise ValueError("prediction device_id must match target_state")
        if prediction.track_id != target_state.track_id:
            raise ValueError("prediction track_id must match target_state")

    return evaluate_risk(
        device_id=target_state.device_id,
        track_id=target_state.track_id,
        timestamp=target_state.timestamp,
        own_position_m=_position_from_schema(own_state, use_world_coordinate),
        own_velocity_mps=own_state.velocity,
        target_position_m=_position_from_schema(
            target_state,
            use_world_coordinate,
        ),
        target_velocity_mps=target_state.velocity,
        prediction=prediction,
        tcpa_threshold_seconds=tcpa_threshold_seconds,
        cpa_threshold_m=cpa_threshold_m,
        risk_thresholds=risk_thresholds,
        use_world_coordinate=use_world_coordinate,
    )


__all__ = [
    "DEFAULT_CPA_ALERT_DISTANCE_M",
    "DEFAULT_TCPA_ALERT_SECONDS",
    "FutureDistanceResult",
    "RiskThresholds",
    "calculate_min_future_distance",
    "classify_risk_level",
    "evaluate_risk",
    "evaluate_track_risk",
    "should_alert",
]
