import pytest

from edge.common.schemas import (
    TrackState,
    TrajectoryPoint,
    TrajectoryPrediction,
)
from edge.risk.risk_engine import (
    calculate_min_future_distance,
    classify_risk_level,
    evaluate_risk,
    evaluate_track_risk,
    should_alert,
)

BASE_TS = 1_718_000_000.0


def _state(
    track_id: int | str,
    world_x: float,
    world_y: float,
    velocity_x: float,
    velocity_y: float,
    timestamp: float = BASE_TS,
) -> TrackState:
    return TrackState(
        device_id="edge-01",
        track_id=track_id,
        timestamp=timestamp,
        bbox=(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        class_name="vessel",
        pixel_coordinate=(world_x, world_y),
        velocity=(velocity_x, velocity_y),
        world_coordinate=(world_x, world_y),
    )


def _prediction(
    *points: TrajectoryPoint,
    device_id: str = "edge-01",
    track_id: int | str = 7,
) -> TrajectoryPrediction:
    return TrajectoryPrediction(
        device_id=device_id,
        track_id=track_id,
        timestamp=BASE_TS,
        points=points,
        horizon_seconds=30.0,
        model_name="lstm-causal-attention",
    )


def _point(
    timestamp_offset: float,
    x: float,
    y: float,
    device_id: str = "edge-01",
    track_id: int | str = 7,
) -> TrajectoryPoint:
    return TrajectoryPoint(
        device_id=device_id,
        track_id=track_id,
        timestamp=BASE_TS + timestamp_offset,
        pixel_coordinate=(x, y),
        velocity=(0.0, 0.0),
        world_coordinate=(x, y),
    )


def test_evaluate_risk_alerts_for_approaching_target() -> None:
    result = evaluate_risk(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(0.0, 0.0),
        target_position_m=(100.0, 0.0),
        target_velocity_mps=(-5.0, 0.0),
    )

    assert result.risk_level == "high"
    assert result.should_alert is True
    assert result.tcpa_seconds == 20.0
    assert result.cpa_distance_m == 0.0


def test_evaluate_risk_does_not_alert_when_target_is_leaving() -> None:
    result = evaluate_risk(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(0.0, 0.0),
        target_position_m=(20.0, 0.0),
        target_velocity_mps=(5.0, 0.0),
    )

    assert result.risk_level == "none"
    assert result.should_alert is False
    assert result.tcpa_seconds == -4.0


def test_evaluate_risk_handles_zero_relative_velocity() -> None:
    result = evaluate_risk(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(1.0, 0.0),
        target_position_m=(30.0, 0.0),
        target_velocity_mps=(1.0, 0.0),
    )

    assert result.should_alert is True
    assert result.tcpa_seconds == 0.0
    assert result.cpa_distance_m == 30.0


def test_evaluate_risk_does_not_alert_for_wide_crossing() -> None:
    result = evaluate_risk(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(0.0, 0.0),
        target_position_m=(100.0, 80.0),
        target_velocity_mps=(-5.0, 0.0),
    )

    assert result.should_alert is False
    assert result.tcpa_seconds == 20.0
    assert result.cpa_distance_m == 80.0


def test_predicted_future_distance_can_trigger_alert() -> None:
    prediction = _prediction(
        _point(5.0, 100.0, 0.0),
        _point(10.0, 40.0, 0.0),
        _point(15.0, 120.0, 0.0),
    )

    result = evaluate_risk(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(0.0, 0.0),
        target_position_m=(200.0, 0.0),
        target_velocity_mps=(0.0, 0.0),
        prediction=prediction,
    )

    assert result.should_alert is True
    assert result.tcpa_seconds == 0.0
    assert result.cpa_distance_m == 40.0


def test_prediction_tightens_tcpa_when_more_dangerous() -> None:
    prediction = _prediction(_point(25.0, 30.0, 0.0))

    result = evaluate_risk(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(0.0, 0.0),
        target_position_m=(50.0, 50.0),
        target_velocity_mps=(-1.0, 0.0),
        prediction=prediction,
    )

    assert result.should_alert is True
    assert result.tcpa_seconds == 25.0
    assert result.cpa_distance_m == 30.0


def test_prediction_does_not_relax_cpa() -> None:
    prediction = _prediction(_point(25.0, 80.0, 0.0))

    result = evaluate_risk(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(0.0, 0.0),
        target_position_m=(50.0, 50.0),
        target_velocity_mps=(-1.0, 0.0),
        prediction=prediction,
    )

    assert result.should_alert is False
    assert result.tcpa_seconds == 50.0
    assert result.cpa_distance_m == 50.0


def test_evaluate_risk_rejects_mismatched_prediction() -> None:
    wrong_track = _prediction(
        _point(10.0, 40.0, 0.0, track_id=8),
        track_id=8,
    )
    with pytest.raises(ValueError, match="track_id"):
        evaluate_risk(
            device_id="edge-01",
            track_id=7,
            timestamp=BASE_TS,
            own_position_m=(0.0, 0.0),
            own_velocity_mps=(0.0, 0.0),
            target_position_m=(100.0, 0.0),
            target_velocity_mps=(-5.0, 0.0),
            prediction=wrong_track,
        )

    wrong_device = _prediction(
        _point(10.0, 40.0, 0.0, device_id="edge-02"),
        device_id="edge-02",
    )
    with pytest.raises(ValueError, match="device_id"):
        evaluate_risk(
            device_id="edge-01",
            track_id=7,
            timestamp=BASE_TS,
            own_position_m=(0.0, 0.0),
            own_velocity_mps=(0.0, 0.0),
            target_position_m=(100.0, 0.0),
            target_velocity_mps=(-5.0, 0.0),
            prediction=wrong_device,
        )


def test_should_alert_at_boundary_values() -> None:
    assert should_alert(29.999, 49.999) is True
    assert should_alert(30.0, 49.999) is False
    assert should_alert(29.999, 50.0) is False
    assert should_alert(-0.001, 10.0) is False


def test_min_future_distance_all_points_in_past() -> None:
    prediction = _prediction(
        _point(-10.0, 10.0, 0.0),
        _point(-1.0, 5.0, 0.0),
    )

    assert (
        calculate_min_future_distance(
            own_position_m=(0.0, 0.0),
            own_velocity_mps=(0.0, 0.0),
            own_timestamp=BASE_TS,
            prediction=prediction,
        )
        is None
    )

    result = evaluate_risk(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(0.0, 0.0),
        target_position_m=(100.0, 0.0),
        target_velocity_mps=(-5.0, 0.0),
        prediction=prediction,
    )

    assert result.tcpa_seconds == 20.0
    assert result.cpa_distance_m == 0.0


def test_risk_level_granularity() -> None:
    assert classify_risk_level(5.0, 10.0) == "critical"
    assert classify_risk_level(15.0, 40.0) == "high"
    assert classify_risk_level(25.0, 45.0) == "low"  # 25≥25 AND 45≥40
    assert classify_risk_level(22.0, 42.0) == "medium"  # 22<25, not low
    assert classify_risk_level(30.0, 45.0) == "none"


def test_classify_risk_level_covers_all_five_levels() -> None:
    """Review item S2: verify all five RiskLevel values are reachable."""
    # none
    assert classify_risk_level(35.0, 10.0) == "none"  # tcpa ≥ 30
    assert classify_risk_level(5.0, 60.0) == "none"  # cpa ≥ 50
    assert classify_risk_level(-1.0, 10.0) == "none"  # tcpa < 0 (leaving)

    # low: both dimensions comfortably within alert envelope
    assert classify_risk_level(26.0, 41.0) == "low"

    # medium: alerting but not low/high/critical
    assert classify_risk_level(22.0, 42.0) == "medium"  # tcpa < 25, cpa >= 40
    assert classify_risk_level(26.0, 38.0) == "medium"  # tcpa >= 25, cpa < 40

    # high: one dimension tight
    assert classify_risk_level(15.0, 40.0) == "high"  # tcpa < 20
    assert classify_risk_level(25.0, 30.0) == "high"  # cpa < 35

    # critical: both tight
    assert classify_risk_level(5.0, 10.0) == "critical"

    # Verify all five Literal values are covered
    results = {
        classify_risk_level(35.0, 60.0),  # none
        classify_risk_level(26.0, 41.0),  # low
        classify_risk_level(22.0, 42.0),  # medium
        classify_risk_level(15.0, 40.0),  # high
        classify_risk_level(5.0, 10.0),  # critical
    }
    assert results == {"none", "low", "medium", "high", "critical"}


def test_classify_risk_level_respects_custom_thresholds() -> None:
    assert (
        classify_risk_level(
            25.0,
            45.0,
            tcpa_threshold_seconds=60.0,
            cpa_threshold_m=100.0,
        )
        == "low"  # 25≥25 AND 45≥40 with wider alert envelope
    )


def test_risk_thresholds_dataclass_rejects_invalid_values() -> None:
    """Review item M4: RiskThresholds validates inputs."""
    from edge.risk.risk_engine import RiskThresholds

    # Valid construction
    t = RiskThresholds(
        tcpa_alert=30.0,
        cpa_alert=50.0,
        tcpa_critical=5.0,
        cpa_critical=10.0,
        tcpa_high=15.0,
        cpa_high=25.0,
        tcpa_low=22.0,
        cpa_low=38.0,
    )
    assert t.tcpa_critical == 5.0

    # Negative values rejected
    import pytest

    with pytest.raises(ValueError):
        RiskThresholds(tcpa_alert=-1.0)

    with pytest.raises(ValueError):
        RiskThresholds(cpa_critical=-0.1)


def test_classify_risk_level_with_custom_risk_thresholds() -> None:
    """Review item M4: band thresholds are fully configurable."""
    from edge.risk.risk_engine import RiskThresholds

    t = RiskThresholds(
        tcpa_alert=60.0,
        cpa_alert=100.0,
        tcpa_critical=5.0,
        cpa_critical=10.0,
        tcpa_high=15.0,
        cpa_high=30.0,
        tcpa_low=40.0,
        cpa_low=70.0,
    )
    # With wider alert envelope, this should be "low"
    assert classify_risk_level(45.0, 80.0, thresholds=t) == "low"
    # Still critical with very tight values
    assert classify_risk_level(3.0, 5.0, thresholds=t) == "critical"


def test_custom_threshold_alert_has_non_none_risk_level() -> None:
    result = evaluate_risk(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(0.0, 0.0),
        target_position_m=(40.0, 80.0),
        target_velocity_mps=(-1.0, 0.0),
        tcpa_threshold_seconds=60.0,
        cpa_threshold_m=100.0,
    )

    assert result.should_alert is True
    # TCPA=40, CPA=80 with custom envelope (60/100).  40≥25 AND 80≥40 → "low"
    assert result.risk_level == "low"


def test_calculate_min_future_distance_extrapolates_ownship_motion() -> None:
    prediction = _prediction(
        _point(5.0, 20.0, 0.0),
        _point(10.0, 15.0, 0.0),
    )

    result = calculate_min_future_distance(
        own_position_m=(0.0, 0.0),
        own_velocity_mps=(1.0, 0.0),
        own_timestamp=BASE_TS,
        prediction=prediction,
    )

    assert result is not None
    assert result.time_to_min_seconds == 10.0
    assert result.distance_m == 5.0


def test_evaluate_track_risk_returns_target_risk_result() -> None:
    own_state = _state("own", 0.0, 0.0, 0.0, 0.0)
    target_state = _state(7, 100.0, 0.0, -5.0, 0.0)

    result = evaluate_track_risk(own_state, target_state)

    assert result.device_id == "edge-01"
    assert result.track_id == 7
    assert result.should_alert is True


def test_evaluate_track_risk_rejects_prediction_for_different_track() -> None:
    own_state = _state("own", 0.0, 0.0, 0.0, 0.0)
    target_state = _state(8, 100.0, 0.0, -5.0, 0.0)
    prediction = _prediction(_point(10.0, 40.0, 0.0))

    with pytest.raises(ValueError, match="track_id"):
        evaluate_track_risk(own_state, target_state, prediction)
