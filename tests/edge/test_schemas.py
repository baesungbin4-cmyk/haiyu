import json

import pytest

from edge.common.schemas import (
    Detection,
    MqttEventEnvelope,
    RiskResult,
    TrackHistory,
    TrackState,
    TrajectoryPoint,
    TrajectoryPrediction,
)

BASE_TS = 1_718_000_000.0


def _track_state(track_id=7):
    return TrackState(
        device_id="edge-01",
        track_id=track_id,
        timestamp=BASE_TS,
        bbox=(10, 20, 40, 60),
        confidence=0.91,
        class_name="vessel",
        pixel_coordinate=(25, 40),
        velocity=(1.2, -0.4),
        world_coordinate=(120.5, 31.2),
    )


def test_detection_schema_serializes_pixel_and_world_coordinates():
    detection = Detection(
        device_id="edge-01",
        timestamp=BASE_TS,
        bbox=(10, 20, 40, 60),
        confidence=0.88,
        class_name="vessel",
        pixel_coordinate=(25, 40),
        world_coordinate=None,
    )

    payload = detection.to_dict()

    assert payload["device_id"] == "edge-01"
    assert payload["bbox"] == [10.0, 20.0, 40.0, 60.0]
    assert payload["pixel_coordinate"] == [25.0, 40.0]
    assert payload["world_coordinate"] is None
    json.dumps(payload)


def test_detection_rejects_invalid_confidence_and_bbox_shape():
    with pytest.raises(ValueError, match="confidence"):
        Detection(
            device_id="edge-01",
            timestamp=BASE_TS,
            bbox=(10, 20, 40, 60),
            confidence=1.1,
            class_name="vessel",
            pixel_coordinate=(25, 40),
        )

    with pytest.raises(ValueError, match="bbox"):
        Detection(
            device_id="edge-01",
            timestamp=BASE_TS,
            bbox=(10, 20, 40),
            confidence=0.7,
            class_name="vessel",
            pixel_coordinate=(25, 40),
        )


def test_track_history_groups_consistent_track_states():
    state = _track_state()

    history = TrackHistory(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        states=[state],
    )

    assert history.states == (state,)
    assert history.to_dict()["states"][0]["velocity"] == [1.2, -0.4]


def test_track_history_rejects_mismatched_track_id():
    state = _track_state(track_id=7)

    with pytest.raises(ValueError, match="track_id"):
        TrackHistory(
            device_id="edge-01",
            track_id=8,
            timestamp=BASE_TS,
            states=[state],
        )


def test_trajectory_prediction_contains_future_points():
    point = TrajectoryPoint(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS + 1,
        pixel_coordinate=(27, 41),
        velocity=(1.0, 0.2),
        world_coordinate=(121.0, 31.4),
    )

    prediction = TrajectoryPrediction(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        points=[point],
        horizon_seconds=5,
        model_name="lstm-causal-attention",
    )

    payload = prediction.to_dict()

    assert payload["points"][0]["world_coordinate"] == [121.0, 31.4]
    assert payload["horizon_seconds"] == 5.0
    json.dumps(payload)


def test_trajectory_prediction_rejects_mismatched_point():
    point = TrajectoryPoint(
        device_id="edge-01",
        track_id=8,
        timestamp=BASE_TS + 1,
        pixel_coordinate=(27, 41),
        velocity=(1.0, 0.2),
    )

    with pytest.raises(ValueError, match="track_id"):
        TrajectoryPrediction(
            device_id="edge-01",
            track_id=7,
            timestamp=BASE_TS,
            points=[point],
        )


def test_risk_result_models_alert_decision_without_risk_algorithm():
    result = RiskResult(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        risk_level="high",
        should_alert=True,
        tcpa_seconds=12,
        cpa_distance_m=35,
        message="approach warning",
    )

    payload = result.to_dict()

    assert payload["risk_level"] == "high"
    assert payload["should_alert"] is True
    assert payload["tcpa_seconds"] == 12.0
    json.dumps(payload)


def test_risk_result_validates_level_and_alert_flag():
    with pytest.raises(ValueError, match="risk_level"):
        RiskResult(
            device_id="edge-01",
            track_id=7,
            timestamp=BASE_TS,
            risk_level="urgent",
            should_alert=True,
        )

    with pytest.raises(TypeError, match="should_alert"):
        RiskResult(
            device_id="edge-01",
            track_id=7,
            timestamp=BASE_TS,
            risk_level="low",
            should_alert=1,
        )


def test_mqtt_event_envelope_wraps_json_ready_payload():
    risk = RiskResult(
        device_id="edge-01",
        track_id=7,
        timestamp=BASE_TS,
        risk_level="medium",
        should_alert=False,
    )
    source_payload = risk.to_dict()

    event = MqttEventEnvelope(
        device_id="edge-01",
        timestamp=BASE_TS,
        seq=3,
        event_type="alarm",
        topic="haiyu/edge-01/alarm",
        payload=source_payload,
        qos=1,
        retain=False,
    )
    source_payload["risk_level"] = "low"

    payload = event.to_dict()

    assert payload["payload"]["risk_level"] == "medium"
    assert payload["qos"] == 1
    json.dumps(payload)


def test_mqtt_event_envelope_validates_qos_and_sequence():
    with pytest.raises(ValueError, match="qos"):
        MqttEventEnvelope(
            device_id="edge-01",
            timestamp=BASE_TS,
            seq=3,
            event_type="track",
            topic="haiyu/edge-01/track",
            payload={},
            qos=3,
        )

    with pytest.raises(ValueError, match="seq"):
        MqttEventEnvelope(
            device_id="edge-01",
            timestamp=BASE_TS,
            seq=-1,
            event_type="track",
            topic="haiyu/edge-01/track",
            payload={},
        )
