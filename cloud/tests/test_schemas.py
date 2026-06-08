from cloud.app.schemas import AlertView, EdgeEvent, HealthResponse


def _event_dict() -> dict:
    return {
        "device_id": "edge-01",
        "timestamp": 100.0,
        "seq": 7,
        "event_type": "alarm",
        "topic": "haiyu/edge-01/alarm",
        "payload": {
            "risk_level": "high",
            "should_alert": True,
            "message": "TCPA/CPA alert",
        },
        "qos": 1,
        "retain": False,
    }


def test_edge_event_round_trips_dict() -> None:
    event = EdgeEvent.from_dict(_event_dict())

    assert event.device_id == "edge-01"
    assert event.seq == 7
    assert event.to_dict()["payload"]["risk_level"] == "high"


def test_alert_view_extracts_payload_fields() -> None:
    event = EdgeEvent.from_dict(_event_dict())

    alert = AlertView.from_event(event)

    assert alert.risk_level == "high"
    assert alert.should_alert is True
    assert alert.to_dict()["message"] == "TCPA/CPA alert"


def test_health_response_serializes() -> None:
    assert HealthResponse().to_dict() == {"status": "ok"}
    assert HealthResponse(auth_warning="token disabled").to_dict() == {
        "status": "ok",
        "auth_warning": "token disabled",
    }


def test_edge_event_rejects_missing_required_field() -> None:
    value = _event_dict()
    value.pop("topic")

    try:
        EdgeEvent.from_dict(value)
    except ValueError as exc:
        assert "topic" in str(exc)
    else:
        raise AssertionError("missing topic should be rejected")
