"""Cloud-side schemas for edge events and API responses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EdgeEvent:
    device_id: str
    timestamp: float
    seq: int
    event_type: str
    topic: str
    payload: Mapping[str, Any]
    qos: int = 1
    retain: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EdgeEvent":
        required = (
            "device_id",
            "timestamp",
            "seq",
            "event_type",
            "topic",
            "payload",
        )
        for field in required:
            if field not in value:
                raise ValueError(f"missing edge event field: {field}")
        event = cls(
            device_id=str(value["device_id"]),
            timestamp=float(value["timestamp"]),
            seq=int(value["seq"]),
            event_type=str(value["event_type"]),
            topic=str(value["topic"]),
            payload=dict(value["payload"]),
            qos=int(value.get("qos", 1)),
            retain=bool(value.get("retain", False)),
        )
        event.validate()
        return event

    def validate(self) -> None:
        if not self.device_id.strip():
            raise ValueError("device_id must be non-empty")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if self.seq < 0:
            raise ValueError("seq must be non-negative")
        if not self.event_type.strip():
            raise ValueError("event_type must be non-empty")
        if not self.topic.strip():
            raise ValueError("topic must be non-empty")
        if self.qos not in (0, 1, 2):
            raise ValueError("qos must be 0, 1, or 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "seq": self.seq,
            "event_type": self.event_type,
            "topic": self.topic,
            "payload": dict(self.payload),
            "qos": self.qos,
            "retain": self.retain,
        }


@dataclass(frozen=True)
class AlertView:
    device_id: str
    timestamp: float
    seq: int
    risk_level: str
    should_alert: bool
    message: str | None = None

    @classmethod
    def from_event(cls, event: EdgeEvent) -> "AlertView":
        payload = event.payload
        return cls(
            device_id=event.device_id,
            timestamp=event.timestamp,
            seq=event.seq,
            risk_level=str(payload.get("risk_level", "none")),
            should_alert=bool(payload.get("should_alert", False)),
            message=payload.get("message"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "seq": self.seq,
            "risk_level": self.risk_level,
            "should_alert": self.should_alert,
            "message": self.message,
        }


@dataclass(frozen=True)
class HealthResponse:
    status: str = "ok"
    auth_warning: str | None = None

    def to_dict(self) -> dict[str, str]:
        response = {"status": self.status}
        if self.auth_warning is not None:
            response["auth_warning"] = self.auth_warning
        return response


__all__ = ["AlertView", "EdgeEvent", "HealthResponse"]
