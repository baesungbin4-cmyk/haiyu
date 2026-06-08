"""FastAPI route registration for the cloud backend."""

from __future__ import annotations

import base64
import binascii
import hmac
from typing import Any

from cloud.app.config import CloudSettings
from cloud.app.mqtt.consumer import EdgeMqttConsumer
from cloud.app.schemas import HealthResponse

try:  # pragma: no cover - FastAPI not installed in this environment
    from fastapi import (
        APIRouter,
        Body,
        Depends,
        Header,
        HTTPException,
        Query,
        WebSocket,
        WebSocketDisconnect,
    )
except ModuleNotFoundError:  # pragma: no cover
    APIRouter = None
    Body = None
    Depends = None
    Header = None
    HTTPException = Exception
    Query = None
    WebSocket = Any
    WebSocketDisconnect = Exception

WEBSOCKET_SUBPROTOCOL = "haiyu.realtime.v1"
_WEBSOCKET_TOKEN_PREFIX = "bearer."


def require_api_token(
    settings: CloudSettings,
    authorization: str | None,
) -> None:
    if _auth_is_disabled(settings):
        return
    if not _is_valid_bearer_token(settings, authorization):
        raise PermissionError("invalid API token")


def _configured_tokens(settings: CloudSettings) -> tuple[str, ...]:
    tokens = []
    if settings.api_token is not None:
        tokens.append(settings.api_token)
    tokens.extend(settings.api_tokens)
    return tuple(dict.fromkeys(tokens))


def _auth_is_disabled(settings: CloudSettings) -> bool:
    return not _configured_tokens(settings) and settings.allow_unauthenticated


def _is_valid_bearer_token(
    settings: CloudSettings,
    authorization: str | None,
) -> bool:
    if _auth_is_disabled(settings):
        return True
    for token in _configured_tokens(settings):
        expected = f"Bearer {token}"
        if hmac.compare_digest(authorization or "", expected):
            return True
    return False


def _is_valid_websocket_token(
    settings: CloudSettings,
    authorization: str | None,
    token: str | None,
    subprotocol_header: str | None = None,
) -> bool:
    if _auth_is_disabled(settings):
        return True
    header_ok = _is_valid_bearer_token(settings, authorization)
    protocol_ok = _is_valid_websocket_subprotocol_token(
        settings,
        subprotocol_header,
    )
    allow_query_token = settings.websocket_allow_query_token
    query_ok = allow_query_token and _token_matches(settings, token)
    return header_ok or protocol_ok or query_ok


def _token_matches(settings: CloudSettings, token: str | None) -> bool:
    return any(
        hmac.compare_digest(token or "", expected)
        for expected in _configured_tokens(settings)
    )


def _is_valid_websocket_subprotocol_token(
    settings: CloudSettings,
    subprotocol_header: str | None,
) -> bool:
    for item in _websocket_subprotocol_items(subprotocol_header):
        if not item.startswith(_WEBSOCKET_TOKEN_PREFIX):
            continue
        prefix_length = len(_WEBSOCKET_TOKEN_PREFIX)
        encoded_token = item[prefix_length:]
        decoded_token = _decode_websocket_protocol_token(encoded_token)
        if _token_matches(settings, decoded_token):
            return True
    return False


def _decode_websocket_protocol_token(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{value}{padding}".encode("ascii"))
    except (binascii.Error, ValueError):
        return value
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        return value


def _websocket_subprotocol_items(header: str | None) -> tuple[str, ...]:
    if header is None:
        return ()
    return tuple(item.strip() for item in header.split(",") if item.strip())


def _selected_websocket_subprotocol(header: str | None) -> str | None:
    if WEBSOCKET_SUBPROTOCOL in _websocket_subprotocol_items(header):
        return WEBSOCKET_SUBPROTOCOL
    return None


def _is_allowed_websocket_origin(
    settings: CloudSettings,
    origin: str | None,
) -> bool:
    if origin is None:
        return True
    return origin in settings.cors_origins


def _events(consumer: EdgeMqttConsumer) -> tuple[Any, ...]:
    return tuple(getattr(consumer.writer, "written_events", ()))


def _latest_device_rows(consumer: EdgeMqttConsumer) -> dict[str, dict[str, Any]]:
    devices: dict[str, dict[str, Any]] = {}
    for event in _events(consumer):
        device = devices.setdefault(
            event.device_id,
            {
                "device_id": event.device_id,
                "online": None,
                "last_seen": None,
                "last_event_type": None,
                "cpu": None,
                "mem": None,
                "npu": None,
                "source": "mqtt_ingestion",
            },
        )
        if device["last_seen"] is None or event.timestamp >= device["last_seen"]:
            device["last_seen"] = event.timestamp
            device["last_event_type"] = event.event_type
        if event.event_type.lower() in {"status", "lwt"}:
            payload = event.payload
            device["online"] = _online_from_payload(event.event_type, payload)
            for key in ("cpu", "mem", "npu"):
                if key in payload:
                    device[key] = payload[key]
    return devices


def _online_from_payload(event_type: str, payload: Any) -> bool:
    value = payload.get("online")
    if value is None:
        value = payload.get("status", payload.get("state"))
    if value is None:
        return event_type.lower() != "lwt"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"offline", "down", "false", "0", "no"}:
        return False
    if normalized in {"online", "up", "true", "1", "yes"}:
        return True
    return bool(value)


def _risk_counts(consumer: EdgeMqttConsumer) -> dict[str, int]:
    counts: dict[str, int] = {}
    for alert in consumer.latest_alerts:
        counts[alert.risk_level] = counts.get(alert.risk_level, 0) + 1
    return counts


def _summary_payload(consumer: EdgeMqttConsumer) -> dict[str, Any]:
    devices = _latest_device_rows(consumer)
    online_count = sum(1 for device in devices.values() if device["online"])
    tracks = [
        event for event in _events(consumer) if event.event_type.lower() == "track"
    ]
    sensors = [
        event for event in _events(consumer) if event.event_type.lower() == "sensor"
    ]
    return {
        "device_count": len(devices),
        "online_device_count": online_count,
        "latest_alert_count": len(consumer.latest_alerts),
        "track_event_count": len(tracks),
        "sensor_event_count": len(sensors),
        "risk_counts": _risk_counts(consumer),
        "source": "mqtt_ingestion",
        "prototype_notice": "No real-port validation is claimed.",
    }


def _calibration_ack_payload(
    device_id: str,
    payload: Any,
) -> dict[str, Any]:
    if isinstance(payload, list):
        control_point_count = len(payload)
    elif isinstance(payload, dict) and isinstance(
        payload.get("control_points"),
        list,
    ):
        control_point_count = len(payload["control_points"])
    else:
        control_point_count = 0
    return {
        "device_id": device_id,
        "accepted": True,
        "control_point_count": control_point_count,
        "status": "queued_for_edge_execution",
        "source": "rest_control_plane",
        "prototype_notice": (
            "Calibration execution/result is not claimed until " "edge reports it."
        ),
    }


def create_router(settings: CloudSettings, consumer: EdgeMqttConsumer) -> Any:
    if APIRouter is None:
        raise RuntimeError("fastapi is required to create API routes")

    router = APIRouter()

    def auth_dependency(
        authorization: str | None = Header(default=None),
    ) -> None:
        try:
            require_api_token(settings, authorization)
        except PermissionError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @router.get("/health")
    def health() -> dict[str, str]:
        auth_warning = None
        if not _configured_tokens(settings) and settings.allow_unauthenticated:
            auth_warning = (
                "API token is not configured; auth is a disabled "
                "non-production placeholder."
            )
        elif not _configured_tokens(settings):
            reject_message = (
                "API token is not configured; " "protected endpoints reject."
            )
            auth_warning = reject_message
        return HealthResponse(auth_warning=auth_warning).to_dict()

    @router.get(
        "/api/v1/alerts/latest",
        dependencies=[Depends(auth_dependency)],
    )
    def latest_alerts() -> list[dict[str, Any]]:
        return [alert.to_dict() for alert in consumer.latest_alerts]

    @router.get(
        "/api/v1/alerts",
        dependencies=[Depends(auth_dependency)],
    )
    def alert_history(
        from_: str = Query(default="-1h", alias="from"),
        to: str | None = Query(default=None),
        level: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        try:
            return consumer.writer.query_alerts(
                start=from_,
                stop=to,
                level=level,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/api/v1/alarms",
        dependencies=[Depends(auth_dependency)],
    )
    def alarm_history(
        from_: str = Query(default="-1h", alias="from"),
        to: str | None = Query(default=None),
        level: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        return alert_history(from_=from_, to=to, level=level, limit=limit)

    @router.get(
        "/api/v1/devices",
        dependencies=[Depends(auth_dependency)],
    )
    def devices() -> list[dict[str, Any]]:
        return list(_latest_device_rows(consumer).values())

    @router.get(
        "/api/v1/devices/{device_id}",
        dependencies=[Depends(auth_dependency)],
    )
    def device_detail(device_id: str) -> dict[str, Any]:
        device = _latest_device_rows(consumer).get(device_id)
        if device is None:
            raise HTTPException(status_code=404, detail="device not found")
        return device

    @router.get(
        "/api/v1/tracks",
        dependencies=[Depends(auth_dependency)],
    )
    def track_history(
        device_id: str | None = Query(default=None, alias="deviceId"),
        from_: str = Query(default="-1h", alias="from"),
        to: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        try:
            return consumer.writer.query_tracks(
                device_id=device_id,
                start=from_,
                stop=to,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/api/v1/sensors",
        dependencies=[Depends(auth_dependency)],
    )
    def sensor_history(
        device_id: str | None = Query(default=None, alias="deviceId"),
        sensor_type: str | None = Query(default=None, alias="type"),
        from_: str = Query(default="-1h", alias="from"),
        to: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        try:
            return consumer.writer.query_sensors(
                device_id=device_id,
                sensor_type=sensor_type,
                start=from_,
                stop=to,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get(
        "/api/v1/stats/summary",
        dependencies=[Depends(auth_dependency)],
    )
    def stats_summary() -> dict[str, Any]:
        return _summary_payload(consumer)

    @router.post(
        "/api/v1/calib/{device_id}",
        dependencies=[Depends(auth_dependency)],
    )
    def submit_calibration(
        device_id: str,
        payload: Any = Body(default=None),
    ) -> dict[str, Any]:
        return _calibration_ack_payload(device_id, payload)

    @router.websocket("/ws/realtime")
    async def websocket_realtime(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if not _is_allowed_websocket_origin(settings, origin):
            await websocket.close(code=1008)
            return
        authorization = websocket.headers.get("authorization")
        token = websocket.query_params.get("token")
        protocol_header = websocket.headers.get("sec-websocket-protocol")
        token_is_valid = _is_valid_websocket_token(
            settings,
            authorization,
            token,
            protocol_header,
        )
        if not token_is_valid:
            await websocket.close(code=1008)
            return
        selected_subprotocol = _selected_websocket_subprotocol(protocol_header)
        connected = await consumer.websocket_manager.connect(
            websocket,
            max_connections=settings.websocket_max_connections,
            subprotocol=selected_subprotocol,
        )
        if not connected:
            await websocket.close(code=1013)
            return
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            consumer.websocket_manager.disconnect(websocket)

    return router


__all__ = ["create_router", "require_api_token"]
