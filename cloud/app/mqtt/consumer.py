"""MQTT consumer interface for edge events."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from cloud.app.config import DEFAULT_MQTT_TOPICS
from cloud.app.db.influx import InfluxWriter
from cloud.app.schemas import AlertView, EdgeEvent
from cloud.app.websocket.manager import WebSocketManager


@dataclass(frozen=True)
class MqttConsumerConfig:
    host: str = "localhost"
    port: int = 1883
    client_id: str = "haiyu-cloud-consumer"
    topics: tuple[str, ...] = DEFAULT_MQTT_TOPICS
    qos: int = 1
    username: str | None = None
    password: str | None = None
    require_credentials: bool = True
    event_hmac_secret: str | None = None
    allowed_device_ids: tuple[str, ...] = ()
    keepalive: int = 60
    tls_enabled: bool = False
    tls_ca_cert: str | None = None
    tls_client_cert: str | None = None
    tls_client_key: str | None = None
    tls_insecure: bool = False
    reconnect_min_delay_seconds: float = 1.0
    reconnect_max_delay_seconds: float = 30.0
    reconnect_max_attempts: int | None = None
    auto_reconnect: bool = True

    @classmethod
    def from_settings(cls, settings: Any) -> "MqttConsumerConfig":
        return cls(
            host=settings.mqtt_host,
            port=settings.mqtt_port,
            client_id=settings.mqtt_client_id,
            topics=settings.mqtt_topics,
            username=settings.mqtt_username,
            password=settings.mqtt_password,
            require_credentials=settings.mqtt_require_credentials,
            event_hmac_secret=settings.mqtt_event_hmac_secret,
            allowed_device_ids=settings.mqtt_allowed_device_ids,
            keepalive=settings.mqtt_keepalive,
            tls_enabled=settings.mqtt_tls_enabled,
            tls_ca_cert=settings.mqtt_tls_ca_cert,
            tls_client_cert=settings.mqtt_tls_client_cert,
            tls_client_key=settings.mqtt_tls_client_key,
            tls_insecure=settings.mqtt_tls_insecure,
            reconnect_max_attempts=settings.mqtt_reconnect_max_attempts,
            auto_reconnect=settings.mqtt_auto_reconnect,
        )


class EdgeMqttConsumer:
    def __init__(
        self,
        *,
        writer: InfluxWriter,
        websocket_manager: WebSocketManager,
        mqtt_config: MqttConsumerConfig | None = None,
        mqtt_client: Any | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_alerts: int = 100,
        max_seen_events: int = 10_000,
    ) -> None:
        self.writer = writer
        self.websocket_manager = websocket_manager
        self.mqtt_config = mqtt_config or MqttConsumerConfig()
        self.mqtt_client = mqtt_client
        self.event_loop = event_loop
        self.sleep = sleep
        self.max_alerts = max_alerts
        self.max_seen_events = max_seen_events
        self.latest_alerts: list[AlertView] = []
        self._seen_event_keys: set[tuple[str, int]] = set()
        self._seen_event_order: deque[tuple[str, int]] = deque()
        self._reconnect_lock = threading.Lock()
        self._reconnect_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        if self.mqtt_client is not None:
            self._bind_client_callbacks(self.mqtt_client)

    async def accept_event(
        self,
        raw_event: Mapping[str, Any] | str,
    ) -> EdgeEvent | None:
        event = self._coerce_event(raw_event)
        return await self._accept_edge_event(event)

    async def accept_mqtt_message(
        self,
        raw_payload: Mapping[str, Any] | str,
        *,
        topic: str,
        qos: int = 1,
        retain: bool = False,
    ) -> EdgeEvent | None:
        event = self._coerce_event(
            raw_payload,
            topic=topic,
            qos=qos,
            retain=retain,
        )
        return await self._accept_edge_event(event)

    async def _accept_edge_event(self, event: EdgeEvent) -> EdgeEvent | None:
        self._validate_event_source(event)
        if self._is_duplicate(event):
            return None
        self._remember_event(event)
        self.writer.write_event(event)
        if event.event_type == "alarm":
            alert = AlertView.from_event(event)
            self._append_latest_alert(alert)
            await self.websocket_manager.broadcast(
                {"type": "alert", "data": alert.to_dict()}
            )
        elif event.event_type == "track":
            await self.websocket_manager.broadcast(
                {"type": "track", "data": self._event_realtime_data(event)}
            )
        elif event.event_type in {"status", "lwt", "sensor"}:
            await self.websocket_manager.broadcast(
                {
                    "type": event.event_type,
                    "data": self._event_realtime_data(event),
                }
            )
        return event

    def connect(self) -> Any:
        client = self._ensure_client()
        self._configure_security(client)
        client.connect(
            self.mqtt_config.host,
            self.mqtt_config.port,
            self.mqtt_config.keepalive,
        )
        return client

    def _configure_security(self, client: Any) -> None:
        credentials_required = self.mqtt_config.require_credentials
        if credentials_required and not self.mqtt_config.username:
            raise ValueError("MQTT username is required")
        if credentials_required and not self.mqtt_config.password:
            raise ValueError("MQTT password is required")
        if self.mqtt_config.username is not None:
            client.username_pw_set(
                self.mqtt_config.username,
                self.mqtt_config.password,
            )
        if self.mqtt_config.tls_enabled:
            tls_set = getattr(client, "tls_set", None)
            if tls_set is None:
                raise RuntimeError("MQTT client does not support TLS")
            tls_set(
                ca_certs=self.mqtt_config.tls_ca_cert,
                certfile=self.mqtt_config.tls_client_cert,
                keyfile=self.mqtt_config.tls_client_key,
            )
            tls_insecure_set = getattr(client, "tls_insecure_set", None)
            if tls_insecure_set is not None:
                tls_insecure_set(self.mqtt_config.tls_insecure)

    def _validate_event_source(self, event: EdgeEvent) -> None:
        allowed = self.mqtt_config.allowed_device_ids
        if allowed and event.device_id not in allowed:
            raise ValueError("MQTT event device is not allowed")
        secret = self.mqtt_config.event_hmac_secret
        if secret is None:
            return
        signature = _event_signature_value(event)
        if signature is None:
            raise ValueError("MQTT event signature is required")
        expected = _sign_event(event, secret)
        if not hmac.compare_digest(signature, expected):
            raise ValueError("MQTT event signature is invalid")

    def start(self) -> Any:
        self._stop_event.clear()
        client = self.connect()
        loop_start = getattr(client, "loop_start", None)
        if loop_start is not None:
            loop_start()
        return client

    def stop(self) -> None:
        self._stop_event.set()
        self.event_loop = None
        if self.mqtt_client is None:
            self._join_reconnect_thread()
            return
        loop_stop = getattr(self.mqtt_client, "loop_stop", None)
        if loop_stop is not None:
            loop_stop()
        disconnect = getattr(self.mqtt_client, "disconnect", None)
        if disconnect is not None:
            disconnect()
        self._join_reconnect_thread()

    def reconnect_with_backoff(self, max_attempts: int | None = None) -> Any:
        client = self._ensure_client()
        delay = self.mqtt_config.reconnect_min_delay_seconds
        attempts = 0
        while not self._stop_event.is_set() and (
            max_attempts is None or attempts < max_attempts
        ):
            try:
                reconnect = getattr(client, "reconnect", None)
                if reconnect is None:
                    client.connect(
                        self.mqtt_config.host,
                        self.mqtt_config.port,
                        self.mqtt_config.keepalive,
                    )
                else:
                    reconnect()
                self.subscribe_topics(client)
                return client
            except Exception:
                attempts += 1
                if max_attempts is not None and attempts >= max_attempts:
                    raise
                if self._wait_before_reconnect(delay):
                    return client
                delay = min(
                    delay * 2,
                    self.mqtt_config.reconnect_max_delay_seconds,
                )
        return client

    def subscribe_topics(self, client: Any | None = None) -> None:
        mqtt_client = client or self._ensure_client()
        for topic in self.mqtt_config.topics:
            mqtt_client.subscribe(topic, qos=self.mqtt_config.qos)

    def _ensure_client(self) -> Any:
        if self.mqtt_client is not None:
            return self.mqtt_client
        try:
            import paho.mqtt.client as mqtt
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("paho-mqtt is required") from exc
        self.mqtt_client = mqtt.Client(client_id=self.mqtt_config.client_id)
        self._bind_client_callbacks(self.mqtt_client)
        return self.mqtt_client

    def _bind_client_callbacks(self, client: Any) -> None:
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message

    def _on_connect(
        self,
        client: Any,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any | None = None,
    ) -> None:
        if _is_success_reason(reason_code):
            self.subscribe_topics(client)

    def _on_disconnect(self, client: Any, userdata: Any, *args: Any) -> None:
        reason_code = _disconnect_reason(args)
        auto_reconnect = self.mqtt_config.auto_reconnect
        disconnected_by_error = not _is_success_reason(reason_code)
        should_reconnect = auto_reconnect and disconnected_by_error
        if should_reconnect:
            self.schedule_reconnect()

    def schedule_reconnect(self) -> None:
        if self._stop_event.is_set():
            return
        with self._reconnect_lock:
            thread = self._reconnect_thread
            if thread is not None and thread.is_alive():
                return
            self._reconnect_thread = threading.Thread(
                target=self._run_reconnect_with_config,
                name="haiyu-mqtt-reconnect",
                daemon=True,
            )
            self._reconnect_thread.start()

    def _run_reconnect_with_config(self) -> None:
        self.reconnect_with_backoff(
            max_attempts=self.mqtt_config.reconnect_max_attempts
        )

    def _wait_before_reconnect(self, delay: float) -> bool:
        if self.sleep is time.sleep:
            return self._stop_event.wait(delay)
        self.sleep(delay)
        return self._stop_event.is_set()

    def _join_reconnect_thread(self) -> None:
        thread = self._reconnect_thread
        if thread is None or thread is threading.current_thread():
            return
        if thread.is_alive():
            thread.join(timeout=1.0)

    def _on_message(self, client: Any, userdata: Any, message: Any) -> Any:
        payload = message.payload
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        try:
            return self._submit_event(
                payload,
                topic=message.topic,
                qos=getattr(message, "qos", self.mqtt_config.qos),
                retain=getattr(message, "retain", False),
            )
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to handle MQTT message on topic %s",
                message.topic,
                exc_info=True,
            )
            return None

    def _submit_event(
        self,
        raw_payload: Mapping[str, Any] | str,
        *,
        topic: str,
        qos: int,
        retain: bool,
    ) -> Any:
        if self._stop_event.is_set():
            return None
        coroutine = self.accept_mqtt_message(
            raw_payload,
            topic=topic,
            qos=qos,
            retain=retain,
        )
        if self.event_loop is not None and self.event_loop.is_running():
            return asyncio.run_coroutine_threadsafe(coroutine, self.event_loop)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        return loop.create_task(coroutine)

    def _coerce_event(
        self,
        raw_event: Mapping[str, Any] | str,
        *,
        topic: str | None = None,
        qos: int | None = None,
        retain: bool | None = None,
    ) -> EdgeEvent:
        if isinstance(raw_event, str):
            try:
                raw_event = json.loads(raw_event)
            except json.JSONDecodeError:
                raw_event = {"_raw": raw_event}
        if not isinstance(raw_event, Mapping):
            raise TypeError("MQTT payload must decode to a JSON object")
        if _has_edge_event_fields(raw_event) and topic is None:
            return EdgeEvent.from_dict(raw_event)

        event_topic = str(raw_event.get("topic") or topic or "")
        payload = raw_event.get("payload")
        if not isinstance(payload, Mapping):
            payload = _payload_without_envelope(raw_event)
        event_qos = qos
        if event_qos is None:
            event_qos = raw_event.get("qos", self.mqtt_config.qos)
        event_retain = retain
        if event_retain is None:
            event_retain = raw_event.get("retain", False)
        event = {
            "device_id": raw_event.get("device_id")
            or raw_event.get("deviceId")
            or _device_id_from_topic(event_topic),
            "timestamp": raw_event.get("timestamp")
            or raw_event.get("ts")
            or time.time(),
            "seq": raw_event.get("seq", 0),
            "event_type": raw_event.get("event_type")
            or raw_event.get("eventType")
            or raw_event.get("type")
            or _event_type_from_topic(event_topic),
            "topic": event_topic,
            "payload": dict(payload),
            "qos": event_qos,
            "retain": event_retain,
        }
        return EdgeEvent.from_dict(event)

    def _is_duplicate(self, event: EdgeEvent) -> bool:
        return (event.device_id, event.seq) in self._seen_event_keys

    def _remember_event(self, event: EdgeEvent) -> None:
        key = (event.device_id, event.seq)
        self._seen_event_keys.add(key)
        self._seen_event_order.append(key)
        while len(self._seen_event_order) > self.max_seen_events:
            old_key = self._seen_event_order.popleft()
            self._seen_event_keys.discard(old_key)

    def _append_latest_alert(self, alert: AlertView) -> None:
        if self.max_alerts <= 0:
            self.latest_alerts.clear()
            return
        self.latest_alerts.append(alert)
        if len(self.latest_alerts) > self.max_alerts:
            del self.latest_alerts[: len(self.latest_alerts) - self.max_alerts]

    def _event_realtime_data(self, event: EdgeEvent) -> dict[str, Any]:
        data = dict(event.payload)
        data.update(
            {
                "device_id": event.device_id,
                "timestamp": event.timestamp,
                "seq": event.seq,
            }
        )
        return data


def _has_edge_event_fields(value: Mapping[str, Any]) -> bool:
    return all(
        field in value
        for field in (
            "device_id",
            "timestamp",
            "seq",
            "event_type",
            "topic",
            "payload",
        )
    )


def _payload_without_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    envelope_keys = {
        "device_id",
        "deviceId",
        "timestamp",
        "ts",
        "seq",
        "event_type",
        "eventType",
        "type",
        "topic",
        "payload",
        "qos",
        "retain",
    }
    payload = {}
    for key, item in value.items():
        if key not in envelope_keys:
            payload[key] = item
    return payload


def _event_signature_value(event: EdgeEvent) -> str | None:
    for field in ("signature", "hmac", "hmac_sha256"):
        value = event.payload.get(field)
        if value is not None:
            return str(value)
    return None


def _sign_event(event: EdgeEvent, secret: str) -> str:
    payload = {
        key: value
        for key, value in event.payload.items()
        if key not in {"signature", "hmac", "hmac_sha256"}
    }
    signing_value = {
        "device_id": event.device_id,
        "timestamp": event.timestamp,
        "seq": event.seq,
        "event_type": event.event_type,
        "topic": event.topic,
        "payload": payload,
    }
    message = json.dumps(
        signing_value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(
        secret.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()


def _device_id_from_topic(topic: str) -> str:
    parts = topic.split("/")
    if len(parts) >= 3 and parts[0] == "haiyu":
        return parts[1]
    return "unknown"


def _event_type_from_topic(topic: str) -> str:
    parts = topic.split("/")
    if len(parts) >= 3 and parts[0] == "haiyu":
        return parts[-1]
    return "unknown"


def _is_success_reason(reason_code: Any) -> bool:
    if reason_code is None:
        return True
    if isinstance(reason_code, int):
        return reason_code == 0
    value = getattr(reason_code, "value", None)
    if isinstance(value, int):
        return value == 0
    text = str(reason_code).strip().lower()
    return text in {"0", "success", "normal disconnection"}


def _disconnect_reason(args: tuple[Any, ...]) -> Any:
    if not args:
        return None
    for value in reversed(args):
        if value is not None:
            return value
    return None


__all__ = ["EdgeMqttConsumer", "MqttConsumerConfig", "_sign_event"]
