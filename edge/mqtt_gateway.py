"""MQTT v5 gateway with offline cache support.

Authentication status: username/password fields are accepted and forwarded to
the MQTT client when available, but broker ACL/JWT/API-key enforcement is not
implemented here. Unit tests use a mock client and do not require a broker.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from edge.cache.file_cache import OfflineEventCache
from edge.common.schemas import MqttEventEnvelope

try:  # pragma: no cover - optional runtime dependency
    import paho.mqtt.client as paho_mqtt
except ModuleNotFoundError:  # pragma: no cover - tested via mock client
    paho_mqtt = None


class PublishInfo(Protocol):
    rc: int


class MqttClientProtocol(Protocol):
    def connect(self, host: str, port: int, keepalive: int) -> Any: ...

    def publish(
        self,
        topic: str,
        payload: str,
        qos: int,
        retain: bool,
    ) -> PublishInfo: ...

    def subscribe(self, topic: str, qos: int = 1) -> Any: ...

    def will_set(
        self,
        topic: str,
        payload: str,
        qos: int,
        retain: bool,
    ) -> None: ...


@dataclass(frozen=True)
class MqttGatewayConfig:
    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60
    client_id: str = "haiyu-edge"
    username: str | None = None
    password: str | None = None
    backoff_initial_seconds: float = 0.1
    backoff_max_seconds: float = 5.0


class MqttGateway:
    """Publish edge events with cache-backed reconnect flushing."""

    def __init__(
        self,
        *,
        config: MqttGatewayConfig | None = None,
        cache: OfflineEventCache | None = None,
        client: MqttClientProtocol | None = None,
        sleep_fn: Any = time.sleep,
    ) -> None:
        self.config = config or MqttGatewayConfig()
        self.cache = cache or OfflineEventCache()
        self.client = client or self._make_paho_client()
        self.sleep_fn = sleep_fn
        self.connected = False
        self._backoff_seconds = self.config.backoff_initial_seconds
        self._sent_keys: set[tuple[str, int]] = set()
        self._configure_client()

    def connect(self) -> None:
        try:
            self.client.connect(
                self.config.host,
                self.config.port,
                self.config.keepalive,
            )
            self.connected = True
            self._backoff_seconds = self.config.backoff_initial_seconds
            self.flush_cache()
        except Exception:
            self.connected = False
            self._sleep_backoff()
            raise

    def publish_event(self, event: MqttEventEnvelope) -> bool:
        if not self.connected:
            self.cache.append(event)
            return False
        if not self._publish_now(event):
            self.cache.append(event)
            self.connected = False
            return False
        return True

    def subscribe(self, topic: str, qos: int = 1) -> Any:
        return self.client.subscribe(topic, qos=qos)

    def flush_cache(self) -> list[MqttEventEnvelope]:
        if not self.connected:
            return []
        sent: list[MqttEventEnvelope] = []
        for event in self.cache.read_all():
            if (event.device_id, event.seq) in self._sent_keys:
                sent.append(event)
                continue
            if not self._publish_now(event):
                break
            sent.append(event)
        if sent:
            self.cache.remove_sent(sent)
        remaining_keys = {
            (event.device_id, event.seq) for event in self.cache.read_all()
        }
        self._sent_keys.intersection_update(remaining_keys)
        return sent

    def on_disconnect(self) -> None:
        self.connected = False

    def reconnect(self) -> None:
        self.connect()

    def _publish_now(self, event: MqttEventEnvelope) -> bool:
        payload_dict = event.to_dict()
        payload = json.dumps(
            payload_dict,
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            info = self.client.publish(
                event.topic,
                payload,
                qos=event.qos,
                retain=event.retain,
            )
        except Exception:
            return False
        rc = getattr(info, "rc", 0)
        if rc != 0:
            return False
        self._sent_keys.add((event.device_id, event.seq))
        return True

    def _configure_client(self) -> None:
        lwt_topic = f"haiyu/{self.config.client_id}/lwt"
        self.client.will_set(
            lwt_topic,
            payload="offline",
            qos=1,
            retain=True,
        )
        username_pw_set = getattr(self.client, "username_pw_set", None)
        if username_pw_set is not None and self.config.username is not None:
            username_pw_set(self.config.username, self.config.password)

    def _sleep_backoff(self) -> None:
        self.sleep_fn(self._backoff_seconds)
        self._backoff_seconds = min(
            self.config.backoff_max_seconds,
            self._backoff_seconds * 2.0,
        )

    def _make_paho_client(self) -> MqttClientProtocol:
        if paho_mqtt is None:
            raise RuntimeError("paho-mqtt is required for a real MQTT client")
        return paho_mqtt.Client(
            client_id=self.config.client_id,
            protocol=paho_mqtt.MQTTv5,
        )


__all__ = ["MqttGateway", "MqttGatewayConfig", "MqttClientProtocol"]
