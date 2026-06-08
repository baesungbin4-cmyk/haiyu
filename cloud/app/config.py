"""Cloud backend configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MQTT_TOPICS = (
    "haiyu/+/status",
    "haiyu/+/track",
    "haiyu/+/alarm",
    "haiyu/+/sensor",
    "haiyu/+/lwt",
)


@dataclass(frozen=True)
class CloudSettings:
    """Runtime settings for the cloud backend skeleton.

    Authentication status: token auth protects API and WebSocket endpoints by
    default. It remains a development-oriented control, not production IAM.
    """

    api_token: str | None = None
    api_tokens: tuple[str, ...] = ()
    allow_unauthenticated: bool = False
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    influxdb_url: str = "http://localhost:8086"
    influxdb_org: str = "haiyu"
    influxdb_bucket: str = "haiyu"
    influxdb_token: str | None = None
    mqtt_enabled: bool = False
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_client_id: str = "haiyu-cloud-consumer"
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    mqtt_require_credentials: bool = True
    mqtt_event_hmac_secret: str | None = None
    mqtt_allowed_device_ids: tuple[str, ...] = ()
    mqtt_keepalive: int = 60
    mqtt_topics: tuple[str, ...] = DEFAULT_MQTT_TOPICS
    mqtt_tls_enabled: bool = False
    mqtt_tls_ca_cert: str | None = None
    mqtt_tls_client_cert: str | None = None
    mqtt_tls_client_key: str | None = None
    mqtt_tls_insecure: bool = False
    mqtt_auto_reconnect: bool = True
    mqtt_reconnect_max_attempts: int | None = None
    websocket_max_connections: int = 100
    websocket_allow_query_token: bool = False

    @property
    def mqtt_topic(self) -> str:
        """Backward-compatible comma-separated topic view."""

        return ",".join(self.mqtt_topics)


def _load_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _load_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _load_optional_string(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value


def _load_optional_strings(name: str) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _load_mqtt_topics() -> tuple[str, ...]:
    value = os.getenv("MQTT_TOPICS") or os.getenv("MQTT_TOPIC")
    if value is None:
        return DEFAULT_MQTT_TOPICS
    topics = tuple(item.strip() for item in value.split(",") if item.strip())
    return topics or DEFAULT_MQTT_TOPICS


def load_settings() -> CloudSettings:
    origins = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    origin_items = origins.split(",")
    cors_origins = tuple(item.strip() for item in origin_items if item.strip())
    reconnect_attempts = _load_optional_int("MQTT_RECONNECT_MAX_ATTEMPTS")
    api_token = _load_optional_string("API_TOKEN")
    api_tokens = _load_optional_strings("API_TOKENS")
    allowed_device_ids = _load_optional_strings("MQTT_ALLOWED_DEVICE_IDS")
    return CloudSettings(
        api_token=api_token,
        api_tokens=api_tokens,
        allow_unauthenticated=_load_bool("ALLOW_UNAUTHENTICATED"),
        cors_origins=cors_origins,
        influxdb_url=os.getenv("INFLUXDB_URL", "http://localhost:8086"),
        influxdb_org=os.getenv("INFLUXDB_ORG", "haiyu"),
        influxdb_bucket=os.getenv("INFLUXDB_BUCKET", "haiyu"),
        influxdb_token=_load_optional_string("INFLUXDB_TOKEN"),
        mqtt_enabled=_load_bool("MQTT_ENABLED"),
        mqtt_host=os.getenv("MQTT_HOST", "localhost"),
        mqtt_port=_load_int("MQTT_PORT", 1883),
        mqtt_client_id=os.getenv("MQTT_CLIENT_ID", "haiyu-cloud-consumer"),
        mqtt_username=_load_optional_string("MQTT_USERNAME"),
        mqtt_password=_load_optional_string("MQTT_PASSWORD"),
        mqtt_require_credentials=_load_bool("MQTT_REQUIRE_CREDENTIALS", True),
        mqtt_event_hmac_secret=_load_optional_string("MQTT_EVENT_HMAC_SECRET"),
        mqtt_allowed_device_ids=allowed_device_ids,
        mqtt_keepalive=_load_int("MQTT_KEEPALIVE", 60),
        mqtt_topics=_load_mqtt_topics(),
        mqtt_tls_enabled=_load_bool("MQTT_TLS_ENABLED"),
        mqtt_tls_ca_cert=_load_optional_string("MQTT_TLS_CA_CERT"),
        mqtt_tls_client_cert=_load_optional_string("MQTT_TLS_CLIENT_CERT"),
        mqtt_tls_client_key=_load_optional_string("MQTT_TLS_CLIENT_KEY"),
        mqtt_tls_insecure=_load_bool("MQTT_TLS_INSECURE"),
        mqtt_auto_reconnect=_load_bool("MQTT_AUTO_RECONNECT", True),
        mqtt_reconnect_max_attempts=reconnect_attempts,
        websocket_max_connections=_load_int("WEBSOCKET_MAX_CONNECTIONS", 100),
        websocket_allow_query_token=_load_bool("WEBSOCKET_ALLOW_QUERY_TOKEN"),
    )


__all__ = [
    "CloudSettings",
    "DEFAULT_MQTT_TOPICS",
    "load_settings",
]
