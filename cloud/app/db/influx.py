"""InfluxDB write abstraction for cloud ingestion."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from cloud.app.schemas import EdgeEvent

_NANOSECONDS_PER_SECOND = 1_000_000_000
_FLUX_DURATION_PATTERN = re.compile(r"^-?\d+[smhdw]$")
_DURATION_UNITS_SECONDS = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
}
_ALLOWED_ALERT_LEVELS = frozenset(
    {
        "none",
        "low",
        "medium",
        "high",
        "critical",
    }
)


def _event_time_ns(event: EdgeEvent) -> int:
    return int(event.timestamp * _NANOSECONDS_PER_SECOND)


def _first_value(
    payload: Mapping[str, Any],
    names: tuple[str, ...],
    default: Any = None,
) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return default


def _copy_present_fields(
    payload: Mapping[str, Any],
    fields: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_name, aliases in fields.items():
        value = _first_value(payload, aliases)
        if value is not None:
            result[field_name] = value
    return result


def _without_none(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class InfluxPoint:
    """Contract-level point representation independent of client libraries."""

    measurement: str
    tags: dict[str, str]
    fields: dict[str, Any]
    time_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement": self.measurement,
            "tags": dict(self.tags),
            "fields": dict(self.fields),
            "time_ns": self.time_ns,
        }

    def to_line_protocol(self) -> str:
        if not self.fields:
            raise ValueError("InfluxDB point must contain at least one field")
        tag_part = ",".join(
            f"{_escape_key(key)}={_escape_key(value)}"
            for key, value in sorted(self.tags.items())
        )
        field_part = ",".join(
            f"{_escape_key(key)}={_format_field_value(value)}"
            for key, value in sorted(self.fields.items())
            if value is not None
        )
        measurement = _escape_key(self.measurement)
        prefix = measurement if not tag_part else f"{measurement},{tag_part}"
        return f"{prefix} {field_part} {self.time_ns}"


def _escape_key(value: Any) -> str:
    text = str(value).replace("\\", "\\\\")
    text = text.replace(" ", "\\ ")
    return text.replace(",", "\\,")


def _format_field_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value}i"
    if isinstance(value, float):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def event_to_influx_point(event: EdgeEvent) -> InfluxPoint:
    """Map edge events to the project contract's Influx measurements."""

    event_type = event.event_type.lower()
    if event_type == "alarm":
        return _alarm_event_point(event)
    if event_type == "track":
        return _target_track_point(event)
    if event_type == "sensor":
        return _sensor_reading_point(event)
    if event_type in {"status", "lwt"}:
        return _device_status_point(event)
    return _generic_event_point(event)


def _alarm_event_point(event: EdgeEvent) -> InfluxPoint:
    payload = event.payload
    cpa = _first_value(payload, ("cpa", "cpa_distance_m"))
    fields = {
        "tcpa": _first_value(payload, ("tcpa", "tcpa_seconds")),
        "cpa": cpa,
        "dist": _first_value(payload, ("dist", "distance_m"), cpa),
        "msg": str(_first_value(payload, ("msg", "message"), "")),
        "timestamp": event.timestamp,
    }
    level = str(_first_value(payload, ("level", "risk_level"), "none"))
    alarm_fields = _without_none(fields)
    return InfluxPoint(
        measurement="alarm_event",
        tags={"deviceId": event.device_id, "level": level},
        fields=alarm_fields,
        time_ns=_event_time_ns(event),
    )


def _target_track_point(event: EdgeEvent) -> InfluxPoint:
    payload = event.payload
    fields = _copy_present_fields(
        payload,
        {
            "x": ("x",),
            "y": ("y",),
            "lon": ("lon", "longitude"),
            "lat": ("lat", "latitude"),
            "cls": ("cls", "class_name", "class"),
            "conf": ("conf", "confidence"),
        },
    )
    _add_xy_from_coordinate(fields, payload)
    fields["timestamp"] = event.timestamp
    track_id = _first_value(payload, ("trackId", "track_id", "id"), "unknown")
    return InfluxPoint(
        measurement="target_track",
        tags={"deviceId": event.device_id, "trackId": str(track_id)},
        fields=fields,
        time_ns=_event_time_ns(event),
    )


def _add_xy_from_coordinate(
    fields: dict[str, Any],
    payload: Mapping[str, Any],
) -> None:
    if "x" in fields and "y" in fields:
        return
    coordinate = _first_value(
        payload,
        ("world_coordinate", "pixel_coordinate", "coordinate"),
    )
    if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
        fields.setdefault("x", coordinate[0])
        fields.setdefault("y", coordinate[1])


def _device_status_point(event: EdgeEvent) -> InfluxPoint:
    payload = event.payload
    fields = _copy_present_fields(
        payload,
        {
            "cpu": ("cpu", "cpu_usage"),
            "mem": ("mem", "memory", "memory_usage"),
            "npu": ("npu", "npu_usage"),
        },
    )
    fields["online"] = _status_online_value(event)
    fields["timestamp"] = event.timestamp
    return InfluxPoint(
        measurement="device_status",
        tags={"deviceId": event.device_id},
        fields=fields,
        time_ns=_event_time_ns(event),
    )


def _sensor_reading_point(event: EdgeEvent) -> InfluxPoint:
    payload = event.payload
    fields = {
        key: value
        for key, value in payload.items()
        if key not in {"signature", "hmac", "hmac_sha256", "sensor_type"}
        and isinstance(value, (str, int, float, bool))
    }
    fields["timestamp"] = event.timestamp
    sensor_type = str(_first_value(payload, ("sensor_type", "type"), "unknown"))
    return InfluxPoint(
        measurement="sensor_reading",
        tags={"deviceId": event.device_id, "sensorType": sensor_type},
        fields=fields,
        time_ns=_event_time_ns(event),
    )


def _status_online_value(event: EdgeEvent) -> bool:
    payload = event.payload
    online = _first_value(payload, ("online",))
    if online is not None:
        return _coerce_online_value(online)
    status = _first_value(payload, ("status", "state"))
    if status is not None:
        return _coerce_online_value(status)
    return event.event_type.lower() != "lwt"


def _coerce_online_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"offline", "down", "false", "0", "no"}:
            return False
        if normalized in {"online", "up", "true", "1", "yes"}:
            return True
    return bool(value)


def _generic_event_point(event: EdgeEvent) -> InfluxPoint:
    fields = dict(event.payload)
    fields["timestamp"] = event.timestamp
    return InfluxPoint(
        measurement="edge_event",
        tags={"deviceId": event.device_id, "eventType": event.event_type},
        fields=fields,
        time_ns=_event_time_ns(event),
    )


def _escape_flux_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _flux_range_value(value: str) -> str:
    if not _FLUX_DURATION_PATTERN.fullmatch(value):
        raise ValueError("Flux range value must be a duration such as -1h")
    return value


def _range_bound(value: str | None, anchor: float) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if _FLUX_DURATION_PATTERN.fullmatch(text):
        amount = int(text[:-1])
        unit = text[-1]
        return anchor + amount * _DURATION_UNITS_SECONDS[unit]
    return float(text)


def _event_in_range(
    event: EdgeEvent,
    *,
    start: str | None,
    stop: str | None,
    anchor: float,
) -> bool:
    start_bound = _range_bound(start, anchor)
    stop_bound = _range_bound(stop, anchor)
    if start_bound is not None and event.timestamp < start_bound:
        return False
    if stop_bound is not None and event.timestamp > stop_bound:
        return False
    return True


def _alert_level_value(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _ALLOWED_ALERT_LEVELS:
        allowed = ", ".join(sorted(_ALLOWED_ALERT_LEVELS))
        raise ValueError(f"alert level must be one of: {allowed}")
    return normalized


def _event_to_alert_row(event: EdgeEvent) -> dict[str, Any]:
    payload = event.payload
    cpa = _first_value(payload, ("cpa", "cpa_distance_m"))
    return {
        "device_id": event.device_id,
        "timestamp": event.timestamp,
        "seq": event.seq,
        "risk_level": _first_value(payload, ("level", "risk_level"), "none"),
        "tcpa": _first_value(payload, ("tcpa", "tcpa_seconds")),
        "cpa": cpa,
        "dist": _first_value(payload, ("dist", "distance_m"), cpa),
        "message": _first_value(payload, ("msg", "message"), ""),
    }


def _event_to_track_row(event: EdgeEvent) -> dict[str, Any]:
    payload = event.payload
    x = _first_value(payload, ("x",))
    y = _first_value(payload, ("y",))
    coordinate = _first_value(
        payload,
        ("world_coordinate", "pixel_coordinate", "coordinate"),
    )
    if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
        x = x if x is not None else coordinate[0]
        y = y if y is not None else coordinate[1]
    return {
        "device_id": event.device_id,
        "timestamp": event.timestamp,
        "seq": event.seq,
        "track_id": _first_value(
            payload,
            ("trackId", "track_id", "id"),
            "unknown",
        ),
        "x": x,
        "y": y,
        "lon": _first_value(payload, ("lon", "longitude")),
        "lat": _first_value(payload, ("lat", "latitude")),
        "class_name": _first_value(payload, ("cls", "class_name", "class")),
        "confidence": _first_value(payload, ("conf", "confidence")),
    }


def _event_to_sensor_row(event: EdgeEvent) -> dict[str, Any]:
    payload = dict(event.payload)
    return {
        "device_id": event.device_id,
        "timestamp": event.timestamp,
        "seq": event.seq,
        "sensor_type": _first_value(payload, ("sensor_type", "type"), "unknown"),
        "payload": payload,
    }


def _tables_to_alert_rows(tables: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in tables:
        for record in getattr(table, "records", ()):
            values = dict(getattr(record, "values", {}))
            timestamp = values.get("timestamp") or str(values.get("_time"))
            rows.append(
                {
                    "device_id": values.get("deviceId"),
                    "timestamp": timestamp,
                    "risk_level": values.get("level"),
                    "tcpa": values.get("tcpa"),
                    "cpa": values.get("cpa"),
                    "dist": values.get("dist"),
                    "message": values.get("msg"),
                }
            )
    return rows


def _tables_to_sensor_rows(tables: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in tables:
        for record in getattr(table, "records", ()):
            values = dict(getattr(record, "values", {}))
            timestamp = values.get("timestamp") or str(values.get("_time"))
            payload = {
                key: value
                for key, value in values.items()
                if not str(key).startswith("_")
                and key not in {"deviceId", "sensorType", "result", "table"}
            }
            rows.append(
                {
                    "device_id": values.get("deviceId"),
                    "timestamp": timestamp,
                    "sensor_type": values.get("sensorType"),
                    "payload": payload,
                }
            )
    return rows


def _tables_to_track_rows(tables: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in tables:
        for record in getattr(table, "records", ()):
            values = dict(getattr(record, "values", {}))
            timestamp = values.get("timestamp") or str(values.get("_time"))
            rows.append(
                {
                    "device_id": values.get("deviceId"),
                    "timestamp": timestamp,
                    "track_id": values.get("trackId"),
                    "x": values.get("x"),
                    "y": values.get("y"),
                    "lon": values.get("lon"),
                    "lat": values.get("lat"),
                    "class_name": values.get("cls"),
                    "confidence": values.get("conf"),
                }
            )
    return rows


@dataclass
class InfluxWriter:
    """In-memory writer used by tests and as a no-token fallback."""

    written_events: list[EdgeEvent] = field(default_factory=list)
    written_points: list[InfluxPoint] = field(default_factory=list)

    def write_event(self, event: EdgeEvent) -> None:
        point = event_to_influx_point(event)
        self.write_point(point)
        self.written_events.append(event)
        self.written_points.append(point)

    def write_point(self, point: InfluxPoint) -> None:
        return None

    def query_alerts(
        self,
        *,
        start: str = "-1h",
        stop: str | None = None,
        level: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        safe_level = _alert_level_value(level) if level else None
        alarm_events = [
            event
            for event in self.written_events
            if event.event_type.lower() == "alarm"
        ]
        anchor = max((event.timestamp for event in alarm_events), default=0.0)
        rows = []
        for event in alarm_events:
            if not _event_in_range(event, start=start, stop=stop, anchor=anchor):
                continue
            row = _event_to_alert_row(event)
            if safe_level and row["risk_level"] != safe_level:
                continue
            rows.append(row)
        return rows[-safe_limit:]

    def query_tracks(
        self,
        *,
        device_id: str | None = None,
        start: str | None = None,
        stop: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        track_events = [
            event
            for event in self.written_events
            if event.event_type.lower() == "track"
        ]
        anchor = max((event.timestamp for event in track_events), default=0.0)
        rows = []
        for event in track_events:
            if device_id and event.device_id != device_id:
                continue
            if not _event_in_range(event, start=start, stop=stop, anchor=anchor):
                continue
            rows.append(_event_to_track_row(event))
        return rows[-safe_limit:]

    def query_sensors(
        self,
        *,
        device_id: str | None = None,
        sensor_type: str | None = None,
        start: str | None = None,
        stop: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        sensor_events = [
            event
            for event in self.written_events
            if event.event_type.lower() == "sensor"
        ]
        anchor = max((event.timestamp for event in sensor_events), default=0.0)
        rows = []
        for event in sensor_events:
            if device_id and event.device_id != device_id:
                continue
            row = _event_to_sensor_row(event)
            if sensor_type and row["sensor_type"] != sensor_type:
                continue
            if not _event_in_range(event, start=start, stop=stop, anchor=anchor):
                continue
            rows.append(row)
        return rows[-safe_limit:]


class InfluxDBWriter(InfluxWriter):
    """Real InfluxDB writer using the official client when configured."""

    def __init__(
        self,
        *,
        url: str,
        org: str,
        bucket: str,
        token: str | None = None,
        client: Any | None = None,
        write_api: Any | None = None,
    ) -> None:
        super().__init__()
        self.org = org
        self.bucket = bucket
        self._client = client
        self._query_api = getattr(client, "query_api", lambda: None)()
        if write_api is not None:
            self._write_api = write_api
            return
        if client is not None:
            self._write_api = client.write_api()
            return
        if token is None:
            raise ValueError("InfluxDB token is required")
        try:
            from influxdb_client import InfluxDBClient
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "influxdb-client is required for InfluxDBWriter"
            ) from exc
        self._client = InfluxDBClient(url=url, token=token, org=org)
        self._write_api = self._client.write_api()
        self._query_api = self._client.query_api()

    def write_point(self, point: InfluxPoint) -> None:
        self._write_api.write(
            bucket=self.bucket,
            org=self.org,
            record=point.to_line_protocol(),
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()

    def query_alerts(
        self,
        *,
        start: str = "-1h",
        stop: str | None = None,
        level: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self._query_api is None:
            return super().query_alerts(
                start=start,
                stop=stop,
                level=level,
                limit=limit,
            )
        safe_limit = max(1, min(int(limit), 1000))
        safe_start = _flux_range_value(start)
        stop_clause = f", stop: {_flux_range_value(stop)}" if stop else ""
        filters = [
            'r["_measurement"] == "alarm_event"',
        ]
        if level:
            safe_level = _alert_level_value(level)
            filters.append(f'r["level"] == "{safe_level}"')
        filter_expression = " and ".join(filters)
        query = (
            f'from(bucket: "{_escape_flux_string(self.bucket)}")'
            f" |> range(start: {safe_start}{stop_clause})"
            f" |> filter(fn: (r) => {filter_expression})"
            ' |> pivot(rowKey: ["_time"],'
            ' columnKey: ["_field"], valueColumn: "_value")'
            f" |> limit(n: {safe_limit})"
        )
        tables = self._query_api.query(query, org=self.org)
        return _tables_to_alert_rows(tables)

    def query_tracks(
        self,
        *,
        device_id: str | None = None,
        start: str | None = None,
        stop: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self._query_api is None:
            return super().query_tracks(
                device_id=device_id,
                start=start,
                stop=stop,
                limit=limit,
            )
        safe_limit = max(1, min(int(limit), 1000))
        safe_start = _flux_range_value(start or "-1h")
        stop_clause = f", stop: {_flux_range_value(stop)}" if stop else ""
        filters = [
            'r["_measurement"] == "target_track"',
        ]
        if device_id:
            safe_device_id = _escape_flux_string(device_id)
            filters.append(f'r["deviceId"] == "{safe_device_id}"')
        filter_expression = " and ".join(filters)
        query = (
            f'from(bucket: "{_escape_flux_string(self.bucket)}")'
            f" |> range(start: {safe_start}{stop_clause})"
            f" |> filter(fn: (r) => {filter_expression})"
            ' |> pivot(rowKey: ["_time"],'
            ' columnKey: ["_field"], valueColumn: "_value")'
            f" |> limit(n: {safe_limit})"
        )
        tables = self._query_api.query(query, org=self.org)
        return _tables_to_track_rows(tables)

    def query_sensors(
        self,
        *,
        device_id: str | None = None,
        sensor_type: str | None = None,
        start: str | None = None,
        stop: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self._query_api is None:
            return super().query_sensors(
                device_id=device_id,
                sensor_type=sensor_type,
                start=start,
                stop=stop,
                limit=limit,
            )
        safe_limit = max(1, min(int(limit), 1000))
        safe_start = _flux_range_value(start or "-1h")
        stop_clause = f", stop: {_flux_range_value(stop)}" if stop else ""
        filters = [
            'r["_measurement"] == "sensor_reading"',
        ]
        if device_id:
            filters.append(f'r["deviceId"] == "{_escape_flux_string(device_id)}"')
        if sensor_type:
            filters.append(f'r["sensorType"] == "{_escape_flux_string(sensor_type)}"')
        filter_expression = " and ".join(filters)
        query = (
            f'from(bucket: "{_escape_flux_string(self.bucket)}")'
            f" |> range(start: {safe_start}{stop_clause})"
            f" |> filter(fn: (r) => {filter_expression})"
            ' |> pivot(rowKey: ["_time"],'
            ' columnKey: ["_field"], valueColumn: "_value")'
            f" |> limit(n: {safe_limit})"
        )
        tables = self._query_api.query(query, org=self.org)
        return _tables_to_sensor_rows(tables)


__all__ = [
    "InfluxDBWriter",
    "InfluxPoint",
    "InfluxWriter",
    "event_to_influx_point",
]
