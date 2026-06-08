# Haiyu Reproduction Monorepo

Prototype reproduction for the Haiyu port-safety IoT platform / weak-vision vessel approach warning terminal.

Current implementation is a prototype core plus integration skeleton; unverified or placeholder paths are documented as such.

## Directory Layout

| Path | Purpose |
|---|---|
| `edge/` | Edge node placeholder for decode, enhancement, detection, tracking, trajectory prediction, risk evaluation, calibration, and cache modules. |
| `cloud/` | Cloud placeholder for FastAPI, WebSocket, InfluxDB integration, and MQTT consumption. |
| `frontend/` | Static cloud monitor for health, latest alerts, and realtime WebSocket events. |
| `training/` | Detection and trajectory training placeholders. |
| `docker/` | Deployment and service configuration placeholder. |
| `tests/` | Edge, cloud, integration, and performance test placeholders. |

## Compose Services And Ports

| Service | Compose name | Default port | Notes |
|---|---|---:|---|
| MQTT broker | `mqtt` | internal `1883` | Mosquitto MQTT TCP listener; no host port is published by default. |
| InfluxDB | `influxdb` | internal `8086` | Initialized from `.env` values; no host port is published by default. |
| FastAPI backend | `api` | `8000` | Runs `cloud.app.main:app`; binds to `127.0.0.1` by default. |
| Static monitor | `frontend` | `5173` | Nginx-served monitor page; binds to `127.0.0.1` by default. |

Copy `.env.example` to `.env` before local deployment and replace every
`local-dev-*` value. MQTT and InfluxDB stay internal-only in the default
compose file; API and frontend bind to localhost unless `API_BIND_HOST` or
`FRONTEND_BIND_HOST` is changed.

## Authentication Status

This skeleton implements repository-local security controls for development
and smoke testing.

- FastAPI REST: `/api/v1/*` requires `Authorization: Bearer $API_TOKEN` by
  default. Additional comma-separated tokens can be supplied through
  `API_TOKENS`. Missing tokens reject protected endpoints unless
  `ALLOW_UNAUTHENTICATED=true` is explicitly set for isolated local work.
- WebSocket `/ws/realtime`: accepts `Authorization: Bearer $API_TOKEN` for
  non-browser clients and a base64url bearer token in `Sec-WebSocket-Protocol`
  for browser clients. URL query tokens are disabled by default because access
  logs can capture them. The route checks `Origin` against `CORS_ORIGINS`, and
  enforces `WEBSOCKET_MAX_CONNECTIONS` atomically at connect time.
- MQTT: Mosquitto rejects anonymous clients, and generates its password file
  plus ACL file from `.env` at container startup. The cloud consumer account
  can read contract topics; edge usernames can write only
  `haiyu/<username>/{status,track,alarm,lwt}`.
- MQTT events: cloud ingestion can enforce `MQTT_ALLOWED_DEVICE_IDS` and
  `MQTT_EVENT_HMAC_SECRET`; duplicate `(device_id, seq)` events are still
  dropped before storage.
- InfluxDB: admin token and password are required through `.env`, with no host
  port published by the default compose file.

Remaining deployment responsibilities: replace local development secrets,
provision real secret storage, enable broker/API TLS with trusted certificates,
and add any production IAM features such as scopes, audit logging, rate
limiting, and token rotation required by the target environment. No software
change can prove absolute safety; this repository keeps known risks fail-closed
where it can and documents the deployment controls that must exist outside the
codebase.

## Cloud Runtime Notes

MQTT ingestion maps contract topics to InfluxDB measurements:

| Event | Measurement | Tags | Fields |
|---|---|---|---|
| `alarm` | `alarm_event` | `deviceId`, `level` | `tcpa`, `cpa`, `dist`, `msg`, `timestamp` |
| `track` | `target_track` | `deviceId`, `trackId` | `x`, `y`, `lon`, `lat`, `cls`, `conf`, `timestamp` |
| `sensor` | `sensor_reading` | `deviceId`, `sensorType` | Raw scalar sensor payload fields plus `timestamp` |
| `status` / `lwt` | `device_status` | `deviceId` | `online`, `cpu`, `mem`, `npu`, `timestamp` |

The API container in `docker-compose.yml` runs the real FastAPI backend. The
frontend compose service serves the static monitor in `frontend/index.html`.
Implemented REST/WebSocket entry points:

| Route | Status |
|---|---|
| `GET /api/v1/devices`, `GET /api/v1/devices/{id}` | Returns devices observed from ingested MQTT status/track/alarm events. Empty means no events have been ingested in this process/query scope. |
| `GET /api/v1/alarms?from&to&level`, `GET /api/v1/alerts?from&to&level` | Queries alarm events from InfluxDB when configured, or the in-memory writer during local tests. |
| `GET /api/v1/tracks?deviceId&from&to` | Queries target track events from InfluxDB when configured, or the in-memory writer during local tests. |
| `GET /api/v1/sensors?deviceId&type&from&to` | Queries sensor events received on `haiyu/{id}/sensor`. Empty response means no real sensor event has been ingested in the query scope. |
| `GET /api/v1/stats/summary` | Summarizes currently ingested devices, latest alerts, and track event counts. |
| `POST /api/v1/calib/{id}` | Accepts calibration control points for the REST control plane; execution is not claimed until an edge-side result is reported. |
| `GET /api/v1/alerts/latest`, `WS /ws/realtime` | Existing latest-alert and realtime event channels. |

## Metrics And Validation Boundaries

No real-port validation is claimed by this skeleton. Prototype-level metrics from the contracts must be written as `样机级，未经实港验证` whenever repeated in code, docs, reports, or comments.

Simulation outputs must be marked as simulation; they must not be described as real-port validation.

The current frontend draws a schematic berth/fairway situation panel, not an administrative or territory map. If a future screen adds jurisdictional or territory map content, its source and valid approval number must be documented before release.

## Licensing And AI Disclosure

The project license is `AGPL-3.0-only` in `pyproject.toml`. Ultralytics YOLOv8 is AGPL-3.0; any distribution or network service deployment that includes it must evaluate and satisfy AGPL source-disclosure obligations. Other dependency licenses still need a deployment-specific review before release.

AI assistance is disclosed at approximately 5%; the project contract states that core algorithms, architecture, training, and engineering are team-owned. No patent is described as granted or accepted; planned patent work is only `拟申请`.

## Validation

```bash
docker compose --env-file .env.example up --build
docker compose --env-file .env.example config
python -m pytest
npm --prefix frontend run lint
npm --prefix frontend run build
pre-commit validate-config
```

`pre-commit validate-config` requires the `pre-commit` CLI to be installed.
