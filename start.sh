#!/bin/bash
# Haiyu Full Stack Startup Script
set -e

ROOT=/d/A_haiyu
cd $ROOT

# Load environment variables from .env
set -a
source .env
set +a

echo "=== Haiyu Full Stack Startup ==="
echo "MQTT: ${MQTT_HOST:-localhost}:${MQTT_PORT:-1883}"
echo "InfluxDB: ${INFLUXDB_URL:-http://localhost:8086}"
echo "API: http://${API_BIND_HOST:-127.0.0.1}:${API_PORT:-8000}"
echo "Frontend: http://${FRONTEND_BIND_HOST:-127.0.0.1}:${FRONTEND_PORT:-5173}"
echo ""

# 1. Ensure Docker containers are running
echo "[1/4] Checking Docker containers..."
if ! docker ps --format '{{.Names}}' | grep -q haiyu-mqtt; then
    echo "  Starting MQTT..."
    docker run -d --name haiyu-mqtt \
        -p 1883:1883 -p 9001:9001 \
        -v $ROOT/docker/mosquitto/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro \
        eclipse-mosquitto:2 2>/dev/null || docker start haiyu-mqtt
    sleep 1
    # Configure MQTT auth
    docker exec haiyu-mqtt sh -c "
        mosquitto_passwd -b -c /tmp/mosquitto.passwd '$MQTT_USERNAME' '$MQTT_PASSWORD' &&
        mosquitto_passwd -b /tmp/mosquitto.passwd '$MQTT_EDGE_USERNAME' '$MQTT_EDGE_PASSWORD' &&
        {
          echo 'user $MQTT_USERNAME';
          echo 'topic read haiyu/+/status';
          echo 'topic read haiyu/+/track';
          echo 'topic read haiyu/+/alarm';
          echo 'topic read haiyu/+/lwt';
          echo '';
          echo 'pattern write haiyu/%u/status';
          echo 'pattern write haiyu/%u/track';
          echo 'pattern write haiyu/%u/alarm';
          echo 'pattern write haiyu/%u/lwt';
        } > /tmp/mosquitto.acl &&
        kill -HUP 1
    " 2>/dev/null
    echo "  MQTT configured."
else
    echo "  MQTT already running."
fi

if ! docker ps --format '{{.Names}}' | grep -q haiyu-influxdb; then
    echo "  Starting InfluxDB..."
    docker run -d --name haiyu-influxdb \
        -p 8086:8086 \
        -e DOCKER_INFLUXDB_INIT_MODE=setup \
        -e DOCKER_INFLUXDB_INIT_USERNAME="$INFLUXDB_USERNAME" \
        -e DOCKER_INFLUXDB_INIT_PASSWORD="$INFLUXDB_PASSWORD" \
        -e DOCKER_INFLUXDB_INIT_ORG="${INFLUXDB_ORG:-haiyu}" \
        -e DOCKER_INFLUXDB_INIT_BUCKET="${INFLUXDB_BUCKET:-haiyu}" \
        -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN="$INFLUXDB_TOKEN" \
        influxdb:2.7 2>/dev/null || docker start haiyu-influxdb
    echo "  InfluxDB started."
else
    echo "  InfluxDB already running."
fi
echo ""

# 2. Start FastAPI backend
echo "[2/4] Starting FastAPI backend..."
PYTHONPATH=$ROOT python -m uvicorn cloud.app.main:app \
    --host 0.0.0.0 --port ${API_PORT:-8000} &
API_PID=$!
sleep 3
echo "  API PID=$API_PID on port ${API_PORT:-8000}"

# 3. Start Frontend dev server
echo "[3/4] Starting Frontend dev server..."
cd $ROOT/frontend
VITE_API_BASE_URL="http://localhost:${API_PORT:-8000}" \
VITE_WS_BASE_URL="ws://localhost:${API_PORT:-8000}" \
npm run dev -- --host 127.0.0.1 --port ${FRONTEND_PORT:-5173} &
FE_PID=$!
sleep 2
cd $ROOT
echo "  Frontend PID=$FE_PID"

# 4. Run quick smoke test
echo "[4/4] Smoke test..."
echo ""
echo "  /health: $(curl -s http://127.0.0.1:${API_PORT:-8000}/health)"
echo "  /docs:   $(curl -s -o /dev/null -w 'HTTP %{http_code}' http://127.0.0.1:${API_PORT:-8000}/docs)"
echo "  frontend: $(curl -s -o /dev/null -w 'HTTP %{http_code}' http://127.0.0.1:${FRONTEND_PORT:-5173})"
echo ""

echo "=== All services running ==="
echo "  API:      http://127.0.0.1:${API_PORT:-8000}"
echo "  Docs:     http://127.0.0.1:${API_PORT:-8000}/docs"
echo "  Frontend: http://127.0.0.1:${FRONTEND_PORT:-5173}"
echo ""
echo "PIDs: API=$API_PID  Frontend=$FE_PID"
echo "Logs: docker logs haiyu-mqtt / haiyu-influxdb"
