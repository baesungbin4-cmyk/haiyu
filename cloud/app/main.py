"""Cloud FastAPI application factory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from cloud.app.api.routes import create_router
from cloud.app.config import CloudSettings, load_settings
from cloud.app.db.influx import InfluxDBWriter, InfluxWriter
from cloud.app.mqtt.consumer import EdgeMqttConsumer, MqttConsumerConfig
from cloud.app.websocket.manager import WebSocketManager

try:  # pragma: no cover - FastAPI not installed in this environment
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = None
    CORSMiddleware = None


def create_app(settings: CloudSettings | None = None):
    if FastAPI is None:
        raise RuntimeError("fastapi is required to create the cloud app")
    settings = settings or load_settings()
    writer = _create_influx_writer(settings)
    websocket_manager = WebSocketManager()
    consumer = EdgeMqttConsumer(
        writer=writer,
        websocket_manager=websocket_manager,
        mqtt_config=MqttConsumerConfig.from_settings(settings),
    )
    app = FastAPI(
        title="Haiyu Cloud Backend",
        lifespan=_create_lifespan(settings, consumer, writer),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(create_router(settings, consumer))
    app.state.settings = settings
    app.state.consumer = consumer
    app.state.influx_writer = writer
    app.state.websocket_manager = websocket_manager
    return app


def _create_lifespan(
    settings: CloudSettings,
    consumer: EdgeMqttConsumer,
    writer: InfluxWriter,
):
    @asynccontextmanager
    async def lifespan(app: object) -> AsyncIterator[None]:
        if settings.mqtt_enabled:
            consumer.event_loop = asyncio.get_running_loop()
            consumer.start()
        try:
            yield
        finally:
            if settings.mqtt_enabled:
                consumer.stop()
            close = getattr(writer, "close", None)
            if close is not None:
                close()

    return lifespan


def _create_influx_writer(settings: CloudSettings) -> InfluxWriter:
    if settings.influxdb_token is None:
        return InfluxWriter()
    return InfluxDBWriter(
        url=settings.influxdb_url,
        org=settings.influxdb_org,
        bucket=settings.influxdb_bucket,
        token=settings.influxdb_token,
    )


app = create_app() if FastAPI is not None else None


__all__ = ["app", "create_app"]
