import asyncio
import base64
import json
import os

from cloud.app.api.routes import (
    _calibration_ack_payload,
    _is_allowed_websocket_origin,
    _is_valid_websocket_token,
    _latest_device_rows,
    _summary_payload,
    require_api_token,
)
from cloud.app.config import CloudSettings, DEFAULT_MQTT_TOPICS, load_settings
from cloud.app.db.influx import (
    InfluxDBWriter,
    InfluxWriter,
    event_to_influx_point,
)
from cloud.app.main import _create_lifespan
from cloud.app.mqtt.consumer import (
    EdgeMqttConsumer,
    MqttConsumerConfig,
    _sign_event,
)
from cloud.app.schemas import EdgeEvent
from cloud.app.websocket.manager import WebSocketManager


def _event_dict(
    *,
    device_id: str = "edge-01",
    seq: int = 7,
    event_type: str = "alarm",
    payload: dict | None = None,
) -> dict:
    payload = payload or {
        "risk_level": "high",
        "should_alert": True,
        "tcpa_seconds": 12.5,
        "cpa_distance_m": 35.0,
        "message": "TCPA/CPA alert",
    }
    return {
        "device_id": device_id,
        "timestamp": 100.0,
        "seq": seq,
        "event_type": event_type,
        "topic": f"haiyu/{device_id}/{event_type}",
        "payload": payload,
        "qos": 1,
        "retain": False,
    }


def _signed_event_dict(
    *,
    secret: str,
    device_id: str = "edge-01",
    seq: int = 71,
) -> dict:
    raw = _event_dict(device_id=device_id, seq=seq)
    event = EdgeEvent.from_dict(raw)
    payload = dict(raw["payload"])
    payload["signature"] = _sign_event(event, secret)
    raw["payload"] = payload
    return raw


def _websocket_protocol_header(token: str) -> str:
    encoded = base64.urlsafe_b64encode(token.encode("utf-8"))
    token_value = encoded.decode("ascii").rstrip("=")
    return f"haiyu.realtime.v1, bearer.{token_value}"


def test_api_token_auth_rejects_when_unconfigured_by_default() -> None:
    try:
        require_api_token(CloudSettings(api_token=None), authorization=None)
    except PermissionError as exc:
        assert "token" in str(exc)
    else:
        raise AssertionError("unconfigured token should reject by default")


def test_api_token_auth_allows_explicit_unauthenticated_mode() -> None:
    settings = CloudSettings(api_token=None, allow_unauthenticated=True)

    require_api_token(settings, authorization=None)


def test_api_token_auth_accepts_correct_bearer() -> None:
    require_api_token(
        CloudSettings(api_token="secret"),
        authorization="Bearer secret",
    )


def test_api_token_auth_accepts_additional_tokens() -> None:
    require_api_token(
        CloudSettings(api_tokens=("old-secret", "new-secret")),
        authorization="Bearer new-secret",
    )


def test_api_token_auth_rejects_wrong_token() -> None:
    try:
        require_api_token(
            CloudSettings(api_token="secret"),
            authorization="Bearer wrong",
        )
    except PermissionError as exc:
        assert "token" in str(exc)
    else:
        raise AssertionError("wrong token should be rejected")


def test_websocket_token_auth_uses_header_or_subprotocol_by_default() -> None:
    settings = CloudSettings(api_token="secret")

    assert _is_valid_websocket_token(settings, "Bearer secret", None) is True
    assert (
        _is_valid_websocket_token(
            settings,
            None,
            None,
            _websocket_protocol_header("secret"),
        )
        is True
    )
    assert _is_valid_websocket_token(settings, None, "secret") is False
    assert (
        _is_valid_websocket_token(
            CloudSettings(
                api_token="secret",
                websocket_allow_query_token=True,
            ),
            None,
            "secret",
        )
        is True
    )
    wrong_token_allowed = _is_valid_websocket_token(
        settings,
        "Bearer wrong",
        "wrong",
    )
    open_ws_allowed = _is_valid_websocket_token(
        CloudSettings(api_token=None, allow_unauthenticated=True),
        None,
        None,
    )
    default_open_ws_allowed = _is_valid_websocket_token(
        CloudSettings(api_token=None),
        None,
        None,
    )

    assert wrong_token_allowed is False
    assert open_ws_allowed is True
    assert default_open_ws_allowed is False


def test_websocket_origin_must_match_cors_origins() -> None:
    settings = CloudSettings(
        cors_origins=("http://localhost:5173", "https://ops.example"),
    )

    assert _is_allowed_websocket_origin(settings, None) is True
    local_origin_allowed = _is_allowed_websocket_origin(
        settings,
        "http://localhost:5173",
    )
    evil_origin_allowed = _is_allowed_websocket_origin(
        settings,
        "http://evil.example",
    )
    assert local_origin_allowed is True
    assert evil_origin_allowed is False


def test_default_mqtt_topics_cover_contract_events() -> None:
    assert DEFAULT_MQTT_TOPICS == (
        "haiyu/+/status",
        "haiyu/+/track",
        "haiyu/+/alarm",
        "haiyu/+/sensor",
        "haiyu/+/lwt",
    )
    assert CloudSettings().mqtt_topic == ",".join(DEFAULT_MQTT_TOPICS)


def test_load_settings_treats_empty_secrets_as_unset() -> None:
    old_values = {
        name: os.environ.get(name)
        for name in (
            "API_TOKEN",
            "API_TOKENS",
            "ALLOW_UNAUTHENTICATED",
            "INFLUXDB_TOKEN",
            "MQTT_USERNAME",
            "MQTT_PASSWORD",
            "MQTT_REQUIRE_CREDENTIALS",
            "MQTT_EVENT_HMAC_SECRET",
            "MQTT_ALLOWED_DEVICE_IDS",
            "MQTT_TLS_ENABLED",
            "MQTT_TLS_CA_CERT",
            "MQTT_TLS_CLIENT_CERT",
            "MQTT_TLS_CLIENT_KEY",
            "MQTT_TLS_INSECURE",
            "WEBSOCKET_MAX_CONNECTIONS",
            "WEBSOCKET_ALLOW_QUERY_TOKEN",
        )
    }
    try:
        os.environ["API_TOKEN"] = ""
        os.environ["API_TOKENS"] = ""
        os.environ["ALLOW_UNAUTHENTICATED"] = ""
        os.environ["INFLUXDB_TOKEN"] = ""
        os.environ["MQTT_USERNAME"] = ""
        os.environ["MQTT_PASSWORD"] = ""
        os.environ["MQTT_REQUIRE_CREDENTIALS"] = ""
        os.environ["MQTT_EVENT_HMAC_SECRET"] = ""
        os.environ["MQTT_ALLOWED_DEVICE_IDS"] = ""
        os.environ["MQTT_TLS_ENABLED"] = ""
        os.environ["MQTT_TLS_CA_CERT"] = ""
        os.environ["MQTT_TLS_CLIENT_CERT"] = ""
        os.environ["MQTT_TLS_CLIENT_KEY"] = ""
        os.environ["MQTT_TLS_INSECURE"] = ""
        os.environ["WEBSOCKET_MAX_CONNECTIONS"] = "7"
        os.environ["WEBSOCKET_ALLOW_QUERY_TOKEN"] = ""

        settings = load_settings()
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert settings.api_token is None
    assert settings.api_tokens == ()
    assert settings.allow_unauthenticated is False
    assert settings.influxdb_token is None
    assert settings.mqtt_username is None
    assert settings.mqtt_password is None
    assert settings.mqtt_require_credentials is True
    assert settings.mqtt_event_hmac_secret is None
    assert settings.mqtt_allowed_device_ids == ()
    assert settings.mqtt_tls_enabled is False
    assert settings.mqtt_tls_ca_cert is None
    assert settings.mqtt_tls_client_cert is None
    assert settings.mqtt_tls_client_key is None
    assert settings.mqtt_tls_insecure is False
    assert settings.websocket_max_connections == 7
    assert settings.websocket_allow_query_token is False


def test_load_settings_parses_security_settings() -> None:
    old_values = {
        name: os.environ.get(name)
        for name in (
            "API_TOKENS",
            "ALLOW_UNAUTHENTICATED",
            "MQTT_REQUIRE_CREDENTIALS",
            "MQTT_EVENT_HMAC_SECRET",
            "MQTT_ALLOWED_DEVICE_IDS",
            "MQTT_TLS_ENABLED",
            "MQTT_TLS_CA_CERT",
            "MQTT_TLS_CLIENT_CERT",
            "MQTT_TLS_CLIENT_KEY",
            "MQTT_TLS_INSECURE",
            "WEBSOCKET_ALLOW_QUERY_TOKEN",
        )
    }
    try:
        os.environ["API_TOKENS"] = "old-secret,new-secret"
        os.environ["ALLOW_UNAUTHENTICATED"] = "true"
        os.environ["MQTT_REQUIRE_CREDENTIALS"] = "true"
        os.environ["MQTT_EVENT_HMAC_SECRET"] = "edge-hmac"
        os.environ["MQTT_ALLOWED_DEVICE_IDS"] = "edge-01,edge-02"
        os.environ["MQTT_TLS_ENABLED"] = "true"
        os.environ["MQTT_TLS_CA_CERT"] = "/certs/ca.pem"
        os.environ["MQTT_TLS_CLIENT_CERT"] = "/certs/client.pem"
        os.environ["MQTT_TLS_CLIENT_KEY"] = "/certs/client.key"
        os.environ["MQTT_TLS_INSECURE"] = "true"
        os.environ["WEBSOCKET_ALLOW_QUERY_TOKEN"] = "true"

        settings = load_settings()
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert settings.api_tokens == ("old-secret", "new-secret")
    assert settings.allow_unauthenticated is True
    assert settings.mqtt_require_credentials is True
    assert settings.mqtt_event_hmac_secret == "edge-hmac"
    assert settings.mqtt_allowed_device_ids == ("edge-01", "edge-02")
    assert settings.mqtt_tls_enabled is True
    assert settings.mqtt_tls_ca_cert == "/certs/ca.pem"
    assert settings.mqtt_tls_client_cert == "/certs/client.pem"
    assert settings.mqtt_tls_client_key == "/certs/client.key"
    assert settings.mqtt_tls_insecure is True
    assert settings.websocket_allow_query_token is True


def test_influx_alarm_mapping_matches_contract() -> None:
    event = EdgeEvent.from_dict(_event_dict())

    point = event_to_influx_point(event)

    assert point.measurement == "alarm_event"
    assert point.tags == {"deviceId": "edge-01", "level": "high"}
    assert point.fields["tcpa"] == 12.5
    assert point.fields["cpa"] == 35.0
    assert point.fields["dist"] == 35.0
    assert point.fields["msg"] == "TCPA/CPA alert"
    assert point.fields["timestamp"] == 100.0
    assert point.time_ns == 100_000_000_000


def test_influx_lwt_online_field_prefers_payload_state() -> None:
    event = EdgeEvent.from_dict(
        _event_dict(
            event_type="lwt",
            payload={"online": "offline", "cpu": 0.2},
        )
    )

    point = event_to_influx_point(event)

    assert point.measurement == "device_status"
    assert point.fields["online"] is False
    assert point.fields["cpu"] == 0.2


def test_influx_sensor_mapping_matches_sensor_interface() -> None:
    event = EdgeEvent.from_dict(
        _event_dict(
            event_type="sensor",
            payload={
                "sensor_type": "meteo",
                "wind_speed_mps": 4.2,
                "temperature_c": 21.5,
            },
        )
    )

    point = event_to_influx_point(event)

    assert point.measurement == "sensor_reading"
    assert point.tags == {"deviceId": "edge-01", "sensorType": "meteo"}
    assert point.fields["wind_speed_mps"] == 4.2
    assert point.fields["temperature_c"] == 21.5


def test_mqtt_consumer_writes_influx_and_broadcasts_alert() -> None:
    async def run() -> None:
        writer = InfluxWriter()
        manager = WebSocketManager()
        consumer = EdgeMqttConsumer(writer=writer, websocket_manager=manager)

        event = await consumer.accept_event(_event_dict())

        assert writer.written_events == [event]
        assert writer.written_points[0].measurement == "alarm_event"
        assert len(consumer.latest_alerts) == 1
        assert manager.broadcast_history[0]["type"] == "alert"
        assert manager.broadcast_history[0]["data"]["risk_level"] == "high"

    asyncio.run(run())


def test_mqtt_consumer_broadcasts_sensor_event() -> None:
    async def run() -> None:
        writer = InfluxWriter()
        manager = WebSocketManager()
        consumer = EdgeMqttConsumer(writer=writer, websocket_manager=manager)

        await consumer.accept_event(
            _event_dict(
                event_type="sensor",
                payload={"sensor_type": "meteo", "wind_speed_mps": 4.2},
            )
        )

        assert writer.written_points[0].measurement == "sensor_reading"
        assert manager.broadcast_history[0]["type"] == "sensor"
        data = manager.broadcast_history[0]["data"]
        assert data["sensor_type"] == "meteo"
        assert data["wind_speed_mps"] == 4.2

    asyncio.run(run())


def test_mqtt_consumer_dedupes_device_seq_before_writing() -> None:
    async def run() -> None:
        writer = InfluxWriter()
        manager = WebSocketManager()
        consumer = EdgeMqttConsumer(writer=writer, websocket_manager=manager)

        first = await consumer.accept_event(_event_dict(seq=8))
        duplicate = await consumer.accept_event(_event_dict(seq=8))

        assert first is not None
        assert duplicate is None
        assert len(writer.written_events) == 1
        assert len(manager.broadcast_history) == 1

    asyncio.run(run())


def test_mqtt_consumer_accepts_signed_allowed_device_event() -> None:
    async def run() -> None:
        writer = InfluxWriter()
        manager = WebSocketManager()
        consumer = EdgeMqttConsumer(
            writer=writer,
            websocket_manager=manager,
            mqtt_config=MqttConsumerConfig(
                require_credentials=False,
                event_hmac_secret="edge-secret",
                allowed_device_ids=("edge-01",),
            ),
        )

        signed_event = _signed_event_dict(secret="edge-secret")
        event = await consumer.accept_event(signed_event)

        assert event is not None
        assert writer.written_events == [event]

    asyncio.run(run())


def test_mqtt_consumer_rejects_missing_event_signature() -> None:
    async def run() -> None:
        writer = InfluxWriter()
        consumer = EdgeMqttConsumer(
            writer=writer,
            websocket_manager=WebSocketManager(),
            mqtt_config=MqttConsumerConfig(
                require_credentials=False,
                event_hmac_secret="edge-secret",
            ),
        )

        try:
            await consumer.accept_event(_event_dict(seq=72))
        except ValueError as exc:
            assert "signature" in str(exc)
        else:
            raise AssertionError("unsigned MQTT event should be rejected")

        assert writer.written_events == []

    asyncio.run(run())


def test_mqtt_consumer_rejects_disallowed_device() -> None:
    async def run() -> None:
        writer = InfluxWriter()
        consumer = EdgeMqttConsumer(
            writer=writer,
            websocket_manager=WebSocketManager(),
            mqtt_config=MqttConsumerConfig(
                require_credentials=False,
                event_hmac_secret="edge-secret",
                allowed_device_ids=("edge-01",),
            ),
        )

        try:
            await consumer.accept_event(
                _signed_event_dict(
                    secret="edge-secret",
                    device_id="edge-02",
                    seq=73,
                )
            )
        except ValueError as exc:
            assert "not allowed" in str(exc)
        else:
            raise AssertionError("unexpected MQTT device should be rejected")

        assert writer.written_events == []

    asyncio.run(run())


def test_latest_alerts_are_bounded() -> None:
    async def run() -> None:
        writer = InfluxWriter()
        manager = WebSocketManager()
        consumer = EdgeMqttConsumer(
            writer=writer,
            websocket_manager=manager,
            max_alerts=2,
        )

        await consumer.accept_event(_event_dict(seq=1))
        await consumer.accept_event(_event_dict(seq=2))
        await consumer.accept_event(_event_dict(seq=3))

        assert [alert.seq for alert in consumer.latest_alerts] == [2, 3]

    asyncio.run(run())


def test_mqtt_connect_requires_credentials_by_default() -> None:
    class FakeClient:
        pass

    consumer = EdgeMqttConsumer(
        writer=InfluxWriter(),
        websocket_manager=WebSocketManager(),
        mqtt_client=FakeClient(),
    )

    try:
        consumer.connect()
    except ValueError as exc:
        assert "username" in str(exc)
    else:
        raise AssertionError("MQTT credentials should be required by default")


def test_mqtt_connect_configures_credentials_and_tls() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.credentials = None
            self.tls = None
            self.tls_insecure = None
            self.connected = None

        def username_pw_set(self, username, password=None) -> None:
            self.credentials = (username, password)

        def tls_set(self, ca_certs=None, certfile=None, keyfile=None) -> None:
            self.tls = (ca_certs, certfile, keyfile)

        def tls_insecure_set(self, value) -> None:
            self.tls_insecure = value

        def connect(self, host, port, keepalive) -> None:
            self.connected = (host, port, keepalive)

    fake_client = FakeClient()
    consumer = EdgeMqttConsumer(
        writer=InfluxWriter(),
        websocket_manager=WebSocketManager(),
        mqtt_client=fake_client,
        mqtt_config=MqttConsumerConfig(
            username="user",
            password="pass",
            tls_enabled=True,
            tls_ca_cert="/certs/ca.pem",
            tls_client_cert="/certs/client.pem",
            tls_client_key="/certs/client.key",
            tls_insecure=True,
        ),
    )

    consumer.connect()

    assert fake_client.credentials == ("user", "pass")
    assert fake_client.tls == (
        "/certs/ca.pem",
        "/certs/client.pem",
        "/certs/client.key",
    )
    assert fake_client.tls_insecure is True
    assert fake_client.connected == ("localhost", 1883, 60)


def test_mqtt_client_lifecycle_uses_mock_client() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.on_connect = None
            self.on_disconnect = None
            self.on_message = None
            self.connected = None
            self.credentials = None
            self.loop_started = False
            self.loop_stopped = False
            self.disconnected = False
            self.subscriptions = []

        def username_pw_set(self, username, password=None) -> None:
            self.credentials = (username, password)

        def connect(self, host, port, keepalive) -> None:
            self.connected = (host, port, keepalive)
            self.on_connect(self, None, None, 0)

        def subscribe(self, topic, qos=0) -> None:
            self.subscriptions.append((topic, qos))

        def loop_start(self) -> None:
            self.loop_started = True

        def loop_stop(self) -> None:
            self.loop_stopped = True

        def disconnect(self) -> None:
            self.disconnected = True

    class FakeMessage:
        topic = "haiyu/edge-01/alarm"
        qos = 1
        retain = False
        payload = json.dumps(
            {
                "deviceId": "edge-01",
                "ts": 101.0,
                "seq": 42,
                "risk_level": "medium",
                "should_alert": True,
                "tcpa_seconds": 3.0,
                "cpa_distance_m": 9.0,
            }
        ).encode("utf-8")

    writer = InfluxWriter()
    manager = WebSocketManager()
    fake_client = FakeClient()
    consumer = EdgeMqttConsumer(
        writer=writer,
        websocket_manager=manager,
        mqtt_client=fake_client,
        mqtt_config=MqttConsumerConfig(
            host="broker.local",
            topics=("haiyu/+/alarm", "haiyu/+/track"),
            username="user",
            password="pass",
        ),
    )

    consumer.start()
    fake_client.on_message(fake_client, None, FakeMessage())
    consumer.stop()

    assert fake_client.connected == ("broker.local", 1883, 60)
    assert fake_client.credentials == ("user", "pass")
    assert fake_client.loop_started is True
    assert fake_client.loop_stopped is True
    assert fake_client.disconnected is True
    assert fake_client.subscriptions == [
        ("haiyu/+/alarm", 1),
        ("haiyu/+/track", 1),
    ]
    assert writer.written_events[0].event_type == "alarm"
    assert writer.written_events[0].device_id == "edge-01"


def test_mqtt_submit_event_uses_configured_running_loop() -> None:
    class FakeFuture:
        pass

    class FakeLoop:
        def __init__(self) -> None:
            self.submitted = []

        def is_running(self) -> bool:
            return True

    async def run() -> None:
        import cloud.app.mqtt.consumer as consumer_module

        fake_loop = FakeLoop()
        original = consumer_module.asyncio.run_coroutine_threadsafe

        def fake_submit(coroutine, loop):
            fake_loop.submitted.append((coroutine, loop))
            coroutine.close()
            return FakeFuture()

        consumer_module.asyncio.run_coroutine_threadsafe = fake_submit
        try:
            consumer = EdgeMqttConsumer(
                writer=InfluxWriter(),
                websocket_manager=WebSocketManager(),
                event_loop=fake_loop,
            )
            result = consumer._submit_event(
                json.dumps(_event_dict()),
                topic="haiyu/edge-01/alarm",
                qos=1,
                retain=False,
            )
        finally:
            consumer_module.asyncio.run_coroutine_threadsafe = original

        assert isinstance(result, FakeFuture)
        assert fake_loop.submitted[0][1] is fake_loop

    asyncio.run(run())


def test_mqtt_stop_clears_loop_and_drops_late_messages() -> None:
    class FakeLoop:
        def is_running(self) -> bool:
            return True

    writer = InfluxWriter()
    consumer = EdgeMqttConsumer(
        writer=writer,
        websocket_manager=WebSocketManager(),
        event_loop=FakeLoop(),
    )

    consumer.stop()
    result = consumer._submit_event(
        json.dumps(_event_dict()),
        topic="haiyu/edge-01/alarm",
        qos=1,
        retain=False,
    )

    assert result is None
    assert consumer.event_loop is None
    assert writer.written_events == []


def test_mqtt_reconnect_with_backoff_retries_mock_client() -> None:
    class ReconnectClient:
        def __init__(self) -> None:
            self.on_connect = None
            self.on_disconnect = None
            self.on_message = None
            self.attempts = 0
            self.subscriptions = []

        def reconnect(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("temporary broker error")

        def subscribe(self, topic, qos=0) -> None:
            self.subscriptions.append((topic, qos))

    sleeps = []
    client = ReconnectClient()
    consumer = EdgeMqttConsumer(
        writer=InfluxWriter(),
        websocket_manager=WebSocketManager(),
        mqtt_client=client,
        mqtt_config=MqttConsumerConfig(topics=("haiyu/+/alarm",)),
        sleep=sleeps.append,
    )

    consumer.reconnect_with_backoff(max_attempts=2)

    assert client.attempts == 2
    assert sleeps == [1.0]
    assert client.subscriptions == [("haiyu/+/alarm", 1)]


def test_mqtt_reconnect_backoff_stops_when_consumer_stops() -> None:
    class FailingReconnectClient:
        def __init__(self) -> None:
            self.on_connect = None
            self.on_disconnect = None
            self.on_message = None
            self.attempts = 0

        def reconnect(self) -> None:
            self.attempts += 1
            raise OSError("broker unavailable")

    client = FailingReconnectClient()
    consumer = EdgeMqttConsumer(
        writer=InfluxWriter(),
        websocket_manager=WebSocketManager(),
        mqtt_client=client,
    )

    def sleep_and_stop(delay) -> None:
        consumer.stop()

    consumer.sleep = sleep_and_stop
    consumer.reconnect_with_backoff()

    assert client.attempts == 1
    assert consumer._stop_event.is_set() is True


def test_mqtt_disconnect_schedules_reconnect_for_failure() -> None:
    consumer = EdgeMqttConsumer(
        writer=InfluxWriter(),
        websocket_manager=WebSocketManager(),
        mqtt_config=MqttConsumerConfig(auto_reconnect=True),
    )
    called = []
    consumer.schedule_reconnect = lambda: called.append(True)

    consumer._on_disconnect(None, None, 1)

    assert called == [True]


def test_lifespan_injects_running_loop_before_mqtt_start() -> None:
    class StartStopConsumer(EdgeMqttConsumer):
        def __init__(self) -> None:
            super().__init__(
                writer=InfluxWriter(),
                websocket_manager=WebSocketManager(),
            )
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True
            assert self.event_loop is asyncio.get_running_loop()

        def stop(self) -> None:
            self.stopped = True

    async def run() -> None:
        consumer = StartStopConsumer()
        writer = InfluxWriter()
        lifespan = _create_lifespan(
            CloudSettings(mqtt_enabled=True),
            consumer,
            writer,
        )
        async with lifespan(object()):
            assert consumer.started is True
            assert consumer.event_loop is asyncio.get_running_loop()
        assert consumer.stopped is True

    asyncio.run(run())


def test_influx_query_alerts_builds_alarm_query() -> None:
    class QueryApi:
        def __init__(self) -> None:
            self.calls = []

        def query(self, query, org):
            self.calls.append((query, org))
            return []

    query_api = QueryApi()
    writer = InfluxDBWriter(
        url="http://unused",
        org="haiyu",
        bucket="haiyu",
        write_api=object(),
        client=None,
    )
    writer._query_api = query_api

    assert writer.query_alerts(start="-10m", level="high", limit=5) == []

    query, org = query_api.calls[0]
    assert org == "haiyu"
    assert 'r["_measurement"] == "alarm_event"' in query
    assert 'r["level"] == "high"' in query
    assert "|> limit(n: 5)" in query


def test_influx_query_alerts_rejects_unknown_level() -> None:
    class QueryApi:
        def __init__(self) -> None:
            self.calls = []

        def query(self, query, org):
            self.calls.append((query, org))
            return []

    query_api = QueryApi()
    writer = InfluxDBWriter(
        url="http://unused",
        org="haiyu",
        bucket="haiyu",
        write_api=object(),
        client=None,
    )
    writer._query_api = query_api

    try:
        writer.query_alerts(start="-10m", level='high" or true')
    except ValueError as exc:
        assert "alert level" in str(exc)
    else:
        raise AssertionError("unknown alert level should be rejected")

    assert query_api.calls == []


def test_in_memory_writer_queries_written_alert_and_track_events() -> None:
    writer = InfluxWriter()
    alarm_event = EdgeEvent.from_dict(_event_dict(seq=91))
    track_event = EdgeEvent.from_dict(
        _event_dict(
            seq=92,
            event_type="track",
            payload={
                "track_id": "T-1",
                "x": 12.0,
                "y": 5.0,
                "confidence": 0.82,
            },
        )
    )
    sensor_event = EdgeEvent.from_dict(
        _event_dict(
            seq=93,
            event_type="sensor",
            payload={"sensor_type": "meteo", "wind_speed_mps": 4.2},
        )
    )

    writer.write_event(alarm_event)
    writer.write_event(track_event)
    writer.write_event(sensor_event)

    assert writer.query_alerts(level="high")[0]["seq"] == 91
    track = writer.query_tracks(device_id="edge-01")[0]
    assert track["track_id"] == "T-1"
    assert track["x"] == 12.0
    sensor = writer.query_sensors(sensor_type="meteo")[0]
    assert sensor["sensor_type"] == "meteo"
    assert sensor["payload"]["wind_speed_mps"] == 4.2


def test_contract_rest_routes_use_ingested_events_only() -> None:
    async def seed(consumer: EdgeMqttConsumer) -> None:
        await consumer.accept_event(
            _event_dict(
                seq=101,
                event_type="status",
                payload={"online": True, "cpu": 0.4, "mem": 0.5, "npu": 0.7},
            )
        )
        await consumer.accept_event(
            _event_dict(
                seq=102,
                event_type="track",
                payload={"track_id": "T-2", "x": 3.0, "y": 4.0},
            )
        )
        await consumer.accept_event(_event_dict(seq=103))
        await consumer.accept_event(
            _event_dict(
                seq=104,
                event_type="sensor",
                payload={"sensor_type": "meteo", "wind_speed_mps": 4.2},
            )
        )

    writer = InfluxWriter()
    consumer = EdgeMqttConsumer(
        writer=writer,
        websocket_manager=WebSocketManager(),
    )
    asyncio.run(seed(consumer))
    devices = list(_latest_device_rows(consumer).values())
    assert devices == [
        {
            "device_id": "edge-01",
            "online": True,
            "last_seen": 100.0,
            "last_event_type": "sensor",
            "cpu": 0.4,
            "mem": 0.5,
            "npu": 0.7,
            "source": "mqtt_ingestion",
        }
    ]
    tracks = consumer.writer.query_tracks(device_id="edge-01", start="-1h")
    assert tracks[0]["track_id"] == "T-2"
    summary = _summary_payload(consumer)
    assert summary["device_count"] == 1
    assert summary["track_event_count"] == 1
    assert summary["sensor_event_count"] == 1
    calib = _calibration_ack_payload(
        "edge-01",
        [{"pixel": [0, 0], "world": [1, 1]}],
    )
    assert calib["status"] == "queued_for_edge_execution"
    assert calib["control_point_count"] == 1


def test_websocket_manager_tracks_connections_and_broadcasts() -> None:
    class DummySocket:
        def __init__(self) -> None:
            self.accepted = False
            self.messages = []

        async def accept(self) -> None:
            self.accepted = True

        async def send_json(self, message) -> None:
            self.messages.append(message)

    async def run() -> None:
        manager = WebSocketManager()
        socket = DummySocket()

        await manager.connect(socket)
        await manager.broadcast({"hello": "world"})
        manager.disconnect(socket)

        assert socket.accepted is True
        assert socket.messages == [{"hello": "world"}]
        assert manager.active_connections == []

    asyncio.run(run())


def test_websocket_manager_enforces_connection_limit_on_connect() -> None:
    class DummySocket:
        def __init__(self) -> None:
            self.accepted = False

        async def accept(self) -> None:
            self.accepted = True

    async def run() -> None:
        manager = WebSocketManager()
        first = DummySocket()
        second = DummySocket()

        first_connected = await manager.connect(first, max_connections=1)
        second_connected = await manager.connect(second, max_connections=1)

        assert first_connected is True
        assert second_connected is False
        assert first.accepted is True
        assert second.accepted is False
        assert manager.active_connections == [first]

    asyncio.run(run())
