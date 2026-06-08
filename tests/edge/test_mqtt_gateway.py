from dataclasses import dataclass

from edge.cache.file_cache import OfflineEventCache
from edge.common.schemas import MqttEventEnvelope
from edge.mqtt_gateway import MqttGateway, MqttGatewayConfig


@dataclass
class _PublishInfo:
    rc: int = 0


class _MockClient:
    def __init__(self, fail_connect: bool = False) -> None:
        self.connected = False
        self.fail_connect = fail_connect
        self.fail_publish = False
        self.published = []
        self.subscriptions = []
        self.will = None
        self.credentials = None

    def connect(self, host: str, port: int, keepalive: int) -> None:
        if self.fail_connect:
            raise RuntimeError("connect failed")
        self.connected = True

    def publish(self, topic: str, payload: str, qos: int, retain: bool):
        if self.fail_publish:
            return _PublishInfo(rc=1)
        self.published.append((topic, payload, qos, retain))
        return _PublishInfo(rc=0)

    def subscribe(self, topic: str, qos: int = 1):
        self.subscriptions.append((topic, qos))
        return (0, 1)

    def will_set(
        self,
        topic: str,
        payload: str,
        qos: int,
        retain: bool,
    ) -> None:
        self.will = (topic, payload, qos, retain)

    def username_pw_set(self, username: str, password: str | None) -> None:
        self.credentials = (username, password)


def _event(seq: int, qos: int = 1) -> MqttEventEnvelope:
    return MqttEventEnvelope(
        device_id="edge-01",
        timestamp=float(seq),
        seq=seq,
        event_type="alarm",
        topic="haiyu/edge-01/alarm",
        payload={"risk_level": "high", "seq": seq},
        qos=qos,
        retain=False,
    )


def test_gateway_configures_lwt_and_authentication_fields(tmp_path) -> None:
    client = _MockClient()

    MqttGateway(
        config=MqttGatewayConfig(
            client_id="edge-01",
            username="user",
            password="secret",
        ),
        cache=OfflineEventCache(tmp_path),
        client=client,
    )

    assert client.will == ("haiyu/edge-01/lwt", "offline", 1, True)
    assert client.credentials == ("user", "secret")


def test_network_failure_mock_writes_to_cache(tmp_path) -> None:
    client = _MockClient()
    cache = OfflineEventCache(tmp_path)
    gateway = MqttGateway(cache=cache, client=client)
    gateway.connected = True
    client.fail_publish = True

    sent = gateway.publish_event(_event(1))

    assert sent is False
    assert [event.seq for event in cache.read_all()] == [1]


def test_reconnect_mock_flushes_cache(tmp_path) -> None:
    client = _MockClient()
    cache = OfflineEventCache(tmp_path)
    gateway = MqttGateway(cache=cache, client=client)
    cache.append(_event(1))
    cache.append(_event(2))

    gateway.reconnect()

    assert [item[2] for item in client.published] == [1, 1]
    assert [event.seq for event in cache.read_all()] == []


def test_flush_cache_skips_already_sent_within_session(tmp_path) -> None:
    client = _MockClient()
    cache = OfflineEventCache(tmp_path)
    gateway = MqttGateway(cache=cache, client=client)
    gateway.connected = True
    event = _event(1)

    gateway.publish_event(event)
    cache.append(event)
    flushed = gateway.flush_cache()

    assert [event.seq for event in flushed] == [1]
    assert len(client.published) == 1
    assert cache.read_all() == []
    assert gateway._sent_keys == set()


def test_offline_publish_caches_and_preserves_sequence(tmp_path) -> None:
    client = _MockClient()
    cache = OfflineEventCache(tmp_path)
    gateway = MqttGateway(cache=cache, client=client)

    gateway.publish_event(_event(3, qos=0))

    cached = cache.read_all()
    assert len(cached) == 1
    assert cached[0].seq == 3
    assert cached[0].timestamp == 3.0
    assert cached[0].qos == 0


def test_backoff_increases_and_caps_at_max(tmp_path) -> None:
    sleeps = []
    client = _MockClient(fail_connect=True)
    gateway = MqttGateway(
        config=MqttGatewayConfig(
            backoff_initial_seconds=1.0,
            backoff_max_seconds=2.0,
        ),
        cache=OfflineEventCache(tmp_path),
        client=client,
        sleep_fn=sleeps.append,
    )

    for _ in range(3):
        try:
            gateway.connect()
        except RuntimeError:
            pass

    assert sleeps == [1.0, 2.0, 2.0]
    assert gateway.connected is False


def test_subscribe_uses_configured_qos(tmp_path) -> None:
    client = _MockClient()
    gateway = MqttGateway(cache=OfflineEventCache(tmp_path), client=client)

    gateway.subscribe("haiyu/edge-01/cmd", qos=1)

    assert client.subscriptions == [("haiyu/edge-01/cmd", 1)]
