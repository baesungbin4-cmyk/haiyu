from edge.cache.file_cache import OfflineEventCache
from edge.common.schemas import MqttEventEnvelope
from edge.mqtt_gateway import MqttGateway


class PublishInfo:
    rc = 0


class MockMqttClient:
    def __init__(self) -> None:
        self.published = []

    def connect(self, host, port, keepalive) -> None:
        return None

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
        return PublishInfo()

    def subscribe(self, topic, qos=1):
        return None

    def will_set(self, topic, payload, qos, retain) -> None:
        return None


def _event(seq: int) -> MqttEventEnvelope:
    return MqttEventEnvelope(
        device_id="edge-01",
        timestamp=100.0 + seq,
        seq=seq,
        event_type="track",
        topic="haiyu/edge-01/track",
        payload={"trackId": "t-1", "x": seq * 1.0, "y": 2.0},
        qos=1,
        retain=False,
    )


def test_offline_cache_resends_after_mock_reconnect(tmp_path) -> None:
    cache = OfflineEventCache(tmp_path)
    client = MockMqttClient()
    gateway = MqttGateway(cache=cache, client=client)

    assert gateway.publish_event(_event(1)) is False
    assert [event.seq for event in cache.read_all()] == [1]
    assert client.published == []

    gateway.connected = True
    sent = gateway.flush_cache()

    assert [event.seq for event in sent] == [1]
    assert cache.read_all() == []
    assert len(client.published) == 1
    assert client.published[0][0] == "haiyu/edge-01/track"
