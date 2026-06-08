import asyncio
import json

from cloud.app.schemas import EdgeEvent
from cloud.app.websocket.manager import WebSocketManager
from edge.cache.file_cache import OfflineEventCache
from edge.common.schemas import MqttEventEnvelope
from edge.mqtt_gateway import MqttGateway


class PublishInfo:
    rc = 0


class MockMqttClient:
    def __init__(self) -> None:
        self.published = []
        self.will = None

    def connect(self, host, port, keepalive) -> None:
        return None

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
        return PublishInfo()

    def subscribe(self, topic, qos=1):
        return None

    def will_set(self, topic, payload, qos, retain) -> None:
        self.will = (topic, payload, qos, retain)


class DummySocket:
    def __init__(self) -> None:
        self.accepted = False
        self.messages = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, message) -> None:
        self.messages.append(message)


def _event(seq: int = 1) -> MqttEventEnvelope:
    return MqttEventEnvelope(
        device_id="edge-01",
        timestamp=100.0 + seq,
        seq=seq,
        event_type="alarm",
        topic="haiyu/edge-01/alarm",
        payload={"risk_level": "high", "tcpa": 12.0, "cpa": 35.0},
        qos=1,
        retain=False,
    )


def test_edge_event_reaches_cloud_schema_and_websocket(tmp_path) -> None:
    client = MockMqttClient()
    gateway = MqttGateway(
        cache=OfflineEventCache(tmp_path),
        client=client,
    )
    gateway.connected = True

    assert gateway.publish_event(_event()) is True

    topic, payload, qos, retain = client.published[0]
    assert topic == "haiyu/edge-01/alarm"
    assert qos == 1
    assert retain is False

    cloud_event = EdgeEvent.from_dict(json.loads(payload))
    assert cloud_event.device_id == "edge-01"
    assert cloud_event.event_type == "alarm"

    async def run() -> None:
        manager = WebSocketManager()
        socket = DummySocket()
        await manager.connect(socket)
        await manager.broadcast(
            {"type": cloud_event.event_type, "data": cloud_event.to_dict()}
        )
        assert socket.accepted is True
        assert socket.messages[0]["type"] == "alarm"

    asyncio.run(run())
