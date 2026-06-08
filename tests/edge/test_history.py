import numpy as np
import torch

from edge.common.schemas import TrackState
from edge.predict.trajectory_dataset import DEFAULT_FEATURE_DIM
from edge.track.history import TrackHistoryBuffer


def _state(
    timestamp: float,
    pixel_coordinate: tuple[float, float],
    velocity: tuple[float, float],
    track_id: int = 1,
    device_id: str = "edge-01",
    world_coordinate: tuple[float, float] | None = None,
) -> TrackState:
    x_value, y_value = pixel_coordinate
    return TrackState(
        device_id=device_id,
        track_id=track_id,
        timestamp=timestamp,
        bbox=(x_value - 1.0, y_value - 1.0, x_value + 1.0, y_value + 1.0),
        confidence=0.9,
        class_name="vessel",
        pixel_coordinate=pixel_coordinate,
        velocity=velocity,
        world_coordinate=world_coordinate,
    )


def test_history_length_is_capped_per_track() -> None:
    buffer = TrackHistoryBuffer(max_length=2)

    buffer.append(_state(1.0, (10.0, 20.0), (1.0, 0.0)))
    buffer.append(_state(2.0, (11.0, 20.0), (1.0, 0.0)))
    buffer.append(_state(3.0, (12.0, 20.0), (1.0, 0.0)))

    history = buffer.get(1, device_id="edge-01")

    assert history is not None
    assert [state.timestamp for state in history.states] == [2.0, 3.0]


def test_history_converts_to_predictor_feature_array() -> None:
    buffer = TrackHistoryBuffer(max_length=4)
    buffer.append(
        _state(
            1.0,
            (10.0, 20.0),
            (1.0, 1.0),
            world_coordinate=(100.0, 200.0),
        )
    )
    buffer.append(
        _state(
            3.0,
            (12.0, 24.0),
            (3.0, 5.0),
            world_coordinate=(106.0, 210.0),
        )
    )

    features = buffer.to_feature_array(1, device_id="edge-01")

    assert features.shape == (2, DEFAULT_FEATURE_DIM)
    assert features.dtype == np.float32
    assert features.tolist() == [
        [100.0, 200.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [106.0, 210.0, 3.0, 5.0, 1.0, 2.0, 2.0],
    ]


def test_history_converts_to_lstm_predictor_tensor() -> None:
    buffer = TrackHistoryBuffer(max_length=4)
    buffer.append(_state(1.0, (10.0, 20.0), (1.0, 0.0)))
    buffer.append(_state(2.0, (11.0, 20.0), (1.0, 0.0)))

    tensor = buffer.to_predictor_tensor(1, device_id="edge-01")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 2, DEFAULT_FEATURE_DIM)
    assert tensor.dtype == torch.float32


def test_history_keeps_same_track_id_on_multiple_devices_separate() -> None:
    buffer = TrackHistoryBuffer(max_length=2)
    buffer.append(_state(1.0, (10.0, 20.0), (1.0, 0.0)))
    buffer.append(
        _state(
            1.0,
            (30.0, 40.0),
            (0.0, 1.0),
            device_id="edge-02",
        )
    )

    first = buffer.get(1, device_id="edge-01")
    second = buffer.get(1, device_id="edge-02")

    assert first is not None
    assert second is not None
    assert first.device_id == "edge-01"
    assert second.device_id == "edge-02"
