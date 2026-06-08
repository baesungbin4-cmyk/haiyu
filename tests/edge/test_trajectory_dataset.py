import numpy as np
import pytest
import torch

from edge.common.schemas import TrackHistory, TrackState
from edge.predict.trajectory_dataset import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_TARGET_COLUMNS,
    TrajectoryWindowDataset,
    track_history_to_feature_array,
)


def _state(
    timestamp: float,
    pixel_x: float,
    pixel_y: float,
    velocity_x: float,
    velocity_y: float,
    world_x: float | None = None,
    world_y: float | None = None,
) -> TrackState:
    world_coordinate = None
    if world_x is not None and world_y is not None:
        world_coordinate = (world_x, world_y)
    return TrackState(
        device_id="edge-01",
        track_id=1,
        timestamp=timestamp,
        bbox=(pixel_x - 1, pixel_y - 1, pixel_x + 1, pixel_y + 1),
        confidence=0.9,
        class_name="vessel",
        pixel_coordinate=(pixel_x, pixel_y),
        velocity=(velocity_x, velocity_y),
        world_coordinate=world_coordinate,
    )


def test_dataset_builds_sliding_windows_from_single_trajectory() -> None:
    trajectory = np.arange(10 * DEFAULT_FEATURE_DIM, dtype=np.float32)
    trajectory = trajectory.reshape(10, DEFAULT_FEATURE_DIM)

    dataset = TrajectoryWindowDataset(
        trajectory,
        history_len=4,
        future_len=2,
    )
    history, future = dataset[1]

    assert len(dataset) == 5
    assert history.shape == (4, DEFAULT_FEATURE_DIM)
    assert future.shape == (2, 2)
    assert torch.equal(history, torch.as_tensor(trajectory[1:5]))
    assert torch.equal(future, torch.as_tensor(trajectory[5:7, :2]))


def test_dataset_default_target_columns_are_x_y() -> None:
    trajectory = np.zeros((6, DEFAULT_FEATURE_DIM), dtype=np.float32)
    trajectory[:, 0] = np.arange(10, 16, dtype=np.float32)
    trajectory[:, 1] = np.arange(20, 26, dtype=np.float32)
    trajectory[:, 2] = np.arange(30, 36, dtype=np.float32)
    trajectory[:, 3] = np.arange(40, 46, dtype=np.float32)

    dataset = TrajectoryWindowDataset(
        trajectory,
        history_len=3,
        future_len=2,
    )
    _, future = dataset[0]

    assert DEFAULT_TARGET_COLUMNS == (0, 1)
    assert torch.equal(future, torch.as_tensor(trajectory[3:5, :2]))


def test_dataset_supports_multiple_trajectories_and_stride() -> None:
    first = np.zeros((8, DEFAULT_FEATURE_DIM), dtype=np.float32)
    second = np.ones((8, DEFAULT_FEATURE_DIM), dtype=np.float32)

    dataset = TrajectoryWindowDataset(
        [first, second],
        history_len=3,
        future_len=2,
        stride=2,
    )

    assert len(dataset) == 4
    history, future = dataset[-1]
    assert torch.equal(history, torch.ones(3, DEFAULT_FEATURE_DIM))
    assert torch.equal(future, torch.ones(2, 2))


def test_dataset_stride_no_overlap_between_history_and_future() -> None:
    trajectory = np.zeros((12, DEFAULT_FEATURE_DIM), dtype=np.float32)
    trajectory[:, 0] = np.arange(12, dtype=np.float32)
    trajectory[:, 1] = np.arange(100, 112, dtype=np.float32)
    dataset = TrajectoryWindowDataset(
        trajectory,
        history_len=4,
        future_len=3,
        stride=2,
    )

    for index in range(len(dataset)):
        history, future = dataset[index]
        last_history_time = history[-1, 0].item()
        first_future_time = future[0, 0].item()

        assert first_future_time == last_history_time + 1.0


def test_dataset_rejects_non_finite_values_and_short_sequences() -> None:
    bad = np.zeros((6, DEFAULT_FEATURE_DIM), dtype=np.float32)
    bad[2, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        TrajectoryWindowDataset(bad, history_len=3, future_len=2)

    short = np.zeros((3, DEFAULT_FEATURE_DIM), dtype=np.float32)
    with pytest.raises(ValueError, match="no trajectory windows"):
        TrajectoryWindowDataset(short, history_len=3, future_len=2)


def test_track_history_to_feature_array_uses_world_coordinates() -> None:
    history = TrackHistory(
        device_id="edge-01",
        track_id=1,
        timestamp=15.0,
        states=(
            _state(10.0, 10, 20, 1.0, 2.0, 100.0, 200.0),
            _state(12.0, 11, 22, 3.0, 5.0, 103.0, 205.0),
            _state(15.0, 13, 24, 6.0, 2.0, 109.0, 208.0),
        ),
    )

    features = track_history_to_feature_array(history)

    assert features.shape == (3, DEFAULT_FEATURE_DIM)
    assert features[0].tolist() == [100.0, 200.0, 1.0, 2.0, 0.0, 0.0, 0.0]
    assert features[1].tolist() == [103.0, 205.0, 3.0, 5.0, 1.0, 1.5, 2.0]
    assert features[2].tolist() == [109.0, 208.0, 6.0, 2.0, 1.0, -1.0, 3.0]


def test_track_history_to_feature_array_can_fallback_to_pixel_coordinates():
    history = TrackHistory(
        device_id="edge-01",
        track_id=1,
        timestamp=12.0,
        states=(
            _state(10.0, 10, 20, 1.0, 1.0),
            _state(12.0, 12, 23, 2.0, 1.0),
        ),
    )

    features = track_history_to_feature_array(history)

    assert features[:, :2].tolist() == [[10.0, 20.0], [12.0, 23.0]]


def test_feature_array_mixed_world_coordinates_fallbacks_to_pixels() -> None:
    history = TrackHistory(
        device_id="edge-01",
        track_id=1,
        timestamp=12.0,
        states=(
            _state(10.0, 10, 20, 1.0, 1.0, 100.0, 200.0),
            _state(12.0, 12, 23, 2.0, 1.0),
        ),
    )

    features = track_history_to_feature_array(history)

    assert features[:, :2].tolist() == [[10.0, 20.0], [12.0, 23.0]]


def test_track_history_to_feature_array_rejects_empty_states() -> None:
    history = TrackHistory(
        device_id="edge-01",
        track_id=1,
        timestamp=0.0,
        states=(),
    )

    with pytest.raises(ValueError, match="at least one state"):
        track_history_to_feature_array(history)


def test_feature_array_timestamp_monotonic_enforced() -> None:
    history = TrackHistory(
        device_id="edge-01",
        track_id=1,
        timestamp=10.0,
        states=(
            _state(10.0, 10, 20, 1.0, 1.0),
            _state(10.0, 11, 21, 1.0, 1.0),
        ),
    )

    with pytest.raises(ValueError, match="timestamps must be increasing"):
        track_history_to_feature_array(history)


def test_dataset_can_be_built_from_track_histories() -> None:
    history = TrackHistory(
        device_id="edge-01",
        track_id=1,
        timestamp=15.0,
        states=(
            _state(10.0, 10, 20, 1.0, 1.0, 100.0, 200.0),
            _state(11.0, 11, 21, 1.0, 1.0, 101.0, 201.0),
            _state(12.0, 12, 22, 1.0, 1.0, 102.0, 202.0),
            _state(13.0, 13, 23, 1.0, 1.0, 103.0, 203.0),
        ),
    )

    dataset = TrajectoryWindowDataset.from_track_histories(
        [history],
        history_len=2,
        future_len=1,
    )
    history_window, future_window = dataset[0]

    assert history_window.shape == (2, DEFAULT_FEATURE_DIM)
    assert future_window.tolist() == [[102.0, 202.0]]


def test_dataset_from_track_histories_passes_kwargs() -> None:
    history = TrackHistory(
        device_id="edge-01",
        track_id=1,
        timestamp=15.0,
        states=(
            _state(10.0, 10, 20, 1.0, 2.0),
            _state(11.0, 11, 21, 2.0, 3.0),
            _state(12.0, 12, 22, 3.0, 4.0),
            _state(13.0, 13, 23, 4.0, 5.0),
            _state(14.0, 14, 24, 5.0, 6.0),
            _state(15.0, 15, 25, 6.0, 7.0),
        ),
    )

    dataset = TrajectoryWindowDataset.from_track_histories(
        [history],
        history_len=2,
        future_len=1,
        target_columns=(2, 3),
        stride=2,
    )
    _, future_window = dataset[1]

    assert len(dataset) == 2
    assert future_window.tolist() == [[5.0, 6.0]]
