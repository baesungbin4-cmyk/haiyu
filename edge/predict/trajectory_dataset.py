"""Trajectory window dataset utilities for prediction training data."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import numpy as np
import torch
from torch.utils.data import Dataset

from edge.common.schemas import TrackHistory, TrackState

DEFAULT_FEATURE_DIM = 7
DEFAULT_TARGET_COLUMNS = (0, 1)
TrajectoryArray = np.ndarray | torch.Tensor | Sequence[Sequence[float]]
TrajectoryInput = TrajectoryArray | Sequence[TrajectoryArray]


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _as_float_array(
    sequence: TrajectoryArray,
) -> np.ndarray:
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().numpy()
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("each trajectory must have shape [time, feature]")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("trajectory dimensions must be positive")
    if not np.isfinite(array).all():
        raise ValueError("trajectory values must be finite")
    return array


def _normalize_trajectories(trajectories: TrajectoryInput) -> list[np.ndarray]:
    if isinstance(trajectories, torch.Tensor):
        if trajectories.ndim == 2:
            return [_as_float_array(trajectories)]
        if trajectories.ndim == 3:
            return [_as_float_array(item) for item in trajectories]
        raise ValueError("trajectories tensor must be 2D or 3D")

    if isinstance(trajectories, np.ndarray):
        if trajectories.ndim == 2:
            return [_as_float_array(trajectories)]
        if trajectories.ndim == 3:
            return [_as_float_array(item) for item in trajectories]
        raise ValueError("trajectories array must be 2D or 3D")

    return [_as_float_array(item) for item in trajectories]


def track_history_to_feature_array(
    history: TrackHistory,
    use_world_coordinate: bool = True,
) -> np.ndarray:
    """Convert a ``TrackHistory`` to ``[x, y, vx, vy, ax, ay, dt]`` rows.

    The first row uses zero ``dt`` and zero acceleration because there is no
    previous state in the supplied history. Stateful cross-window inference
    should make that boundary convention explicit instead of stitching windows
    as though their first rows carried physical elapsed time.
    """

    if not isinstance(history, TrackHistory):
        raise TypeError("history must be a TrackHistory")
    states = history.states
    if not states:
        raise ValueError("history must contain at least one state")

    use_world_for_history = use_world_coordinate and all(
        state.world_coordinate is not None for state in states
    )
    rows = []
    previous_state: TrackState | None = None
    previous_velocity: tuple[float, float] | None = None
    for state in states:
        if use_world_for_history and state.world_coordinate is not None:
            x_value, y_value = state.world_coordinate
        else:
            x_value, y_value = state.pixel_coordinate

        vx_value, vy_value = state.velocity
        if previous_state is None or previous_velocity is None:
            dt_value = 0.0
            ax_value = 0.0
            ay_value = 0.0
        else:
            dt_value = state.timestamp - previous_state.timestamp
            if dt_value <= 0:
                raise ValueError("timestamps must be increasing")
            ax_value = (vx_value - previous_velocity[0]) / dt_value
            ay_value = (vy_value - previous_velocity[1]) / dt_value

        rows.append(
            [
                x_value,
                y_value,
                vx_value,
                vy_value,
                ax_value,
                ay_value,
                dt_value,
            ]
        )
        previous_state = state
        previous_velocity = (vx_value, vy_value)

    return np.asarray(rows, dtype=np.float32)


class TrajectoryWindowDataset(Dataset):
    """Build fixed history/future windows from trajectory feature sequences.

    Split train/validation data by trajectory before constructing datasets.
    Randomly splitting windows from the same trajectory can leak near-duplicate
    overlapping samples across splits when ``stride`` is small.
    """

    def __init__(
        self,
        trajectories: TrajectoryInput,
        history_len: int,
        future_len: int,
        target_columns: tuple[int, int] = DEFAULT_TARGET_COLUMNS,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        _validate_positive_int(history_len, "history_len")
        _validate_positive_int(future_len, "future_len")
        _validate_positive_int(stride, "stride")
        if len(target_columns) != 2:
            raise ValueError("target_columns must contain two column indexes")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating point torch dtype")

        self.history_len = int(history_len)
        self.future_len = int(future_len)
        columns = []
        for column in target_columns:
            if isinstance(column, bool) or not isinstance(column, Integral):
                raise TypeError("target columns must be integers")
            columns.append(int(column))
        self.target_columns = tuple(columns)
        self.dtype = dtype
        self._trajectories = _normalize_trajectories(trajectories)
        self._windows: list[tuple[int, int]] = []

        total_len = self.history_len + self.future_len
        for trajectory_index, trajectory in enumerate(self._trajectories):
            feature_dim = trajectory.shape[1]
            for column in self.target_columns:
                if column < 0 or column >= feature_dim:
                    raise ValueError("target column is out of range")
            max_start = trajectory.shape[0] - total_len
            if max_start < 0:
                continue
            for start in range(0, max_start + 1, stride):
                self._windows.append((trajectory_index, start))
        if not self._windows:
            raise ValueError("no trajectory windows can be built")

    @classmethod
    def from_track_histories(
        cls,
        histories: Sequence[TrackHistory],
        history_len: int,
        future_len: int,
        use_world_coordinate: bool = True,
        **kwargs: object,
    ) -> "TrajectoryWindowDataset":
        trajectories = [
            track_history_to_feature_array(history, use_world_coordinate)
            for history in histories
        ]
        return cls(trajectories, history_len, future_len, **kwargs)

    @classmethod
    def split_by_trajectory(
        cls,
        trajectories: TrajectoryInput,
        history_len: int,
        future_len: int,
        *,
        train_ratio: float = 0.8,
        seed: int = 42,
        target_columns: tuple[int, int] = DEFAULT_TARGET_COLUMNS,
        stride: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> tuple["TrajectoryWindowDataset", "TrajectoryWindowDataset"]:
        """Split trajectories into train/val datasets **before** windowing.

        Unlike ``random_split`` on windows, this guarantees that windows from
        the same trajectory never appear in both the training and validation
        sets, preventing leakage through overlapping windows (contract §4 /
        §7 dataset hygiene).

        Parameters
        ----------
        trajectories:
            Same input format accepted by ``TrajectoryWindowDataset.__init__``.
        history_len, future_len:
            Passed to the child datasets.
        train_ratio:
            Fraction of trajectories assigned to the training set.  Must be
            strictly between 0 and 1.
        seed:
            RNG seed for the trajectory shuffle.
        target_columns, stride, dtype:
            Forwarded to the child dataset constructors.

        Returns
        -------
        tuple[TrajectoryWindowDataset, TrajectoryWindowDataset]
            ``(train_dataset, val_dataset)`` built from disjoint trajectory sets.
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1 (exclusive)")

        arrays = _normalize_trajectories(trajectories)
        n_traj = len(arrays)
        if n_traj < 2:
            raise ValueError("need at least 2 trajectories to perform a split")

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_traj).tolist()

        n_train = max(1, min(n_traj - 1, int(n_traj * train_ratio)))
        train_indices = set(indices[:n_train])
        val_indices = set(indices[n_train:])

        train_arrays = [arr for i, arr in enumerate(arrays) if i in train_indices]
        val_arrays = [arr for i, arr in enumerate(arrays) if i in val_indices]

        # Build each split; if one side produces zero windows (because every
        # trajectory is too short), raise early instead of silently returning
        # an empty dataset.
        train_ds = cls(
            train_arrays,
            history_len,
            future_len,
            target_columns=target_columns,
            stride=stride,
            dtype=dtype,
        )
        val_ds = cls(
            val_arrays,
            history_len,
            future_len,
            target_columns=target_columns,
            stride=stride,
            dtype=dtype,
        )
        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_index, start = self._windows[index]
        trajectory = self._trajectories[trajectory_index]
        history_end = start + self.history_len
        future_end = history_end + self.future_len

        history = trajectory[start:history_end]
        future = trajectory[history_end:future_end, self.target_columns]
        return (
            torch.as_tensor(history, dtype=self.dtype),
            torch.as_tensor(future, dtype=self.dtype),
        )


__all__ = [
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_TARGET_COLUMNS",
    "TrajectoryWindowDataset",
    "track_history_to_feature_array",
]
