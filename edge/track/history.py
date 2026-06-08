"""Fixed-length track history buffers for trajectory prediction."""

from __future__ import annotations

from collections import deque
from numbers import Integral

import numpy as np
import torch

from edge.common.schemas import TrackHistory, TrackId, TrackState
from edge.predict.trajectory_dataset import track_history_to_feature_array

_HistoryKey = tuple[str, TrackId]


class TrackHistoryBuffer:
    """Store recent ``TrackState`` objects per device and track id."""

    def __init__(
        self,
        max_length: int,
        use_world_coordinate: bool = True,
    ) -> None:
        is_bool = isinstance(max_length, bool)
        is_integer = isinstance(max_length, Integral)
        if is_bool or not is_integer:
            raise TypeError("max_length must be a positive integer")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.max_length = int(max_length)
        self.use_world_coordinate = bool(use_world_coordinate)
        self._states: dict[_HistoryKey, deque[TrackState]] = {}

    def append(self, state: TrackState) -> None:
        """Append one state into the bounded per-track buffer."""

        if not isinstance(state, TrackState):
            raise TypeError("state must be a TrackState")

        key = (state.device_id, state.track_id)
        states = self._states.setdefault(key, deque(maxlen=self.max_length))
        if states and state.timestamp <= states[-1].timestamp:
            raise ValueError("state timestamps must be increasing per track")
        states.append(state)

    def add(self, state: TrackState) -> None:
        """Alias for callers that use an event-buffer naming style."""

        self.append(state)

    def extend(
        self,
        states: list[TrackState] | tuple[TrackState, ...],
    ) -> None:
        """Append a batch of states in the supplied order."""

        for state in states:
            self.append(state)

    def get(
        self,
        track_id: TrackId,
        device_id: str | None = None,
    ) -> TrackHistory | None:
        """Return a schema ``TrackHistory`` for a retained track."""

        key = self._resolve_key(track_id, device_id)
        if key is None:
            return None
        return self._to_history(key)

    def all_histories(self) -> list[TrackHistory]:
        """Return histories for all retained tracks in deterministic order."""

        keys = sorted(self._states, key=lambda item: (item[0], str(item[1])))
        return [self._to_history(key) for key in keys]

    def to_feature_array(
        self,
        track_id: TrackId,
        device_id: str | None = None,
        use_world_coordinate: bool | None = None,
    ) -> np.ndarray:
        """Convert one retained history to predictor feature rows."""

        history = self.get(track_id, device_id=device_id)
        if history is None:
            raise KeyError("track history not found")
        if use_world_coordinate is None:
            use_world_coordinate = self.use_world_coordinate
        return track_history_to_feature_array(
            history,
            use_world_coordinate=use_world_coordinate,
        )

    def to_features(
        self,
        track_id: TrackId,
        device_id: str | None = None,
        use_world_coordinate: bool | None = None,
    ) -> np.ndarray:
        """Alias for LSTM input feature extraction."""

        return self.to_feature_array(
            track_id,
            device_id=device_id,
            use_world_coordinate=use_world_coordinate,
        )

    def to_predictor_tensor(
        self,
        track_id: TrackId,
        device_id: str | None = None,
        use_world_coordinate: bool | None = None,
        batch_dim: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a torch tensor shaped as ``[B, T, F]`` by default."""

        features = self.to_feature_array(
            track_id,
            device_id=device_id,
            use_world_coordinate=use_world_coordinate,
        )
        tensor = torch.as_tensor(features, dtype=dtype)
        if batch_dim:
            tensor = tensor.unsqueeze(0)
        return tensor

    def clear(self) -> None:
        """Remove all retained histories."""

        self._states.clear()

    def _resolve_key(
        self,
        track_id: TrackId,
        device_id: str | None,
    ) -> _HistoryKey | None:
        if device_id is not None:
            key = (device_id, track_id)
            if key in self._states:
                return key
            return None

        matches = [key for key in self._states if key[1] == track_id]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError("device_id is required for ambiguous track_id")
        return matches[0]

    def _to_history(self, key: _HistoryKey) -> TrackHistory:
        states = tuple(self._states[key])
        device_id, track_id = key
        timestamp = states[-1].timestamp
        return TrackHistory(
            device_id=device_id,
            track_id=track_id,
            timestamp=timestamp,
            states=states,
        )


__all__ = ["TrackHistoryBuffer"]
