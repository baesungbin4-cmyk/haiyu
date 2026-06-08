"""Bridge ``LSTMCausalAttentionPredictor`` to ``TrajectoryPredictor``.

Review H1/M1: the real LSTM model operates on ``[B, T, F]`` tensors, but
``EdgeProcessor`` expects a ``TrajectoryPredictor`` whose ``predict`` method
takes ``Sequence[TrackState]`` and returns ``TrajectoryPrediction``.  This
adapter closes the integration gap.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from edge.common.schemas import (
    TrackState,
    TrajectoryPoint,
    TrajectoryPrediction,
)

if TYPE_CHECKING:
    from edge.predict.lstm_causal_attention import (
        LSTMCausalAttentionPredictor,
    )


class LSTMTrajectoryPredictorAdapter:
    """Adapt ``LSTMCausalAttentionPredictor`` → ``TrajectoryPredictor`` Protocol.

    Converts a sequence of ``TrackState`` objects into the ``[1, T, 7]``
    feature tensor the model expects, runs inference, and packages the
    output ``[1, T_out, 2]`` tensor into a ``TrajectoryPrediction`` schema
    object.

    Parameters
    ----------
    model:
        A trained ``LSTMCausalAttentionPredictor`` instance.
    history_len:
        Number of recent ``TrackState`` entries the model expects.
        If the supplied history is longer, only the most recent
        *history_len* entries are used.
    use_world_coordinate:
        When *True* and all states have ``world_coordinate``, use
        world (geographic) coordinates; otherwise fall back to pixel.
    device:
        Torch device.  Defaults to CPU.
    """

    def __init__(
        self,
        model: LSTMCausalAttentionPredictor,
        *,
        history_len: int = 12,
        use_world_coordinate: bool = True,
        device: str | None = None,
    ) -> None:
        self._model = model
        self._history_len = history_len
        self._use_world_coordinate = use_world_coordinate
        self._device = device or "cpu"
        self._model.to(self._device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Public API — satisfies TrajectoryPredictor Protocol
    # ------------------------------------------------------------------

    def predict(
        self,
        history: Sequence[TrackState],
        *,
        timestamp: float,
    ) -> TrajectoryPrediction:
        """Run trajectory prediction from tracked state history.

        Parameters
        ----------
        history:
            Time-ordered sequence of ``TrackState`` entries.  Must contain
            at least ``history_len`` entries.
        timestamp:
            Current frame timestamp (forwarded to the output schema).

        Returns
        -------
        TrajectoryPrediction
            Validated schema object with predicted future points.
        """
        if not history:
            raise ValueError("history must contain at least one TrackState")
        if len(history) < self._history_len:
            raise ValueError(
                f"history must contain at least {self._history_len} "
                f"states (got {len(history)})"
            )

        # Use the most recent history_len entries
        recent = tuple(history[-self._history_len :])
        last_state = recent[-1]

        # 1. Sequence[TrackState] → feature array [T, 7]
        features = _track_states_to_feature_array(
            recent,
            use_world_coordinate=self._use_world_coordinate,
        )
        # 2. numpy array → torch tensor [1, T, 7]
        tensor = torch.as_tensor(
            features, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        # 3. model forward → [1, T_out, 2]
        with torch.no_grad():
            prediction_tensor = self._model(tensor)  # [1, future_len, 2]

        # 4. tensor → schema
        future_xy = prediction_tensor.squeeze(0).cpu().numpy()  # [T_out, 2]
        dt = self._infer_dt(recent)
        future_points = _build_trajectory_points(
            device_id=last_state.device_id,
            track_id=last_state.track_id,
            base_timestamp=timestamp,
            dt=dt,
            future_xy=future_xy,
            use_world_coordinate=self._use_world_coordinate,
            last_state=last_state,
        )

        return TrajectoryPrediction(
            device_id=last_state.device_id,
            track_id=last_state.track_id,
            timestamp=timestamp,
            points=tuple(future_points),
            horizon_seconds=dt * len(future_points),
            model_name="lstm-causal-attention",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_dt(states: tuple[TrackState, ...]) -> float:
        """Estimate dt from the last two states, default 0.067 (~15 fps)."""
        if len(states) >= 2 and states[-1].timestamp > states[-2].timestamp:
            return float(states[-1].timestamp - states[-2].timestamp)
        return 0.067  # sensible default


# ---------------------------------------------------------------------------
# Internal conversion helpers
# ---------------------------------------------------------------------------


def _track_states_to_feature_array(
    states: tuple[TrackState, ...],
    *,
    use_world_coordinate: bool = True,
) -> np.ndarray:
    """Convert a tuple of TrackState to a ``[T, 7]`` feature array.

    Feature columns: ``[x, y, vx, vy, ax, ay, dt]``.

    Mirrors ``track_history_to_feature_array`` but accepts a bare
    ``tuple[TrackState, ...]`` instead of requiring a ``TrackHistory``
    wrapper.
    """
    use_world = use_world_coordinate and all(
        s.world_coordinate is not None for s in states
    )
    rows = []
    prev_state: TrackState | None = None
    prev_velocity: tuple[float, float] | None = None

    for state in states:
        if use_world and state.world_coordinate is not None:
            x_value, y_value = state.world_coordinate
        else:
            x_value, y_value = state.pixel_coordinate

        vx_value, vy_value = state.velocity
        if prev_state is None or prev_velocity is None:
            dt_value = 0.0
            ax_value = 0.0
            ay_value = 0.0
        else:
            dt_value = state.timestamp - prev_state.timestamp
            if dt_value <= 0:
                raise ValueError("timestamps must be strictly increasing")
            ax_value = (vx_value - prev_velocity[0]) / dt_value
            ay_value = (vy_value - prev_velocity[1]) / dt_value

        rows.append(
            [x_value, y_value, vx_value, vy_value, ax_value, ay_value, dt_value]
        )
        prev_state = state
        prev_velocity = (vx_value, vy_value)

    return np.asarray(rows, dtype=np.float32)


def _build_trajectory_points(
    *,
    device_id: str,
    track_id: str | int,
    base_timestamp: float,
    dt: float,
    future_xy: np.ndarray,
    use_world_coordinate: bool,
    last_state: TrackState,
) -> list[TrajectoryPoint]:
    """Convert absolute ``[T_out, 2]`` positions into schema points."""
    points = []
    for i in range(future_xy.shape[0]):
        ts = base_timestamp + dt * (i + 1)
        px = float(future_xy[i, 0])
        py = float(future_xy[i, 1])
        if use_world_coordinate and last_state.world_coordinate is not None:
            wc = (px, py)
            pc = last_state.pixel_coordinate  # keep pixel from last known
        else:
            wc = None
            pc = (px, py)
        points.append(
            TrajectoryPoint(
                device_id=device_id,
                track_id=track_id,
                timestamp=ts,
                pixel_coordinate=pc,
                velocity=last_state.velocity,
                world_coordinate=wc,
            )
        )
    return points


__all__ = ["LSTMTrajectoryPredictorAdapter"]
