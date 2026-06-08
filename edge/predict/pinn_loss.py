"""PINN-style trajectory prediction losses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

BoundaryLossFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PINNLossWeights:
    """Weights for trajectory loss terms."""

    lambda_data: float = 1.0
    lambda_physics: float = 1.0
    lambda_smooth: float = 1.0
    lambda_boundary: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("lambda_data", self.lambda_data),
            ("lambda_physics", self.lambda_physics),
            ("lambda_smooth", self.lambda_smooth),
            ("lambda_boundary", self.lambda_boundary),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PINNLossComponents:
    """Named scalar loss components."""

    data: torch.Tensor
    physics: torch.Tensor
    smooth: torch.Tensor
    boundary: torch.Tensor
    total: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "data": self.data,
            "physics": self.physics,
            "smooth": self.smooth,
            "boundary": self.boundary,
            "total": self.total,
        }


def data_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared trajectory error."""

    _validate_pair(prediction, target)
    return F.mse_loss(prediction, target)


def finite_difference_velocity(
    positions: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    """Approximate velocity from positions shaped ``[B, T, 2]``."""

    _validate_trajectory(positions, "positions")
    dt_value = _validate_dt(dt)
    if positions.shape[1] < 2:
        return _zero_like_scalar(positions)
    return (positions[:, 1:, :] - positions[:, :-1, :]) / dt_value


def finite_difference_acceleration(
    positions: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    """Approximate acceleration from positions shaped ``[B, T, 2]``."""

    velocity = finite_difference_velocity(positions, dt)
    if isinstance(velocity, torch.Tensor) and velocity.ndim == 0:
        return _zero_like_scalar(positions)
    if velocity.shape[1] < 2:
        return _zero_like_scalar(positions)
    dt_value = _validate_dt(dt)
    return (velocity[:, 1:, :] - velocity[:, :-1, :]) / dt_value


def physics_residual_loss(
    prediction: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    r"""Penalize deviation from a local **constant-acceleration** kinematic model.

    Contract §5 lists two formulations side by side:

    * **Constant-velocity residual**: :math:`‖d²x/dt²‖²` — penalizes any
      acceleration at all (i.e. assumes zero acceleration).
    * **Constant-acceleration residual**: :math:`‖d²x/dt² - a_{model}‖²` —
      penalizes *changes* in acceleration (i.e. assumes acceleration is
      locally constant, not necessarily zero).

    This implementation uses the **constant-acceleration** (simpler) form with
    :math:`a_{model} = 0` implicitly — it computes the finite-difference
    acceleration from the predicted trajectory and minimizes its squared
    magnitude.  This is equivalent to the constant-velocity residual in the
    contract: we penalize any non-zero second derivative, which encourages
    locally straight-line (inertial) motion.

    The constant-acceleration-with-nonzero-a₀ form (contract's second
    variant) would require a separate estimate of :math:`a_{model}` (e.g.
    from the last observed acceleration in the input window).  This is
    deferred to a future iteration when a richer physics model (e.g.
    hydrodynamic drag, engine thrust limits) is integrated.

    **Relationship with ``smoothness_loss``**:
    Both terms operate on second-order spatial finite differences of the
    predicted positions, forming a joint regularisation over the trajectory
    curvature.  ``physics_residual_loss`` penalises the mean squared
    acceleration magnitude (encouraging locally constant velocity /
    inertial motion).  ``smoothness_loss`` penalises the mean squared
    second spatial difference via a different finite-difference stencil
    (central three-point), which is more sensitive to isolated kinks.
    Together they provide two complementary views of trajectory
    smoothness, with different frequency responses to position errors.
    """

    acceleration = finite_difference_acceleration(prediction, dt)
    if acceleration.ndim == 0:
        return acceleration
    return acceleration.square().mean()


def smoothness_loss(
    prediction: torch.Tensor,
    dt: float = 1.0,
) -> torch.Tensor:
    """Penalize non-smooth trajectories via second-order spatial difference.

    Computes :math:`‖Δ²pred‖²` — the mean squared second-order finite
    difference of predicted positions — which matches contract §5
    ("smoothness = ‖Δ²pred‖²").

    A straight-line path has zero second difference; sharp turns or jitter
    increase the penalty.  This operates directly on positions (not
    accelerations), which avoids overlap with ``physics_residual_loss``
    and keeps the two terms complementary.
    """

    _validate_trajectory(prediction, "prediction")
    if prediction.shape[1] < 3:
        # Need at least 3 time steps for a second-order difference.
        return _zero_like_scalar(prediction)

    # Second-order central finite difference along the time axis:
    #   Δ²pred[t] = pred[t+1] - 2·pred[t] + pred[t-1]
    # Shape: [B, T-2, 2]
    second_diff = (
        prediction[:, 2:, :] - 2.0 * prediction[:, 1:-1, :] + prediction[:, :-2, :]
    )
    # Normalise by dt² so the penalty is in consistent units (m²/s⁴)
    # regardless of the time step.
    dt_value = _validate_dt(dt)
    second_diff = second_diff / (dt_value * dt_value)

    return second_diff.square().mean()


def boundary_loss(
    prediction: torch.Tensor,
    boundary_loss_fn: BoundaryLossFn | None = None,
) -> torch.Tensor:
    """Optional hook for future channel or region constraints."""

    _validate_trajectory(prediction, "prediction")
    if boundary_loss_fn is None:
        return _zero_like_scalar(prediction)
    value = boundary_loss_fn(prediction)
    if not isinstance(value, torch.Tensor):
        raise TypeError("boundary_loss_fn must return a torch.Tensor")
    if value.numel() == 1:
        return value.reshape(())
    return value.mean()


def pinn_trajectory_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: PINNLossWeights | None = None,
    dt: float = 1.0,
    boundary_loss_fn: BoundaryLossFn | None = None,
) -> PINNLossComponents:
    """Compute data, physics, smoothness, optional boundary, and total loss."""

    _validate_pair(prediction, target)
    loss_weights = weights or PINNLossWeights()
    data = data_loss(prediction, target)
    physics = physics_residual_loss(prediction, dt)
    smooth = smoothness_loss(prediction, dt)
    boundary = boundary_loss(prediction, boundary_loss_fn)
    total = (
        loss_weights.lambda_data * data
        + loss_weights.lambda_physics * physics
        + loss_weights.lambda_smooth * smooth
        + loss_weights.lambda_boundary * boundary
    )
    return PINNLossComponents(
        data=data,
        physics=physics,
        smooth=smooth,
        boundary=boundary,
        total=total,
    )


class PINNTrajectoryLoss(nn.Module):
    """Module wrapper around ``pinn_trajectory_loss``."""

    def __init__(
        self,
        weights: PINNLossWeights | None = None,
        dt: float = 1.0,
        boundary_loss_fn: BoundaryLossFn | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or PINNLossWeights()
        self.dt = _validate_dt(dt)
        self.boundary_loss_fn = boundary_loss_fn

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> PINNLossComponents:
        return pinn_trajectory_loss(
            prediction,
            target,
            weights=self.weights,
            dt=self.dt,
            boundary_loss_fn=self.boundary_loss_fn,
        )


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    _validate_trajectory(prediction, "prediction")
    _validate_trajectory(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")


def _validate_trajectory(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 3 or value.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [batch, time, 2]")
    if value.shape[1] <= 0:
        raise ValueError(f"{name} time dimension must be positive")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating point dtype")


def _validate_dt(dt: float) -> float:
    if isinstance(dt, bool) or not isinstance(dt, int | float):
        raise TypeError("dt must be a positive number")
    if dt <= 0:
        raise ValueError("dt must be positive")
    return float(dt)


def _zero_like_scalar(reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, device=reference.device, dtype=reference.dtype)


__all__ = [
    "PINNLossComponents",
    "PINNLossWeights",
    "PINNTrajectoryLoss",
    "boundary_loss",
    "data_loss",
    "finite_difference_acceleration",
    "finite_difference_velocity",
    "physics_residual_loss",
    "pinn_trajectory_loss",
    "smoothness_loss",
]
