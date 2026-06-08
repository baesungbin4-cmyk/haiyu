import torch
import pytest

from edge.predict.pinn_loss import (
    PINNLossWeights,
    PINNTrajectoryLoss,
    boundary_loss,
    data_loss,
    physics_residual_loss,
    pinn_trajectory_loss,
    smoothness_loss,
)


def _constant_velocity_trajectory(
    batch_size: int = 2,
    steps: int = 6,
) -> torch.Tensor:
    time = torch.arange(steps, dtype=torch.float32)
    x = 2.0 + 1.5 * time
    y = -3.0 + 0.25 * time
    trajectory = torch.stack((x, y), dim=-1)
    return trajectory.unsqueeze(0).repeat(batch_size, 1, 1)


def test_each_loss_component_is_non_negative() -> None:
    prediction = torch.randn(2, 6, 2)
    target = torch.zeros_like(prediction)

    components = pinn_trajectory_loss(prediction, target)

    for value in components.as_dict().values():
        assert value.ndim == 0
        assert torch.isfinite(value)
        assert value.item() >= 0.0


def test_total_loss_is_scalar() -> None:
    prediction = torch.randn(3, 5, 2)
    target = torch.zeros_like(prediction)

    components = pinn_trajectory_loss(prediction, target)

    assert components.total.shape == torch.Size([])


def test_backward_pass_works() -> None:
    prediction = torch.randn(2, 6, 2, requires_grad=True)
    target = torch.zeros_like(prediction)

    components = PINNTrajectoryLoss()(prediction, target)
    components.total.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_lambda_weights_affect_total_loss() -> None:
    prediction = torch.randn(2, 7, 2)
    target = torch.zeros_like(prediction)

    low_weight = pinn_trajectory_loss(
        prediction,
        target,
        weights=PINNLossWeights(lambda_physics=0.0, lambda_smooth=0.0),
    )
    high_weight = pinn_trajectory_loss(
        prediction,
        target,
        weights=PINNLossWeights(lambda_physics=5.0, lambda_smooth=3.0),
    )

    assert high_weight.total > low_weight.total


def test_lambda_data_affects_total_loss() -> None:
    prediction = torch.ones(2, 5, 2)
    target = torch.zeros_like(prediction)

    half_data = pinn_trajectory_loss(
        prediction,
        target,
        weights=PINNLossWeights(
            lambda_data=0.5,
            lambda_physics=0.0,
            lambda_smooth=0.0,
        ),
    )
    double_data = pinn_trajectory_loss(
        prediction,
        target,
        weights=PINNLossWeights(
            lambda_data=2.0,
            lambda_physics=0.0,
            lambda_smooth=0.0,
        ),
    )

    assert half_data.physics.item() == 0.0
    assert half_data.smooth.item() == 0.0
    assert torch.isclose(double_data.total, half_data.total * 4.0)


def test_validate_trajectory_rejects_zero_time_dimension() -> None:
    empty_prediction = torch.empty(2, 0, 2)
    empty_target = torch.empty(2, 0, 2)

    with pytest.raises(ValueError, match="time dimension"):
        pinn_trajectory_loss(empty_prediction, empty_target)


def test_constant_velocity_has_lower_physics_residual() -> None:
    constant = _constant_velocity_trajectory()
    noise = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, -0.2],
            [-0.8, 0.4],
            [0.9, -0.7],
            [-0.4, 0.2],
            [0.7, -0.5],
        ],
        dtype=torch.float32,
    )
    noisy = constant + noise.unsqueeze(0)

    constant_loss = physics_residual_loss(constant)
    noisy_loss = physics_residual_loss(noisy)

    assert torch.isclose(constant_loss, torch.tensor(0.0), atol=1e-6)
    assert noisy_loss > constant_loss


def test_constant_acceleration_physics_residual_is_nonzero() -> None:
    time = torch.arange(6, dtype=torch.float32)
    trajectory = torch.stack((0.5 * time.square(), time), dim=-1)
    trajectory = trajectory.unsqueeze(0)

    loss = physics_residual_loss(trajectory)

    assert loss > 0


def test_pinn_loss_weights_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="lambda_data"):
        PINNLossWeights(lambda_data=-0.5)


def test_physics_loss_with_non_unit_dt_keeps_constant_velocity_zero() -> None:
    constant = _constant_velocity_trajectory()

    loss = physics_residual_loss(constant, dt=0.033)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_dt_scales_physics_and_smoothness() -> None:
    prediction = _constant_velocity_trajectory(steps=6)
    prediction[:, 3, :] += torch.tensor([1.0, -0.5])

    physics_dt_1 = physics_residual_loss(prediction, dt=1.0)
    physics_dt_small = physics_residual_loss(prediction, dt=0.1)
    smooth_dt_1 = smoothness_loss(prediction, dt=1.0)
    smooth_dt_small = smoothness_loss(prediction, dt=0.1)

    assert physics_dt_small > physics_dt_1 * 1000.0
    # smoothness_loss is second-order (Δ²pos/dt²), so scales as 1/dt⁴.
    # dt=0.1 → factor=10000; dt=1.0 → factor=1.  Must be >~5000× larger.
    assert smooth_dt_small > smooth_dt_1 * 5000.0


def test_short_trajectories_return_safe_zero_scalars() -> None:
    huge = torch.full((1, 1, 2), 1e38)

    physics = physics_residual_loss(huge)
    smooth = smoothness_loss(huge)
    boundary = boundary_loss(huge)

    assert physics.item() == 0.0
    assert smooth.item() == 0.0
    assert boundary.item() == 0.0
    assert torch.isfinite(physics)
    assert torch.isfinite(smooth)
    assert torch.isfinite(boundary)


def test_boundary_loss_hook_is_disabled_by_default() -> None:
    prediction = torch.randn(2, 5, 2)

    assert boundary_loss(prediction).item() == 0.0


def test_boundary_weight_uses_optional_hook() -> None:
    prediction = torch.ones(1, 5, 2)
    target = torch.zeros_like(prediction)

    without_boundary = pinn_trajectory_loss(
        prediction,
        target,
        weights=PINNLossWeights(lambda_boundary=0.0),
        boundary_loss_fn=lambda value: value.abs().mean(),
    )
    with_boundary = pinn_trajectory_loss(
        prediction,
        target,
        weights=PINNLossWeights(lambda_boundary=2.0),
        boundary_loss_fn=lambda value: value.abs().mean(),
    )

    assert with_boundary.boundary > 0
    assert with_boundary.total > without_boundary.total


def test_boundary_loss_fn_returning_non_scalar_is_averaged() -> None:
    prediction = torch.ones(2, 5, 2)

    loss = boundary_loss(prediction, lambda value: value.square())

    assert loss.shape == torch.Size([])
    assert loss.item() == 1.0


def test_data_physics_and_smoothness_functions_are_differentiable() -> None:
    prediction = torch.randn(2, 6, 2, requires_grad=True)
    target = torch.zeros_like(prediction)

    total = (
        data_loss(prediction, target)
        + physics_residual_loss(prediction)
        + smoothness_loss(prediction)
    )
    total.backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
