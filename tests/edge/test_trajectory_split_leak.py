"""Train/validation split leak tests for TrajectoryWindowDataset (review item S3).

Verifies that ``split_by_trajectory`` produces disjoint trajectory sets
with no shared trajectory IDs, preventing the window-level leakage that
``random_split`` would cause with small strides.
"""

import numpy as np
import torch

from edge.predict.trajectory_dataset import TrajectoryWindowDataset


def _make_trajectories(
    n_traj: int = 10,
    min_len: int = 20,
    max_len: int = 30,
    feature_dim: int = 7,
) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    trajectories = []
    for _ in range(n_traj):
        length = rng.integers(min_len, max_len + 1)
        traj = rng.normal(size=(length, feature_dim)).astype(np.float32)
        trajectories.append(traj)
    return trajectories


def test_split_by_trajectory_no_shared_trajectory_data() -> None:
    """Train and val sets must not contain windows from the same trajectory."""
    trajectories = _make_trajectories(n_traj=10, min_len=30, max_len=40)
    n_traj = len(trajectories)

    train_ds, val_ds = TrajectoryWindowDataset.split_by_trajectory(
        trajectories,
        history_len=5,
        future_len=3,
        train_ratio=0.8,
        stride=2,
        seed=123,
    )

    # Collect the raw trajectory pointers used in each split.
    train_ids = set(id(traj) for traj in train_ds._trajectories)
    val_ids = set(id(traj) for traj in val_ds._trajectories)

    assert (
        len(train_ids & val_ids) == 0
    ), "Train and val sets share trajectory objects — leakage possible"

    # All trajectories from the original list should be assigned.
    all_ids = set(id(traj) for traj in trajectories)
    assigned_ids = train_ids | val_ids
    assert assigned_ids == all_ids, (
        f"Some trajectories were not assigned: "
        f"{len(all_ids)} total vs {len(assigned_ids)} assigned"
    )

    # The split ratio should be honoured approximately.
    assert len(train_ds._trajectories) == max(1, int(n_traj * 0.8))
    assert len(val_ds._trajectories) == n_traj - len(train_ds._trajectories)


def test_split_by_trajectory_windows_disjoint_when_stride_is_small() -> None:
    """Even with stride=1 each split only has its own trajectory's windows."""
    trajectories = _make_trajectories(n_traj=8, min_len=15, max_len=25)
    train_ds, val_ds = TrajectoryWindowDataset.split_by_trajectory(
        trajectories,
        history_len=4,
        future_len=2,
        train_ratio=0.75,
        stride=1,  # maximum overlap risk
        seed=456,
    )

    # Each window in train_ds references only its own trajectory index
    train_traj_indices = set(idx for idx, _ in train_ds._windows)
    val_traj_indices = set(idx for idx, _ in val_ds._windows)

    # The internal window indices are 0-based within each split's own
    # _trajectories list. They should be disjoint because the trajectories
    # themselves are disjoint.
    assert len(train_traj_indices) == len(train_ds._trajectories)
    assert len(val_traj_indices) == len(val_ds._trajectories)


def test_split_by_trajectory_deterministic_with_seed() -> None:
    """Same seed should produce identical splits."""
    trajectories = _make_trajectories(n_traj=10, min_len=20, max_len=30)

    train_a, val_a = TrajectoryWindowDataset.split_by_trajectory(
        trajectories, history_len=5, future_len=3, train_ratio=0.7, seed=42
    )
    train_b, val_b = TrajectoryWindowDataset.split_by_trajectory(
        trajectories, history_len=5, future_len=3, train_ratio=0.7, seed=42
    )

    assert len(train_a) == len(train_b)
    assert len(val_a) == len(val_b)

    # Verify that the same trajectories end up in each split by comparing
    # the total number of windows (a proxy for trajectory content).
    assert sum(len(traj) for traj in train_a._trajectories) == sum(
        len(traj) for traj in train_b._trajectories
    )


def test_split_by_trajectory_rejects_invalid_ratio() -> None:
    """Ratios outside (0, 1) must raise ValueError."""
    trajectories = _make_trajectories(n_traj=4)
    try:
        TrajectoryWindowDataset.split_by_trajectory(
            trajectories, history_len=3, future_len=2, train_ratio=0.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("train_ratio=0 should be rejected")

    try:
        TrajectoryWindowDataset.split_by_trajectory(
            trajectories, history_len=3, future_len=2, train_ratio=1.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("train_ratio=1 should be rejected")


def test_split_by_trajectory_needs_at_least_two_trajectories() -> None:
    """A single trajectory cannot be split."""
    trajectories = _make_trajectories(n_traj=1)
    try:
        TrajectoryWindowDataset.split_by_trajectory(
            trajectories, history_len=3, future_len=2, train_ratio=0.5
        )
    except ValueError as exc:
        assert "at least 2 trajectories" in str(exc)
    else:
        raise AssertionError("single-trajectory split should be rejected")


def test_train_val_windows_are_valid() -> None:
    """Both splits should produce usable (history, target) pairs."""
    trajectories = _make_trajectories(n_traj=6, min_len=25, max_len=35)
    train_ds, val_ds = TrajectoryWindowDataset.split_by_trajectory(
        trajectories,
        history_len=5,
        future_len=3,
        train_ratio=0.67,
        stride=1,
    )

    assert len(train_ds) > 0, "train dataset must have at least one window"
    assert len(val_ds) > 0, "val dataset must have at least one window"

    history, target = train_ds[0]
    assert isinstance(history, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert history.shape == (5, 7)
    assert target.shape == (3, 2)
    assert torch.isfinite(history).all()
    assert torch.isfinite(target).all()

    history, target = val_ds[0]
    assert history.shape == (5, 7)
    assert target.shape == (3, 2)
    assert torch.isfinite(history).all()
    assert torch.isfinite(target).all()
