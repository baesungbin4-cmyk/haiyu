"""Verify attention hyperparameters are configurable via CLI (review S3/M1)."""

import sys

from training.detection.train_yolo import build_config, parse_args


def test_attention_params_default_to_cli_defaults() -> None:
    """Default config values must match argparse defaults."""
    sys.argv = ["train_yolo.py", "--dry-run"]
    args = parse_args()
    config = build_config(args)
    assert config.attention_reduction == 16
    assert config.attention_min_hidden == 4


def test_attention_params_can_be_overridden() -> None:
    """--attention-reduction and --attention-min-hidden must be honoured."""
    sys.argv = [
        "train_yolo.py",
        "--dry-run",
        "--attention-reduction",
        "8",
        "--attention-min-hidden",
        "2",
    ]
    args = parse_args()
    config = build_config(args)
    assert config.attention_reduction == 8
    assert config.attention_min_hidden == 2


def test_attention_params_reject_invalid_values() -> None:
    """Non-positive reduction/min_hidden must be rejected."""
    import pytest

    sys.argv = [
        "train_yolo.py",
        "--dry-run",
        "--attention-reduction",
        "0",
    ]
    with pytest.raises(ValueError, match="positive"):
        build_config(parse_args())

    sys.argv = [
        "train_yolo.py",
        "--dry-run",
        "--attention-reduction",
        "-1",
    ]
    with pytest.raises(ValueError, match="positive"):
        build_config(parse_args())

    sys.argv = [
        "train_yolo.py",
        "--dry-run",
        "--attention-min-hidden",
        "0",
    ]
    with pytest.raises(ValueError, match="positive"):
        build_config(parse_args())


def test_run_edge_cli_dry_run() -> None:
    """Review S1: python run_edge.py --dry-run must exit cleanly."""
    import subprocess

    result = subprocess.run(
        ["python", "run_edge.py", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "dry_run" in result.stdout


def test_run_edge_cli_dry_run_with_attention() -> None:
    """Review S2: --dry-run --use-residual-attention must exit cleanly."""
    import subprocess

    result = subprocess.run(
        ["python", "run_edge.py", "--dry-run", "--use-residual-attention"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Pipeline assembled" in result.stdout
