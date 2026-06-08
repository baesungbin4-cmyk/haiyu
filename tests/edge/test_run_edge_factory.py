"""Smoke tests for run_edge.py factory function (review S1/S2)."""

import sys
from pathlib import Path

import pytest


def test_run_edge_cli_dry_run() -> None:
    """Review S1: python run_edge.py --dry-run must exit cleanly."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "run_edge.py", "--dry-run"],
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
        [
            sys.executable,
            "run_edge.py",
            "--dry-run",
            "--use-residual-attention",
            "--attention-reduction",
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Pipeline assembled" in result.stdout or "dry_run" in result.stdout


@pytest.mark.skipif(
    not Path("weights/yolov8n_haiyu.pt").exists() and not Path("yolov8n.pt").exists(),
    reason="No YOLO weights available for factory construction test",
)
def test_create_edge_pipeline_with_real_weights() -> None:
    """If weights exist, factory must not crash."""
    from run_edge import EdgePipelineConfig, create_edge_pipeline

    weights = "weights/yolov8n_haiyu.pt"
    if not Path(weights).exists():
        weights = "yolov8n.pt"

    config = EdgePipelineConfig(
        weights=weights,
        use_residual_attention=False,
        attention_reduction=16,
        attention_min_hidden=4,
        confidence_threshold=0.5,
        iou_threshold=0.45,
        tracker_iou_threshold=0.3,
        tracker_max_missed=1,
        prediction_history_len=12,
        prediction_future_len=5,
        prediction_hidden_dim=64,
        prediction_num_layers=1,
        prediction_num_heads=4,
        use_world_coordinate=True,
        tcpa_alert_s=30.0,
        cpa_alert_m=50.0,
        tcpa_critical_s=10.0,
        cpa_critical_m=20.0,
        tcpa_high_s=20.0,
        cpa_high_m=35.0,
        tcpa_low_s=25.0,
        cpa_low_m=40.0,
        device_id="edge-01",
        device="cpu",
        fps=15.0,
        history_len=4,
        dry_run=True,
    )
    processor = create_edge_pipeline(config, mqtt_enabled=False)
    assert processor is not None
    assert processor.config.device_id == "edge-01"
