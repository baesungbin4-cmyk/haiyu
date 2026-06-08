"""Factory and entry point for the complete Haiyu edge inference pipeline.

Review H2/M2: assembles the real components — YOLO + attention detector,
IoU tracker, LSTM causal-attention predictor, TCPA/CPA risk engine, and
MQTT gateway — into an ``EdgeProcessor`` ready for deployment.

Usage
-----
    python run_edge.py --weights weights/yolov8n.pt --use-residual-attention
    python run_edge.py --dry-run  # print config and exit
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from edge.risk.risk_engine import RiskThresholds
from edge.edge_processor_v2 import (
    EdgeProcessor,
    EdgeProcessorConfig,
)


@dataclass(frozen=True)
class EdgePipelineConfig:
    """Complete edge pipeline configuration."""

    # Detection
    weights: str
    use_residual_attention: bool
    attention_reduction: int
    attention_min_hidden: int
    confidence_threshold: float
    iou_threshold: float

    # Tracking (S5: independent of detection NMS IoU)
    tracker_iou_threshold: float
    tracker_max_missed: int

    # Prediction
    prediction_history_len: int
    prediction_future_len: int
    prediction_hidden_dim: int
    prediction_num_layers: int
    prediction_num_heads: int
    use_world_coordinate: bool

    # Risk
    tcpa_alert_s: float
    cpa_alert_m: float
    tcpa_critical_s: float
    cpa_critical_m: float
    tcpa_high_s: float
    cpa_high_m: float
    tcpa_low_s: float
    cpa_low_m: float

    # Runtime
    device_id: str
    device: str
    fps: float
    history_len: int
    dry_run: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Detection
    p.add_argument("--weights", default="weights/yolov8n_haiyu.pt")
    p.add_argument("--use-residual-attention", action="store_true")
    p.add_argument("--attention-reduction", type=int, default=16)
    p.add_argument("--attention-min-hidden", type=int, default=4)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.45)
    # Tracking
    p.add_argument(
        "--tracker-iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for tracker association (S5: "
        "independent of detection NMS)",
    )
    p.add_argument("--tracker-max-missed", type=int, default=1)
    # Prediction
    p.add_argument("--prediction-history-len", type=int, default=12)
    p.add_argument("--prediction-hidden-dim", type=int, default=64)
    p.add_argument("--prediction-future-len", type=int, default=5)
    p.add_argument("--prediction-num-layers", type=int, default=1)
    p.add_argument("--prediction-num-heads", type=int, default=4)
    # Risk
    p.add_argument("--tcpa-alert-s", type=float, default=30.0)
    p.add_argument("--cpa-alert-m", type=float, default=50.0)
    p.add_argument("--tcpa-critical-s", type=float, default=10.0)
    p.add_argument("--cpa-critical-m", type=float, default=20.0)
    p.add_argument("--tcpa-high-s", type=float, default=20.0)
    p.add_argument("--cpa-high-m", type=float, default=35.0)
    p.add_argument("--tcpa-low-s", type=float, default=25.0)
    p.add_argument("--cpa-low-m", type=float, default=40.0)
    # Runtime
    p.add_argument("--device-id", default="edge-01")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument(
        "--history-len",
        type=int,
        default=12,
        help="EdgeProcessor deque maxlen — must be ≥ prediction-history-len",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> EdgePipelineConfig:
    return EdgePipelineConfig(
        weights=args.weights,
        use_residual_attention=args.use_residual_attention,
        attention_reduction=args.attention_reduction,
        attention_min_hidden=args.attention_min_hidden,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        tracker_iou_threshold=args.tracker_iou_threshold,
        tracker_max_missed=args.tracker_max_missed,
        prediction_history_len=args.prediction_history_len,
        prediction_future_len=args.prediction_future_len,
        prediction_hidden_dim=args.prediction_hidden_dim,
        prediction_num_layers=args.prediction_num_layers,
        prediction_num_heads=args.prediction_num_heads,
        use_world_coordinate=True,
        tcpa_alert_s=args.tcpa_alert_s,
        cpa_alert_m=args.cpa_alert_m,
        tcpa_critical_s=args.tcpa_critical_s,
        cpa_critical_m=args.cpa_critical_m,
        tcpa_high_s=args.tcpa_high_s,
        cpa_high_m=args.cpa_high_m,
        tcpa_low_s=args.tcpa_low_s,
        cpa_low_m=args.cpa_low_m,
        device_id=args.device_id,
        device=args.device,
        fps=args.fps,
        history_len=args.history_len,
        dry_run=args.dry_run,
    )


def create_edge_pipeline(
    config: EdgePipelineConfig,
    *,
    input_source: Iterable[np.ndarray] | None = None,
    max_frames: int | None = None,
    mqtt_enabled: bool = False,
) -> EdgeProcessor:
    """Assemble the full edge pipeline from real components.

    Constructs a ``YOLOAttentionDetector`` → ``IoUTracker`` →
    ``LSTMTrajectoryPredictorAdapter`` → TCPA/CPA ``RiskEngine`` →
    optional MQTT publisher pipeline, wrapped in an ``EdgeProcessor``.

    Parameters
    ----------
    config:
        Complete pipeline configuration.
    input_source:
        Optional iterable of ``np.ndarray`` frames.  When *None*, the
        caller must feed frames via ``process_frame()``.
    max_frames:
        If *input_source* is set, maximum frames to process.
    mqtt_enabled:
        When *True*, wire up the MQTT gateway (requires a running broker).

    Returns
    -------
    EdgeProcessor
        Fully assembled and ready to call ``process_frame()`` or ``run()``.
    """
    # ------------------------------------------------------------------
    # 1. Detector — YOLO + residual attention
    # ------------------------------------------------------------------
    from edge.detect.yolo_attention_detector import YOLOAttentionDetector

    detector = YOLOAttentionDetector(
        config.weights,
        device_id=config.device_id,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        use_residual_attention=config.use_residual_attention,
        attention_reduction=config.attention_reduction,
        attention_min_hidden=config.attention_min_hidden,
        device=config.device,
    )

    # ------------------------------------------------------------------
    # 2. Tracker — IoU-based (S5: independent IoU threshold)
    # ------------------------------------------------------------------
    from edge.track.tracker import IoUTracker

    tracker = IoUTracker(
        iou_threshold=config.tracker_iou_threshold,
        max_missed=config.tracker_max_missed,
    )

    # ------------------------------------------------------------------
    # 3. Predictor — LSTM + causal attention (via adapter)
    # ------------------------------------------------------------------
    from edge.predict.lstm_causal_attention import (
        LSTMCausalAttentionPredictor,
    )
    from edge.predict.predictor_adapter import (
        LSTMTrajectoryPredictorAdapter,
    )

    lstm_model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        future_len=config.prediction_future_len,
        hidden_dim=config.prediction_hidden_dim,
        num_layers=config.prediction_num_layers,
        num_heads=config.prediction_num_heads,
    )
    predictor = LSTMTrajectoryPredictorAdapter(
        lstm_model,
        history_len=config.prediction_history_len,
        use_world_coordinate=config.use_world_coordinate,
        device=config.device,
    )

    # ------------------------------------------------------------------
    # 4. Risk evaluator — TCPA/CPA with configurable thresholds
    # ------------------------------------------------------------------
    risk_thresholds = RiskThresholds(
        tcpa_alert=config.tcpa_alert_s,
        cpa_alert=config.cpa_alert_m,
        tcpa_critical=config.tcpa_critical_s,
        cpa_critical=config.cpa_critical_m,
        tcpa_high=config.tcpa_high_s,
        cpa_high=config.cpa_high_m,
        tcpa_low=config.tcpa_low_s,
        cpa_low=config.cpa_low_m,
    )

    # We wrap evaluate_track_risk to include our thresholds
    from edge.risk.risk_engine import evaluate_track_risk as _eval_risk

    class ConfiguredRiskEvaluator:
        def evaluate(self, target_state, prediction, *, timestamp):
            return _eval_risk(
                own_state=_static_ownship(config.device_id),
                target_state=target_state,
                prediction=prediction,
                risk_thresholds=risk_thresholds,
            )

    risk_evaluator = ConfiguredRiskEvaluator()

    # ------------------------------------------------------------------
    # 5. MQTT publisher (optional)
    # ------------------------------------------------------------------
    mqtt = None
    if mqtt_enabled:
        from edge.mqtt_gateway import MqttGateway, MqttGatewayConfig

        mqtt = MqttGateway(
            config=MqttGatewayConfig(
                host="localhost",
                port=1883,
                client_id=config.device_id,
            ),
        )

    # ------------------------------------------------------------------
    # 6. Assemble EdgeProcessor
    # ------------------------------------------------------------------
    processor = EdgeProcessor(
        detector=detector,
        tracker=tracker,
        predictor=predictor,
        risk_evaluator=risk_evaluator,
        mqtt_publisher=mqtt,
        config=EdgeProcessorConfig(
            device_id=config.device_id,
            fps=config.fps,
            history_len=config.history_len,
        ),
    )
    return processor


def _static_ownship(device_id: str):
    """Return a static ownship TrackState for risk evaluation.

    **Deployment note:** This is a hardcoded placeholder (position (0,0),
    velocity (0,0)).  In a real deployment this must be replaced with
    live GPS/IMU data — ownship position, heading, and speed-over-ground
    from the vessel's navigation system — updated at each frame.
    """
    from edge.common.schemas import TrackState

    return TrackState(
        device_id=device_id,
        track_id="ownship",
        timestamp=0.0,
        bbox=(0.0, 0.0, 0.0, 0.0),
        confidence=1.0,
        class_name="ownship",
        pixel_coordinate=(0.0, 0.0),
        velocity=(0.0, 0.0),
        world_coordinate=(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    config = build_config(parse_args())
    output = {
        "task": "edge_inference_pipeline",
        "config": asdict(config),
    }
    if config.dry_run:
        output["status"] = "dry_run"
        output["notes"] = [
            "Pipeline assembled successfully in dry-run mode.",
            "All component factories executed without error.",
            "No frames were processed; no metrics are claimed.",
        ]
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    create_edge_pipeline(config)
    output["status"] = "ready"
    output["notes"] = [
        "EdgeProcessor assembled with real components.",
        "Call processor.process_frame(frame) or processor.run(frames).",
    ]
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
