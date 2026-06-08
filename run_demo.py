"""Full-stack demo: simulate edge pipeline → MQTT → cloud API → frontend.

Generates synthetic approaching-vessel scenarios and runs the complete
detection → tracking → prediction → risk → MQTT pipeline.
"""

from __future__ import annotations

import math
import sys
import numpy as np

from edge.common.schemas import Detection, TrackState, RiskResult, TrajectoryPrediction
from edge.edge_processor_v2 import EdgeProcessor, EdgeProcessorConfig
from edge.risk.risk_engine import RiskThresholds, evaluate_track_risk
from edge.track.tracker import IoUTracker
from edge.predict.lstm_causal_attention import LSTMCausalAttentionPredictor
from edge.predict.predictor_adapter import LSTMTrajectoryPredictorAdapter

# Fix Unicode output on Windows
sys.stdout.reconfigure(encoding="utf-8")


def _ownship_state(device_id: str, timestamp: float) -> TrackState:
    return TrackState(
        device_id=device_id,
        track_id="ownship",
        timestamp=timestamp,
        bbox=(0.0, 0.0, 0.0, 0.0),
        confidence=1.0,
        class_name="ownship",
        pixel_coordinate=(0.0, 0.0),
        velocity=(0.0, 0.0),
        world_coordinate=(0.0, 0.0),
    )


class SimulatedApproachDetector:
    """Generate synthetic detections for a vessel approaching from the right."""

    def __init__(self, device_id: str, approach_speed: float = 3.0):
        self.device_id = device_id
        self.approach_speed = approach_speed
        self._frame_idx = 0

    @property
    def warmup_frames(self) -> int:
        return 0

    def detect(self, frame: np.ndarray, *, timestamp: float) -> list[Detection]:
        """Simulate a vessel moving from right (x=500) toward ownship at (x=0, y=0)."""
        self._frame_idx += 1
        t = self._frame_idx
        x = max(10.0, 500.0 - self.approach_speed * t)
        y = 20.0 + 3.0 * math.sin(t * 0.15)

        bbox = (x - 15.0, y - 8.0, x + 15.0, y + 8.0)
        ts = float(t) / 15.0

        detection = Detection(
            device_id=self.device_id,
            timestamp=ts,
            bbox=bbox,
            confidence=0.92,
            class_name="vessel",
            pixel_coordinate=(x, y),
            world_coordinate=(x, y),
        )
        return [detection]


class ConfiguredRiskEvaluator:
    """Risk evaluator wrapping evaluate_track_risk with configured thresholds."""

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.thresholds = RiskThresholds(
            tcpa_alert=30.0,
            cpa_alert=50.0,
            tcpa_critical=10.0,
            cpa_critical=20.0,
            tcpa_high=20.0,
            cpa_high=35.0,
            tcpa_low=25.0,
            cpa_low=40.0,
        )

    def evaluate(
        self,
        target_state: TrackState,
        prediction: TrajectoryPrediction | None,
        *,
        timestamp: float,
    ) -> RiskResult | None:
        return evaluate_track_risk(
            own_state=_ownship_state(self.device_id, timestamp),
            target_state=target_state,
            prediction=prediction,
            risk_thresholds=self.thresholds,
        )


def main() -> None:
    device_id = "edge-01"
    total_frames = 200

    # ------------------------------------------------------------------
    # 1. Detection — synthetic approach scenario
    # ------------------------------------------------------------------
    detector = SimulatedApproachDetector(device_id=device_id, approach_speed=3.0)

    # ------------------------------------------------------------------
    # 2. Tracker — IoU-based
    # ------------------------------------------------------------------
    tracker = IoUTracker(iou_threshold=0.3, max_missed=3)

    # ------------------------------------------------------------------
    # 3. Predictor — LSTM + causal attention
    # ------------------------------------------------------------------
    lstm_model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        future_len=5,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
    )
    predictor = LSTMTrajectoryPredictorAdapter(
        lstm_model,
        history_len=12,
        use_world_coordinate=True,
        device="cpu",
    )

    # ------------------------------------------------------------------
    # 4. Risk evaluator
    # ------------------------------------------------------------------
    risk_evaluator = ConfiguredRiskEvaluator(device_id)

    # ------------------------------------------------------------------
    # 5. MQTT publisher — real MQTT gateway (adapted to protocol)
    # ------------------------------------------------------------------
    from edge.mqtt_gateway import MqttGateway, MqttGatewayConfig
    from edge.common.schemas import MqttEventEnvelope

    class MqttPublisherAdapter:
        """Adapt MqttGateway to the MqttPublisher protocol."""

        def __init__(self, gateway: MqttGateway):
            self._gw = gateway

        def publish(self, event: MqttEventEnvelope) -> None:
            self._gw.publish_event(event)

    gw = MqttGateway(
        config=MqttGatewayConfig(
            host="localhost",
            port=1883,
            client_id=device_id,
            username="edge-01",
            password="local-dev-edge-password",
        ),
    )
    gw.connect()
    mqtt = MqttPublisherAdapter(gw)

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
            device_id=device_id,
            fps=15.0,
            history_len=12,
        ),
    )

    # ------------------------------------------------------------------
    # 7. Run the pipeline
    # ------------------------------------------------------------------
    print(f"Starting pipeline: {total_frames} frames, device={device_id}")
    print(
        "Scenario: vessel approaching from (500,20) at "
        f"{detector.approach_speed} px/frame"
    )
    print(
        "MQTT → localhost:1883 | API → http://localhost:8000 | "
        "Frontend → http://localhost:5174"
    )
    print("-" * 70)

    alerts_triggered = 0

    for i in range(total_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = processor.process_frame(frame)

        if result.errors:
            print(f"  Frame {i:3d}: ERRORS: {result.errors}")

        if result.risks:
            for risk in result.risks:
                if risk.should_alert:
                    alerts_triggered += 1
                    distance = (
                        math.sqrt(
                            (result.tracks[0].world_coordinate[0]) ** 2
                            + (result.tracks[0].world_coordinate[1]) ** 2
                        )
                        if result.tracks
                        else 0
                    )
                    print(
                        f"  Frame {i:3d}: ALERT level={risk.risk_level:8s}  "
                        f"TCPA={risk.tcpa_seconds or 0:.1f}s  "
                        f"CPA={risk.cpa_distance_m or 0:.1f}m  "
                        f"dist={distance:.0f}px  track={risk.track_id}"
                    )

        if i % 40 == 0:
            print(
                f"  Frame {i:3d}: processed, {alerts_triggered} alerts so far, "
                f"{len(result.tracks)} tracks active"
            )

    print("-" * 70)
    print(f"Done. {alerts_triggered} alerts triggered over {total_frames} frames.")
    print()
    print("Now check the API:")
    print(
        '  curl -H "Authorization: Bearer local-dev-api-token" '
        "http://localhost:8000/api/v1/devices"
    )
    print(
        '  curl -H "Authorization: Bearer local-dev-api-token" '
        "http://localhost:8000/api/v1/alerts/latest"
    )
    print(
        '  curl -H "Authorization: Bearer local-dev-api-token" '
        "http://localhost:8000/api/v1/stats/summary"
    )

    # Pipeline complete
    print("Pipeline finished successfully!")


if __name__ == "__main__":
    main()
