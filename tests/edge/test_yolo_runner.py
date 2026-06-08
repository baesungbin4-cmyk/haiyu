import torch

from edge.common.schemas import Detection
from edge.detect.yolo_runner import (
    MockBackend,
    PreprocessConfig,
    YoloDetectionRunner,
    preprocess_frame,
)


def test_preprocess_frame_resizes_normalizes_and_converts_to_chw() -> None:
    frame = torch.full((8, 10, 3), 255, dtype=torch.uint8)

    tensor = preprocess_frame(frame, PreprocessConfig(input_size=(16, 12)))

    assert tensor.shape == (1, 3, 16, 12)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor.max(), torch.tensor(1.0))


def test_preprocess_frame_rejects_float_input_when_normalizing() -> None:
    frame = torch.rand(8, 10, 3)

    try:
        preprocess_frame(frame)
    except TypeError as exc:
        assert "uint8" in str(exc)
    else:
        raise AssertionError("float input should be rejected when normalizing")


def test_preprocess_frame_handles_grayscale_hwc_input() -> None:
    frame = torch.full((3, 10, 1), 255, dtype=torch.uint8)

    tensor = preprocess_frame(frame, PreprocessConfig(input_size=(6, 8)))

    assert tensor.shape == (1, 1, 6, 8)
    assert torch.allclose(tensor.max(), torch.tensor(1.0))


def test_mock_backend_output_is_converted_to_detection_schema() -> None:
    backend = MockBackend(
        [
            [2, 4, 10, 12, 0.95, 0],
            [20, 20, 30, 30, 0.30, 0],
        ]
    )
    runner = YoloDetectionRunner(
        backend,
        device_id="edge-01",
        class_names={0: "vessel"},
        preprocess_config=PreprocessConfig(input_size=(32, 32)),
    )
    frame = torch.full((16, 16, 3), 128, dtype=torch.uint8)

    detections = runner.detect(frame, timestamp=1_718_000_000.0)

    assert backend.last_input is not None
    assert backend.last_input.shape == (1, 3, 32, 32)
    assert len(detections) == 1
    detection = detections[0]
    assert isinstance(detection, Detection)
    assert detection.device_id == "edge-01"
    assert detection.class_name == "vessel"
    assert detection.bbox == (2.0, 4.0, 10.0, 12.0)
    assert detection.pixel_coordinate == (6.0, 8.0)
