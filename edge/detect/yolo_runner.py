"""YOLO-compatible detection runner interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from edge.common.schemas import Detection
from edge.detect.postprocess import detections_from_predictions


class ModelBackend(Protocol):
    """Minimal detector backend contract."""

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run backend inference on a preprocessed ``[1, C, H, W]`` tensor."""
        ...


@dataclass(frozen=True)
class OnnxBackend:
    """ONNX export/runtime placeholder.

    The Ultralytics path is the real detector path in this repository. ONNX
    loading and CPU fallback are not implemented or edge-validated here.
    """

    model_path: str | Path

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "ONNX backend is a documented placeholder; no CPU fallback "
            "runtime has been implemented or edge-validated."
        )


@dataclass(frozen=True)
class RknnBackend:
    """RKNN export/runtime placeholder.

    RKNN/NPU loading is not implemented or measured in this repository.
    """

    model_path: str | Path

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "RKNN backend is a documented placeholder; no NPU runtime "
            "has been implemented or edge-validated."
        )


class MockBackend:
    """Deterministic backend for unit tests."""

    def __init__(
        self,
        output: torch.Tensor | Sequence[Sequence[float]],
    ) -> None:
        self.output = torch.as_tensor(output, dtype=torch.float32)
        self.last_input: torch.Tensor | None = None

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.last_input = input_tensor.detach().clone()
        return self.output.clone()


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing options for detector input tensors."""

    input_size: tuple[int, int] = (640, 640)
    normalize: bool = True
    channel_layout: Literal["auto", "hwc", "chw"] = "auto"


class YoloDetectionRunner:
    """Runs preprocessing, backend inference, and schema postprocessing."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        device_id: str,
        class_names: Mapping[int, str] | Sequence[str] | None = None,
        preprocess_config: PreprocessConfig | None = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> None:
        self.backend = backend
        self.device_id = device_id
        self.class_names = class_names
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def detect(
        self,
        frame: np.ndarray | torch.Tensor,
        *,
        timestamp: float,
    ) -> list[Detection]:
        """Run detection on a ``[H, W, 3]`` frame.

        Accepts ``np.ndarray`` (per ``DetectionRunner`` Protocol) or
        ``torch.Tensor`` (for testing convenience).  Internally converted
        to tensor by ``preprocess_frame``.
        """
        input_tensor = preprocess_frame(frame, self.preprocess_config)
        predictions = self.backend.infer(input_tensor)
        return detections_from_predictions(
            predictions,
            device_id=self.device_id,
            timestamp=timestamp,
            class_names=self.class_names,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )


def preprocess_frame(
    frame: np.ndarray | torch.Tensor,
    config: PreprocessConfig | None = None,
) -> torch.Tensor:
    """Resize, normalize, and convert one image to ``[1, C, H, W]``.

    Accepts ``np.ndarray`` or ``torch.Tensor`` of shape ``[H, W, C]``
    or ``[C, H, W]``.
    """

    config = config or PreprocessConfig()
    if len(config.input_size) != 2:
        raise ValueError("input_size must be (height, width)")
    height, width = config.input_size
    if height <= 0 or width <= 0:
        raise ValueError("input_size dimensions must be positive")

    tensor = torch.as_tensor(frame)
    if tensor.ndim != 3:
        raise ValueError("frame must have shape [H, W, C] or [C, H, W]")

    if config.channel_layout not in ("auto", "hwc", "chw"):
        raise ValueError("channel_layout must be 'auto', 'hwc', or 'chw'")
    if _is_hwc(tensor, config.channel_layout):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] not in (1, 3):
        raise ValueError("frame must have 1 or 3 channels")

    if config.normalize and tensor.dtype != torch.uint8:
        raise TypeError("normalize=True expects a uint8 frame")
    tensor = tensor.to(dtype=torch.float32)
    if config.normalize:
        tensor = tensor / 255.0

    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return tensor.contiguous()


def _is_hwc(tensor: torch.Tensor, channel_layout: str) -> bool:
    if channel_layout == "hwc":
        return True
    if channel_layout == "chw":
        return False
    return tensor.shape[-1] in (1, 3)


__all__ = [
    "MockBackend",
    "ModelBackend",
    "OnnxBackend",
    "PreprocessConfig",
    "RknnBackend",
    "YoloDetectionRunner",
    "preprocess_frame",
]
