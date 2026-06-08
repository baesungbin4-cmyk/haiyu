"""Concrete YOLO + residual-attention detector implementing ``DetectionRunner``.

This module provides ``YOLOAttentionDetector``, the primary detector for the
Haiyu edge pipeline.  It loads a YOLO model from weights, optionally injects
``SEResidualAttention`` blocks into the PAN-FPN neck (contract §3), and exposes
a ``detect(frame, timestamp=...)`` method that returns validated ``Detection``
objects.

``MockDetector`` (in ``edge_processor_v2``) remains available for tests but is
no longer the only concrete detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from edge.common.schemas import Detection
from edge.detect.postprocess import detections_from_predictions

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class YOLOAttentionDetector:
    """YOLO detector with optional residual attention injection.

    Parameters
    ----------
    model_path:
        Path to a YOLO weights file (``.pt``, ultralytics format).
    device_id:
        Logical device identifier written into every ``Detection``.
    class_names:
        Mapping from class-ID (int) to human-readable label.  If *None*,
        generic ``class_N`` names are emitted.
    confidence_threshold:
        Minimum confidence for NMS filtering (default 0.5).
    iou_threshold:
        IoU overlap threshold for NMS suppression (default 0.45).
    use_residual_attention:
        When *True*, inject ``SEResidualAttention`` blocks into the YOLO
        neck after loading the weights (contract §3).  Requires
        ``edge.detect.attention_blocks.SEResidualAttention``.
    attention_reduction:
        Channel reduction ratio for each injected attention block.
    attention_min_hidden:
        Minimum hidden channels for each injected attention block.
    device:
        Torch device string (e.g. ``"cpu"``, ``"cuda:0"``).  Defaults to
        CUDA when available, CPU otherwise.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device_id: str,
        class_names: Mapping[int, str] | Sequence[str] | None = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        use_residual_attention: bool = False,
        attention_reduction: int = 16,
        attention_min_hidden: int = 4,
        device: str | None = None,
    ) -> None:
        self.device_id = device_id
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # ------------------------------------------------------------------
        # Load the YOLO model
        # ------------------------------------------------------------------
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {model_path}. "
                "Automatic download is disabled; provide local weights."
            )

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLOAttentionDetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._yolo = YOLO(str(model_path))
        self._model_path = str(model_path)

        # ------------------------------------------------------------------
        # Optionally inject residual attention into the neck
        # ------------------------------------------------------------------
        self._attention_injected: list[dict] = []
        if use_residual_attention:
            self._attention_injected = self._inject_attention(
                attention_reduction,
                attention_min_hidden,
            )

    # ------------------------------------------------------------------
    # Public API — matches DetectionRunner protocol
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        *,
        timestamp: float,
    ) -> list[Detection]:
        """Run YOLO detection on a single frame.

        Parameters
        ----------
        frame:
            Image as ``np.ndarray`` of shape ``[H, W, 3]`` (uint8).
        timestamp:
            Unix timestamp or frame time in seconds.

        Returns
        -------
        list[Detection]
            Detections filtered by confidence and NMS.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must have shape [H, W, 3], got {frame.shape}")
        import warnings

        # Only suppress the known ultralytics inference warning about
        # "no visible/enabled labels" when running with verbose=False.
        # Broad "ignore" would mask DeprecationWarning and model
        # compatibility warnings (review M4).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*labels.*",
                category=UserWarning,
            )
            results = self._yolo(frame, verbose=False)
        if not results or len(results) == 0:
            return []
        result = results[0]
        if result.boxes is None:
            return []
        raw = result.boxes.data
        if raw is None or raw.numel() == 0:
            return []
        # raw shape: [N, 6] = [x1, y1, x2, y2, conf, cls]
        return detections_from_predictions(
            raw,
            device_id=self.device_id,
            timestamp=timestamp,
            class_names=self.class_names,
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
        )

    @property
    def attention_injected_layers(self) -> list[dict]:
        """Return the audit trail of layers that received attention blocks."""
        return list(self._attention_injected)

    @property
    def model_path(self) -> str:
        """Path to the underlying YOLO weights file."""
        return self._model_path

    @property
    def device(self) -> str:
        """Torch device used for inference."""
        return self._device

    # ------------------------------------------------------------------
    # Attention injection — delegates to shared injector (review M1)
    # ------------------------------------------------------------------

    def _inject_attention(
        self,
        reduction: int,
        min_hidden: int,
    ) -> list[dict]:
        """Inject SEResidualAttention via the shared ``attention_injector``.

        Uses ``edge.detect.attention_injector.inject_attention_to_yolo``,
        the single source of truth shared with ``train_yolo.py``.
        """
        from edge.detect.attention_injector import (
            inject_attention_to_yolo,
        )

        info = inject_attention_to_yolo(
            self._yolo,
            reduction=reduction,
            min_hidden_channels=min_hidden,
            device=self._device,
        )
        return info["injected_layers"]


__all__ = ["YOLOAttentionDetector"]
