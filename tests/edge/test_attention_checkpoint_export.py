"""Integration tests for attention checkpoint lifecycle and ONNX export
(review items S1, S2, S4)."""

from pathlib import Path

import torch
from torch import nn

from edge.detect.attention_injector import (
    inject_attention_to_yolo,
    strip_attention_from_yolo,
    verify_injection,
)

# ---------------------------------------------------------------------------
# Mock YOLO wrapper — mimics the nesting that inject_attention_to_yolo expects
# ---------------------------------------------------------------------------


class _MockDetectionModel(nn.Module):
    """Mimics ultralytics DetectionModel: .model is the nn.Sequential."""

    def __init__(self, seq: nn.Sequential) -> None:
        super().__init__()
        self.model = seq


class _MockYOLO(nn.Module):
    """Mimics ultralytics YOLO: .model is a DetectionModel (nn.Module)."""

    def __init__(self, seq: nn.Sequential) -> None:
        super().__init__()
        self.model = _MockDetectionModel(seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.model(x)


class DummyDetectForChecks(nn.Module):
    """Minimal Detect-like module so the injector finds a 'Detect' layer."""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return args[0] if len(args) == 1 else args[0]


def _make_sequential() -> nn.Sequential:
    """Build the internal Sequential shared by all mock wrappers."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),  # index 2
        nn.ReLU(inplace=False),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),  # index 4
        nn.ReLU(inplace=False),
        nn.Conv2d(128, 256, 3, stride=2, padding=1),  # index 6
        DummyDetectForChecks(),  # index 7
    )


def _make_injectable_model() -> _MockYOLO:
    """Return a mock accepted by ``inject_attention_to_yolo``."""
    return _MockYOLO(_make_sequential())


# ---------------------------------------------------------------------------
# S1: Checkpoint save → strip → load → verify
# ---------------------------------------------------------------------------


def test_inject_then_strip_restores_original_model() -> None:
    """Review S1: after strip, the model must be byte-identical to original."""
    torch.manual_seed(99)
    yolo = _make_injectable_model()
    seq = yolo.model.model

    original_state = {idx: seq[idx].state_dict().copy() for idx in [2, 4, 6]}

    info = inject_attention_to_yolo(yolo, reduction=8)
    assert len(info["injected_layers"]) == 3
    assert any(isinstance(seq[idx], nn.Sequential) for idx in [2, 4, 6])

    meta = strip_attention_from_yolo(yolo)
    assert len(meta["stripped_layers"]) == 3

    for idx in [2, 4, 6]:
        assert not isinstance(seq[idx], nn.Sequential)
        restored = seq[idx].state_dict()
        for key in original_state[idx]:
            assert torch.equal(restored[key], original_state[idx][key])


def test_strip_removes_all_attention_and_nothing_else() -> None:
    """strip should only affect layers that contain SEResidualAttention."""
    torch.manual_seed(42)
    yolo = _make_injectable_model()
    seq = yolo.model.model

    n_layers_before = len(seq)
    inject_attention_to_yolo(yolo, reduction=8)
    assert len(seq) == n_layers_before

    meta = strip_attention_from_yolo(yolo)
    stripped = {s["layer_index"] for s in meta["stripped_layers"]}
    assert stripped == {2, 4, 6}

    for idx in range(len(seq)):
        if idx in stripped:
            assert isinstance(seq[idx], nn.Conv2d)
        else:
            assert not isinstance(seq[idx], nn.Sequential)


def test_attention_state_dicts_are_preserved_in_strip_metadata() -> None:
    """Strip returns attention state_dicts."""
    torch.manual_seed(77)
    yolo = _make_injectable_model()
    inject_attention_to_yolo(yolo)

    meta = strip_attention_from_yolo(yolo)
    assert len(meta["attention_state_dicts"]) == 3
    for sd in meta["attention_state_dicts"]:
        assert len(sd) >= 4


# ---------------------------------------------------------------------------
# S4: verify_injection sanity check
# ---------------------------------------------------------------------------


def test_verify_injection_passes_on_healthy_model() -> None:
    """Review S4: verify_injection must report all checks OK."""
    torch.manual_seed(123)
    yolo = _make_injectable_model()
    inject_attention_to_yolo(yolo)

    # verify_injection works on the inner Sequential
    seq = yolo.model.model
    result = verify_injection(seq)
    assert result["output_finite"]
    assert result["shape_ok"]
    assert result["gradients_ok"]


def test_verify_injection_detects_nan_output() -> None:
    """verify_injection must report output_finite=False when output is NaN."""
    yolo = _make_injectable_model()
    inject_attention_to_yolo(yolo)
    seq = yolo.model.model
    for module in seq.modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data.fill_(float("nan"))
            break

    result = verify_injection(seq)
    assert not result["output_finite"]


# ---------------------------------------------------------------------------
# S2: ONNX checker test
# ---------------------------------------------------------------------------


def test_mock_model_onnx_export_validation() -> None:
    """Review S2: verify the exported graph passes onnx.checker."""
    torch.manual_seed(555)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.Conv2d(128, 64, 1),
    )

    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    out = traced(dummy)
    assert torch.isfinite(out).all()

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={},
        )
        assert Path(onnx_path).stat().st_size > 0

        try:
            import onnx

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
        except ImportError:
            pass  # onnx not installed — skip checker validation
    finally:
        Path(onnx_path).unlink(missing_ok=True)
