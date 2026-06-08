"""Integration tests verifying SEResidualAttention injection into a
YOLO-style detector neck (contract §3 / review item S1)."""

import torch
from torch import nn

from edge.detect.attention_blocks import SEResidualAttention


def _make_mock_yolo_neck() -> nn.Sequential:
    """Build a minimal YOLO-style neck that mimics PAN-FPN multi-scale outputs.

    Returns a Sequential whose last three feature-producing layers output
    channels 64, 128, 256 (P3/P4/P5 scales) before a dummy Detect head.
    """
    return nn.Sequential(
        # Backbone stub
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),  # index 2 — P5 scale
        nn.ReLU(inplace=False),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),  # index 4 — P4 scale
        nn.ReLU(inplace=False),
        nn.Conv2d(128, 256, 3, stride=2, padding=1),  # index 6 — P3 scale
        DummyDetect(),  # index 7
    )


class DummyDetect(nn.Module):
    """Placeholder detection head that just returns identity."""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return args[0] if len(args) == 1 else args[0]


def test_attention_injection_into_mock_neck_preserves_output_shape() -> None:
    """Inject attention after each scale output and verify forward shape."""
    neck = _make_mock_yolo_neck()
    # Indices 2 (ch=64), 4 (ch=128), 6 (ch=256) — the three scale producers
    for idx in [2, 4, 6]:
        out_ch = neck[idx].out_channels  # type: ignore[attr-defined]
        neck[idx] = nn.Sequential(neck[idx], SEResidualAttention(channels=out_ch))

    x = torch.randn(1, 3, 64, 64)
    y = neck(x)
    assert y.shape == (1, 256, 8, 8), f"unexpected output shape {y.shape}"


def test_attention_injection_gradient_flows_through_all_blocks() -> None:
    """Verify gradients reach every injected attention block after backward."""
    neck = _make_mock_yolo_neck()
    attention_blocks = []
    for idx in [2, 4, 6]:
        out_ch = neck[idx].out_channels  # type: ignore[attr-defined]
        attn = SEResidualAttention(channels=out_ch)
        neck[idx] = nn.Sequential(neck[idx], attn)
        attention_blocks.append(attn)

    x = torch.randn(1, 3, 64, 64)
    y = neck(x)
    loss = y.mean()
    loss.backward()

    for i, attn in enumerate(attention_blocks):
        for name, param in attn.named_parameters():
            assert (
                param.grad is not None
            ), f"attention block {i} param {name} has no gradient"
            assert torch.isfinite(
                param.grad
            ).all(), f"attention block {i} param {name} gradient has NaN/Inf"


def test_attention_injection_output_has_no_nan() -> None:
    """Injected attention should not introduce NaN values."""
    neck = _make_mock_yolo_neck()
    for idx in [2, 4, 6]:
        out_ch = neck[idx].out_channels  # type: ignore[attr-defined]
        neck[idx] = nn.Sequential(neck[idx], SEResidualAttention(channels=out_ch))

    x = torch.randn(2, 3, 64, 64)
    y = neck(x)
    assert torch.isfinite(y).all()


def test_attention_injection_respects_residual_identity() -> None:
    """With residual_scale=0.0, injected attention should be a no-op."""
    torch.manual_seed(42)
    neck = _make_mock_yolo_neck()
    for idx in [2, 4, 6]:
        out_ch = neck[idx].out_channels  # type: ignore[attr-defined]
        neck[idx] = nn.Sequential(
            neck[idx], SEResidualAttention(channels=out_ch, residual_scale=0.0)
        )

    # Build reference neck with the SAME weights
    torch.manual_seed(42)
    ref_neck = _make_mock_yolo_neck()

    x = torch.randn(1, 3, 64, 64)
    y = neck(x)
    y_ref = ref_neck(x)
    assert torch.allclose(
        y, y_ref, atol=1e-6
    ), "Attention with residual_scale=0 should be identity"


# ---------------------------------------------------------------------------
# S1: Shared injector consistency test
# ---------------------------------------------------------------------------


def test_shared_injector_produces_consistent_layers() -> None:
    """Review item S1: verify ``attention_injector`` returns immutable results.

    The shared injector's ``find_neck_scale_outputs`` must return the
    same layer indices and channel counts for the same model, regardless
    of which consumer calls it — guaranteeing training and inference use
    identical injection points.
    """
    from edge.detect.attention_injector import (
        find_detect_layer_index,
        find_neck_scale_outputs,
    )

    torch.manual_seed(1234)

    # Build two identical mock necks
    neck_a = _make_mock_yolo_neck()
    neck_b = _make_mock_yolo_neck()
    # Copy weights so they are identical
    for a, b in zip(neck_a, neck_b):
        if hasattr(a, "state_dict") and hasattr(b, "state_dict"):
            b.load_state_dict(a.state_dict())

    detect_a = find_detect_layer_index(neck_a)
    detect_b = find_detect_layer_index(neck_b)
    assert detect_a == detect_b == 7

    outputs_a = find_neck_scale_outputs(neck_a, detect_a)
    outputs_b = find_neck_scale_outputs(neck_b, detect_b)

    assert len(outputs_a) == len(outputs_b) == 3
    for (idx_a, ch_a), (idx_b, ch_b) in zip(outputs_a, outputs_b):
        assert idx_a == idx_b, f"layer index mismatch: {idx_a} != {idx_b}"
        assert ch_a == ch_b, f"channel count mismatch at idx {idx_a}: {ch_a} != {ch_b}"


# ---------------------------------------------------------------------------
# S5: Large-channel YOLO variant test
# ---------------------------------------------------------------------------


def test_neck_scale_search_handles_large_channels() -> None:
    """Review item S5: verify the heuristic works for YOLOv8m/l/x channels.

    YOLOv8m uses 96/192/384; YOLOv8l uses 128/256/512; YOLOv8x uses
    160/320/640.  The default max_channels=512 covers n/s/m/l but x
    (640) needs a higher bound.  This test ensures the search gracefully
    handles channels > 512 when the bound is raised.
    """
    from edge.detect.attention_injector import (
        find_detect_layer_index,
        find_neck_scale_outputs,
        is_plausible_scale_channel,
    )

    # Default bounds: 32–512
    assert is_plausible_scale_channel(64) is True  # YOLOv8n P3
    assert is_plausible_scale_channel(256) is True  # YOLOv8n P5
    assert is_plausible_scale_channel(384) is True  # YOLOv8m P5
    assert is_plausible_scale_channel(512) is True  # YOLOv8l P5

    # YOLOv8x P5 (640) exceeds default upper bound
    assert is_plausible_scale_channel(640) is False

    # With a raised bound it should pass
    assert is_plausible_scale_channel(640, max_channels=1024) is True

    # Build a mock model with YOLOv8x-like channels
    torch.manual_seed(5678)
    neck = nn.Sequential(
        nn.Conv2d(3, 80, 3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(80, 160, 3, stride=2, padding=1),  # P5-like
        nn.ReLU(inplace=False),
        nn.Conv2d(160, 320, 3, stride=2, padding=1),  # P4-like
        nn.ReLU(inplace=False),
        nn.Conv2d(320, 640, 3, stride=2, padding=1),  # P3-like (640ch!)
        DummyDetect(),
    )
    detect_idx = find_detect_layer_index(neck)
    assert detect_idx == 7

    # Default search (max 512) should miss the 640-channel layer
    default = find_neck_scale_outputs(neck, detect_idx)
    channels_found_default = {ch for _, ch in default}
    assert (
        640 not in channels_found_default
    ), "640-ch layer should be excluded by default 512 upper bound"

    # Raised bound (max 1024) should find it
    extended = find_neck_scale_outputs(neck, detect_idx, max_channels=1024)
    channels_found_extended = {ch for _, ch in extended}
    assert (
        640 in channels_found_extended
    ), "640-ch layer should be found with max_channels=1024"
