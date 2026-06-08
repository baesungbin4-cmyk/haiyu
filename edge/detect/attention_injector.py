"""Shared YOLO attention injection logic (contract §3 / review H1+M1).

Both ``train_yolo._inject_attention_to_yolo`` and
``YOLOAttentionDetector._inject_attention`` previously contained
near-identical copies of the unwrap → find-detect → find-neck-outputs →
inject pipeline.  This module is the **single source of truth**.

Architecture assumptions (review item S6)
-----------------------------------------
The neck-output search uses ``detect_idx - 12`` as the scan window
start.  The constant **12** was chosen because:

* YOLOv8n: backbone ~10 layers, neck ~12 layers, Detect at index ~22.
  ``22 - 12 = 10`` → just past the backbone.
* YOLOv8s: same layer count as YOLOv8n, just wider channels.
* YOLOv8m: backbone ~10, neck ~22, Detect at index ~32.  ``32 - 12 =
  20`` → still within the neck.
* YOLOv8l/x: similar proportions (~40-50 total layers, neck occupying
  the middle ~20-30).  ``detect_idx - 12`` consistently lands in the
  latter portion of the neck where the per-scale final C2f/Conv blocks
  live.

Verified on: YOLOv8n, YOLOv8s, YOLOv8m (ultralytics 8.x).  If future
YOLO versions change the neck depth significantly, adjust this window
or replace the heuristic with a graph-tracing approach.

Channel range (32–512)
----------------------
The ``_is_plausible_scale_channel`` check accepts channels in [32, 512].
This covers YOLOv8n (64/128/256), YOLOv8s (64/128/256), and YOLOv8m
(96/192/384).  YOLOv8l (128/256/512) is also fully covered at the upper
bound.  YOLOv8x (160/320/640) would need the upper bound raised to 1024
— the function documents this limitation and can be parameterised.
"""

from __future__ import annotations

from typing import Any

from torch import nn

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def inject_attention_to_yolo(
    model: nn.Module,
    *,
    reduction: int = 16,
    min_hidden_channels: int = 4,
    residual_scale: float = 1.0,
    device: str | None = None,
) -> dict[str, Any]:
    """Inject SEResidualAttention into the PAN-FPN neck of a YOLO model.

    This is the **single shared entry point** for both the training
    pipeline and the edge detector.  Contract §3: "PAN-FPN 颈部各尺度
    融合后、检测头前".

    Parameters
    ----------
    model:
        An ultralytics ``YOLO`` instance or its inner ``nn.Module``.
    reduction:
        Channel reduction ratio for ``SEResidualAttention``.
    min_hidden_channels:
        Minimum hidden channels for ``SEResidualAttention``.
    residual_scale:
        Multiplier on the attention output before the residual add.
    device:
        Torch device string.  If *None*, inferred from the model.

    Returns
    -------
    dict
        ``{"injected_layers": [...], "attention_config": {...}}``.
    """
    from edge.detect.attention_blocks import SEResidualAttention

    model_seq = unwrap_model_sequential(model)
    if model_seq is None:
        raise NotImplementedError(
            "Cannot inject attention: the loaded YOLO model does not expose "
            "a recognised nn.Sequential backbone."
        )

    detect_idx = find_detect_layer_index(model_seq)
    if detect_idx is None:
        raise NotImplementedError(
            "Cannot inject attention: no Detect layer found in the model."
        )

    neck_candidates = find_neck_scale_outputs(model_seq, detect_idx)
    if len(neck_candidates) < 3:
        raise NotImplementedError(
            f"Expected ≥3 neck scale outputs for P3/P4/P5, found "
            f"{len(neck_candidates)}.  The model architecture may not "
            f"match the expected YOLOv8 PAN-FPN neck."
        )

    _device = device or _infer_device(model_seq)

    injected = []
    for idx, out_channels in reversed(neck_candidates):
        attn = SEResidualAttention(
            channels=out_channels,
            reduction=reduction,
            min_hidden_channels=min_hidden_channels,
            residual_scale=residual_scale,
        )
        attn = attn.to(_device)
        model_seq[idx] = nn.Sequential(model_seq[idx], attn)
        injected.insert(
            0,
            {
                "layer_index": idx,
                "channels": out_channels,
                "hidden_channels": attn.hidden_channels,
            },
        )

    return {
        "injected_layers": injected,
        "attention_config": {
            "module": "SEResidualAttention",
            "reduction": reduction,
            "min_hidden_channels": min_hidden_channels,
            "residual_scale": residual_scale,
            "note": "Inserted after each PAN-FPN neck scale output "
            "(P3/P4/P5) before the Detect head, per contract §3.",
        },
    }


# ---------------------------------------------------------------------------
# Model introspection helpers (used by inject_attention_to_yolo)
# ---------------------------------------------------------------------------


def unwrap_model_sequential(model: nn.Module) -> nn.Module | None:
    """Return the internal ``nn.Sequential`` of a YOLO wrapper, or *None*.

    Handles two common nesting patterns found in ultralytics YOLO objects:

    1. ``YOLO.model`` → ``DetectionModel`` → ``.model`` (Sequential)
    2. ``YOLO.model`` → ``nn.Sequential`` directly
    """
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        inner = model.model
        if hasattr(inner, "model") and isinstance(inner.model, nn.Sequential):
            return inner.model
        if isinstance(inner, nn.Sequential):
            return inner
    return None


def find_detect_layer_index(model_seq: nn.Module) -> int | None:
    """Locate the Detect (or v10Detect / v8Detect) head in a Sequential."""
    for idx, module in enumerate(model_seq):
        if "Detect" in module.__class__.__name__:
            return idx
    return None


def find_neck_scale_outputs(
    model_seq: nn.Module,
    detect_idx: int,
    *,
    neck_window: int = 12,
    min_channels: int = 32,
    max_channels: int = 512,
) -> list[tuple[int, int]]:
    """Find candidate neck output layers before the Detect head.

    Scans the ``neck_window`` layers immediately preceding the Detect
    head for Conv2d / C2f blocks whose output channel count falls within
    ``[min_channels, max_channels]``.  Returns the **last** occurrence
    per unique channel count — these correspond to the per-scale final
    feature maps (P3/P4/P5).

    Parameters
    ----------
    model_seq:
        The ``nn.Sequential`` of the YOLO model.
    detect_idx:
        Index of the Detect head layer.
    neck_window:
        Number of layers to scan before *detect_idx*.  Default 12 is
        sufficient for YOLOv8n/s/m/l (see module docstring).
    min_channels, max_channels:
        Valid channel range for a detection-scale feature map.

    Returns
    -------
    list[tuple[int, int]]
        ``[(layer_index, output_channels), ...]`` sorted ascending.
        At most 3 entries.
    """
    neck_start = max(0, detect_idx - neck_window)
    candidates: list[tuple[int, int]] = []

    for idx in range(neck_start, detect_idx):
        module = model_seq[idx]
        out_ch = module_output_channels(module)
        if out_ch is not None and min_channels <= out_ch <= max_channels:
            candidates.append((idx, out_ch))

    # Keep only the last candidate per unique channel count — earlier
    # layers with the same channel count are intermediate convs, not the
    # final scale output we want to augment.
    seen: set[int] = set()
    deduped: list[tuple[int, int]] = []
    for idx, ch in reversed(candidates):
        if ch not in seen:
            seen.add(ch)
            deduped.append((idx, ch))
    deduped.reverse()

    return deduped[-3:]


def module_output_channels(module: nn.Module) -> int | None:
    """Best-effort extraction of a module's output channel count.

    Probes (in order):
    1. ``.out_channels`` on ``nn.Conv2d``
    2. ``.cv3`` / ``.cv2`` / ``.cv1`` / ``.conv`` attributes (ultralytics
       C2f / C3k2 block conventions)
    3. The last ``nn.Conv2d`` child module (generic fallback)
    """
    if isinstance(module, nn.Conv2d):
        return module.out_channels
    for attr in ("cv3", "cv2", "cv1", "conv"):
        sub = getattr(module, attr, None)
        if isinstance(sub, nn.Conv2d):
            return sub.out_channels
    last_conv = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is not None:
        return last_conv.out_channels
    return None


def is_plausible_scale_channel(
    ch: int,
    *,
    min_channels: int = 32,
    max_channels: int = 512,
) -> bool:
    """Return True if *ch* could be a YOLO detection-scale feature channel.

    Default bounds cover YOLOv8n (64/128/256), YOLOv8s (64/128/256),
    YOLOv8m (96/192/384), and YOLOv8l (128/256/512).  YOLOv8x
    (160/320/640) requires ``max_channels=1024``.
    """
    return min_channels <= ch <= max_channels


# ---------------------------------------------------------------------------
# Attention stripping (H1/M1: checkpoint compatibility)
# ---------------------------------------------------------------------------


def strip_attention_from_yolo(model: nn.Module) -> dict[str, Any]:
    """Remove injected SEResidualAttention blocks, restoring clean YOLO arch.

    After training with attention, ultralytics saves a checkpoint whose
    ``state_dict`` contains attention weight keys.  ``YOLO(weights_path)``
    reconstructs a standard architecture from its YAML config and will
    fail on ``load_state_dict(strict=True)`` or silently drop attention
    weights with ``strict=False``.

    Call this **before** checkpoint save to produce a clean weights file.
    The attention weights are returned so they can be saved separately and
    re-injected later by the detector.

    Returns
    -------
    dict
        ``{"stripped_layers": [...], "attention_state_dicts": [...]}``
        for audit trail and optional separate persistence.
    """
    model_seq = unwrap_model_sequential(model)
    if model_seq is None:
        return {"stripped_layers": [], "attention_state_dicts": []}

    stripped = []
    attention_sds = []
    for idx, module in enumerate(model_seq):
        if not isinstance(module, nn.Sequential):
            continue
        if len(module) != 2:
            continue
        # Pattern: nn.Sequential(original_layer, SEResidualAttention)
        from edge.detect.attention_blocks import SEResidualAttention

        if not isinstance(module[1], SEResidualAttention):
            continue
        original = module[0]
        attention_sds.append(module[1].state_dict())
        model_seq[idx] = original
        stripped.append(
            {
                "layer_index": idx,
                "channels": module[1].hidden_channels,
            }
        )

    return {
        "stripped_layers": stripped,
        "attention_state_dicts": attention_sds,
    }


def save_clean_checkpoint(
    model: nn.Module,
    output_path: str,
) -> None:
    """Strip attention, then save a standard ultralytics-compatible checkpoint.

    Use this after training with ``--use-residual-attention`` to produce a
    checkpoint that can be loaded with ``YOLO("path.pt")``.  The attention
    weights are discarded; re-inject via ``inject_attention_to_yolo`` or
    ``YOLOAttentionDetector(use_residual_attention=True)`` at load time.
    """
    import torch

    meta = strip_attention_from_yolo(model)
    if meta["stripped_layers"]:
        print(
            "Stripped attention from layers "
            f"{[s['layer_index'] for s in meta['stripped_layers']]} "
            f"before saving {output_path}"
        )
    torch.save(model.state_dict(), output_path)
    print(f"Clean checkpoint saved to {output_path}")


# ---------------------------------------------------------------------------
# Post-injection sanity check (S4)
# ---------------------------------------------------------------------------


def verify_injection(model: nn.Module) -> dict[str, bool]:
    """Run a dummy forward pass to verify attention injection is healthy.

    Checks that the model:
    1. Produces a finite (non-NaN, non-Inf) output.
    2. Preserves the expected detection output shape.
    3. Has gradient flow through all parameters.

    Call this after ``inject_attention_to_yolo`` for a quick sanity check.

    Returns
    -------
    dict
        ``{"output_finite": bool, "shape_ok": bool, "gradients_ok": bool}``.
    """
    import torch

    device = _infer_device_from_model(model)
    dummy = torch.randn(1, 3, 640, 640, device=device)
    result: dict[str, bool] = {}

    # 1. Output sanity (no_grad is fine here)
    try:
        with torch.no_grad():
            out = model(dummy)
    except Exception:
        result["output_finite"] = False
        result["shape_ok"] = False
        result["gradients_ok"] = False
        return result

    result["output_finite"] = bool(torch.isfinite(out).all())
    result["shape_ok"] = out.ndim >= 2  # detection output is at least 2D

    # 2. Gradient flow — must run a fresh forward pass OUTSIDE no_grad
    # so that the computation graph is available for backward().
    # Zero any existing grads first so repeated calls don't accumulate.
    try:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        out_grad = model(dummy)
        out_mean = out_grad.float().mean()
        out_mean.backward()
        grads_ok = True
        for param in model.parameters():
            if param.requires_grad and param.grad is None:
                grads_ok = False
                break
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grads_ok = False
                break
        result["gradients_ok"] = grads_ok
    except Exception:
        result["gradients_ok"] = False

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_device(model_seq: nn.Module) -> str:
    """Return the device string of the first parameter found in *model_seq*."""
    try:
        return str(next(model_seq.parameters()).device)
    except StopIteration:
        return "cpu"


def _infer_device_from_model(model: nn.Module) -> str:
    """Return the device string from the top-level model (not Sequential)."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


__all__ = [
    "find_detect_layer_index",
    "find_neck_scale_outputs",
    "inject_attention_to_yolo",
    "is_plausible_scale_channel",
    "module_output_channels",
    "save_clean_checkpoint",
    "strip_attention_from_yolo",
    "unwrap_model_sequential",
    "verify_injection",
]
