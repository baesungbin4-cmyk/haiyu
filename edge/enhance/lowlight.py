"""Configurable low-light enhancement using CLAHE and Retinex.

Inputs are ``uint8`` numpy frames in ``[H, W, 3]`` layout. Set
``color_order`` to ``"RGB"`` or ``"BGR"`` so luminance weights match the
incoming frame. Outputs keep the same shape and dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

EnhancementMode = Literal["disabled", "clahe", "retinex", "clahe_retinex"]
ColorOrder = Literal["RGB", "BGR"]


@dataclass(frozen=True)
class LowLightConfig:
    """Low-light enhancement configuration."""

    mode: EnhancementMode = "clahe_retinex"
    color_order: ColorOrder = "BGR"
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    retinex_sigma: float = 15.0


def enhance_low_light(
    frame: np.ndarray,
    config: LowLightConfig | None = None,
) -> np.ndarray:
    """Apply the configured enhancement pipeline to one RGB or BGR frame."""

    config = config or LowLightConfig()
    _validate_config(config)
    _validate_frame(frame)

    if config.mode == "disabled":
        return frame.copy()

    output = frame
    if config.mode in ("clahe", "clahe_retinex"):
        output = apply_clahe(output, config)
    if config.mode in ("retinex", "clahe_retinex"):
        output = apply_retinex(output, config)
    return output


def apply_clahe(
    frame: np.ndarray,
    config: LowLightConfig | None = None,
) -> np.ndarray:
    """Apply contrast-limited adaptive histogram equalization on luminance.

    .. warning::
       **当前为简化实现（tiled histogram equalization，无邻域插值），非完整 CLAHE。**

       标准 CLAHE 对每个像素使用其 4 个邻域 tile 的映射函数做双线性插值以消除
       tile 边界伪影。当前实现每个 tile 独立映射，边界处映射函数跳变，在
       ``clahe_tile_grid_size`` 较小时可能产生可见的 tile 边界（棋盘格伪影）。

       Contract §9 将 CLAHE 列为"可降级"模块，此简化实现在弱光场景下仍能
       改善局部对比度，但输出质量低于 OpenCV ``cv2.createCLAHE``。

       如需完整 CLAHE，建议直接调用 OpenCV 实现或在此函数中添加双线性插值逻辑。
    """

    config = config or LowLightConfig(mode="clahe")
    _validate_config(config)
    _validate_frame(frame)

    luminance = _luminance(frame, config.color_order).astype(np.uint8)
    enhanced_luminance = _clahe_luminance(
        luminance,
        clip_limit=config.clahe_clip_limit,
        tile_grid_size=config.clahe_tile_grid_size,
    )
    return _replace_luminance(frame, luminance, enhanced_luminance)


def apply_retinex(
    frame: np.ndarray,
    config: LowLightConfig | None = None,
) -> np.ndarray:
    """Apply single-scale Retinex to each image channel."""

    config = config or LowLightConfig(mode="retinex")
    _validate_config(config)
    _validate_frame(frame)

    image = frame.astype(np.float32) + 1.0
    illumination = _gaussian_blur(image, sigma=config.retinex_sigma) + 1.0
    retinex = np.log(image) - np.log(illumination)
    return _normalize_channels(retinex)


def _clahe_luminance(
    luminance: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    height, width = luminance.shape
    tiles_y, tiles_x = tile_grid_size
    output = np.empty_like(luminance)
    tile_h = int(np.ceil(height / tiles_y))
    tile_w = int(np.ceil(width / tiles_x))

    for tile_y in range(tiles_y):
        y1 = tile_y * tile_h
        y2 = min(height, y1 + tile_h)
        if y1 >= height:
            continue
        for tile_x in range(tiles_x):
            x1 = tile_x * tile_w
            x2 = min(width, x1 + tile_w)
            if x1 >= width:
                continue
            tile = luminance[y1:y2, x1:x2]
            lookup = _clahe_lookup(tile, clip_limit)
            output[y1:y2, x1:x2] = lookup[tile]
    return output


def _clahe_lookup(tile: np.ndarray, clip_limit: float) -> np.ndarray:
    histogram = np.bincount(tile.reshape(-1), minlength=256).astype(np.float32)
    tile_area = float(tile.size)
    clip_count = max(1.0, clip_limit * tile_area / 256.0)
    clipped = np.minimum(histogram, clip_count)
    excess = histogram.sum() - clipped.sum()
    clipped += excess / 256.0

    cdf = np.cumsum(clipped)
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return np.arange(256, dtype=np.uint8)
    cdf_min = nonzero[0]
    denominator = cdf[-1] - cdf_min
    if denominator <= 0:
        return np.arange(256, dtype=np.uint8)
    lookup = (cdf - cdf_min) / denominator * 255.0
    return np.clip(np.rint(lookup), 0, 255).astype(np.uint8)


def _luminance(frame: np.ndarray, color_order: ColorOrder) -> np.ndarray:
    image = frame.astype(np.float32)
    if color_order == "RGB":
        red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    else:
        blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _replace_luminance(
    frame: np.ndarray,
    old_luminance: np.ndarray,
    new_luminance: np.ndarray,
) -> np.ndarray:
    old_values = old_luminance.astype(np.float32)
    new_values = new_luminance.astype(np.float32)
    ratio = (new_values + 1.0) / (old_values + 1.0)
    output = frame.astype(np.float32) * ratio[..., None]
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel_1d(sigma)
    blurred = _convolve_axis(image, kernel, axis=0)
    return _convolve_axis(blurred, kernel, axis=1)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _convolve_axis(
    image: np.ndarray,
    kernel: np.ndarray,
    axis: int,
) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(image, pad_width=pad_width, mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)

    for offset, weight in enumerate(kernel):
        start = offset
        end = start + image.shape[axis]
        slices = [slice(None)] * image.ndim
        slices[axis] = slice(start, end)
        output += weight * padded[tuple(slices)]
    return output


def _normalize_channels(image: np.ndarray) -> np.ndarray:
    output = np.empty_like(image, dtype=np.float32)
    for channel in range(image.shape[-1]):
        values = image[..., channel]
        low, high = np.percentile(values, (1.0, 99.0))
        if high <= low:
            output[..., channel] = 0.0
            continue
        output[..., channel] = (values - low) * 255.0 / (high - low)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray")
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("frame must have shape [H, W, 3]")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("frame height and width must be positive")
    if frame.dtype != np.uint8:
        raise TypeError("frame must use uint8 dtype")


def _validate_config(config: LowLightConfig) -> None:
    if config.mode not in ("disabled", "clahe", "retinex", "clahe_retinex"):
        raise ValueError("unsupported enhancement mode")
    if config.color_order not in ("RGB", "BGR"):
        raise ValueError("color_order must be RGB or BGR")
    if config.clahe_clip_limit <= 0:
        raise ValueError("clahe_clip_limit must be positive")
    tiles_y, tiles_x = config.clahe_tile_grid_size
    if tiles_y <= 0 or tiles_x <= 0:
        raise ValueError("clahe_tile_grid_size dimensions must be positive")
    if config.retinex_sigma <= 0:
        raise ValueError("retinex_sigma must be positive")


__all__ = [
    "LowLightConfig",
    "apply_clahe",
    "apply_retinex",
    "enhance_low_light",
]
