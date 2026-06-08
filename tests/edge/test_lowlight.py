import numpy as np

from edge.enhance.lowlight import (
    LowLightConfig,
    apply_clahe,
    apply_retinex,
    enhance_low_light,
)


def _sample_frame() -> np.ndarray:
    height, width = 24, 32
    x_grad = np.linspace(10, 90, width, dtype=np.float32)
    y_grad = np.linspace(0, 40, height, dtype=np.float32)[:, None]
    base = x_grad + y_grad
    frame = np.stack(
        [
            base,
            np.clip(base * 0.8 + 8, 0, 255),
            np.clip(base * 0.6 + 16, 0, 255),
        ],
        axis=-1,
    )
    return np.clip(np.rint(frame), 0, 255).astype(np.uint8)


def test_enhancement_modes_keep_shape_dtype_and_finite_values() -> None:
    frame = _sample_frame()

    for mode in ("disabled", "clahe", "retinex", "clahe_retinex"):
        output = enhance_low_light(
            frame,
            LowLightConfig(
                mode=mode,
                retinex_sigma=2.0,
                clahe_tile_grid_size=(4, 4),
            ),
        )

        assert output.shape == frame.shape
        assert output.dtype == np.uint8
        assert np.isfinite(output).all()


def test_disabled_mode_returns_equivalent_copy() -> None:
    frame = _sample_frame()

    output = enhance_low_light(frame, LowLightConfig(mode="disabled"))

    assert np.array_equal(output, frame)
    assert output is not frame
    assert not np.shares_memory(output, frame)


def test_apply_clahe_keeps_output_valid_for_rgb_frame() -> None:
    frame = _sample_frame()

    output = apply_clahe(
        frame,
        LowLightConfig(
            mode="clahe",
            color_order="RGB",
            clahe_tile_grid_size=(4, 4),
        ),
    )

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert np.isfinite(output).all()


def test_apply_retinex_keeps_output_valid_for_bgr_frame() -> None:
    frame = _sample_frame()

    output = apply_retinex(
        frame,
        LowLightConfig(mode="retinex", color_order="BGR", retinex_sigma=2.0),
    )

    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert np.isfinite(output).all()
