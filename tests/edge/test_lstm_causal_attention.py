import torch
import pytest

from edge.predict.lstm_causal_attention import (
    LSTMCausalAttentionPredictor,
    create_causal_mask,
)


def test_create_causal_mask_shape_and_values() -> None:
    mask = create_causal_mask(4)

    assert mask.shape == (4, 4)
    assert torch.equal(torch.diag(mask), torch.zeros(4))
    assert torch.isneginf(mask[0, 1])
    assert torch.isneginf(mask[0, 3])
    assert mask[3, 0].item() == 0.0
    assert mask[3, 3].item() == 0.0


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.bfloat16],
)
def test_causal_mask_different_dtypes(dtype: torch.dtype) -> None:
    mask = create_causal_mask(3, dtype=dtype)

    assert mask.dtype == dtype
    assert torch.equal(torch.diag(mask), torch.zeros(3, dtype=dtype))
    assert torch.isneginf(mask[0, 1])
    assert mask[2, 0].item() == 0.0


@pytest.mark.parametrize(
    ("seq_len", "dtype", "error_type", "error_match"),
    [
        (True, torch.float32, TypeError, "positive integer"),
        (0, torch.float32, ValueError, "must be positive"),
        (3, torch.int32, TypeError, "floating point"),
    ],
)
def test_create_causal_mask_rejects_invalid_args(
    seq_len: int,
    dtype: torch.dtype,
    error_type: type[Exception],
    error_match: str,
) -> None:
    with pytest.raises(error_type, match=error_match):
        create_causal_mask(seq_len, dtype=dtype)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "error_match"),
    [
        (
            {
                "feature_dim": 7,
                "hidden_dim": 16,
                "future_len": 2,
                "num_heads": 3,
                "num_layers": 1,
                "dropout": 0.0,
            },
            ValueError,
            "divisible",
        ),
        (
            {
                "feature_dim": 7,
                "hidden_dim": 16,
                "future_len": 2,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 1.0,
            },
            ValueError,
            r"\[0, 1\)",
        ),
        (
            {
                "feature_dim": 7,
                "hidden_dim": 16,
                "future_len": 2,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": -0.1,
            },
            ValueError,
            r"\[0, 1\)",
        ),
        (
            {
                "feature_dim": 7,
                "hidden_dim": 16,
                "future_len": 2,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": True,
            },
            TypeError,
            "number",
        ),
        (
            {
                "feature_dim": 0,
                "hidden_dim": 16,
                "future_len": 2,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            ValueError,
            "positive",
        ),
        (
            {
                "feature_dim": 7,
                "hidden_dim": 16,
                "future_len": 0,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            ValueError,
            "positive",
        ),
        (
            {
                "feature_dim": 7,
                "hidden_dim": 16,
                "future_len": 2,
                "num_heads": 4,
                "num_layers": 0,
                "dropout": 0.0,
            },
            ValueError,
            "positive",
        ),
        (
            {
                "feature_dim": 7,
                "hidden_dim": 16,
                "future_len": 2,
                "num_heads": 0,
                "num_layers": 1,
                "dropout": 0.0,
            },
            ValueError,
            "positive",
        ),
        (
            {
                "feature_dim": True,
                "hidden_dim": 16,
                "future_len": 2,
                "num_heads": 4,
                "num_layers": 1,
                "dropout": 0.0,
            },
            TypeError,
            "positive integer",
        ),
    ],
)
def test_validate_config_rejects_invalid_params(
    kwargs: dict[str, object],
    error_type: type[Exception],
    error_match: str,
) -> None:
    with pytest.raises(error_type, match=error_match):
        LSTMCausalAttentionPredictor(**kwargs)


def test_predictor_output_shape() -> None:
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=5,
        num_heads=4,
    )
    x = torch.randn(3, 8, 7)

    y = model(x)

    assert y.shape == (3, 5, 2)


def test_predictor_backward_pass_has_finite_gradients() -> None:
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=3,
        num_heads=4,
    )
    x = torch.randn(2, 6, 7, requires_grad=True)

    loss = model(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    parameter_grads = [param.grad for param in model.parameters()]
    assert all(grad is not None for grad in parameter_grads)
    assert all(torch.isfinite(grad).all() for grad in parameter_grads)


def test_predictor_output_has_no_nan() -> None:
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=4,
        num_heads=4,
    )
    x = torch.randn(2, 5, 7)

    y = model(x)

    assert torch.isfinite(y).all()


def test_model_forward_with_history_len_1() -> None:
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=3,
        num_heads=4,
    )
    x = torch.randn(2, 1, 7)

    y = model(x)

    assert y.shape == (2, 3, 2)
    assert torch.isfinite(y).all()


def test_forward_prediction_depends_on_last_timestep() -> None:
    torch.manual_seed(99)
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=2,
        num_heads=4,
    )
    model.eval()
    x = torch.randn(1, 5, 7)
    changed = x.clone()
    changed[:, -1, :] += 50.0

    with torch.no_grad():
        original_prediction = model(x)
        changed_prediction = model(changed)

    assert not torch.allclose(
        original_prediction,
        changed_prediction,
        atol=1e-5,
    ), "Forward prediction must depend on the last history timestep"


def test_forward_prediction_depends_on_first_timestep() -> None:
    torch.manual_seed(101)
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=2,
        num_heads=4,
    )
    model.eval()
    x = torch.randn(1, 5, 7)
    changed = x.clone()
    changed[:, 0, :] += 50.0

    with torch.no_grad():
        original_prediction = model(x)
        changed_prediction = model(changed)

    message = "Forward prediction must depend on early history through LSTM"
    assert not torch.allclose(
        original_prediction,
        changed_prediction,
        atol=1e-5,
    ), message


def test_forward_and_encode_history_last_are_consistent() -> None:
    torch.manual_seed(103)
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=2,
        num_heads=4,
    )
    model.eval()
    x = torch.randn(3, 5, 7)

    with torch.no_grad():
        forward_prediction = model(x)
        encoded = model.encode_history(x)
        manual_prediction = model.output_head(encoded[:, -1, :]).view(3, 2, 2)

    assert torch.equal(forward_prediction, manual_prediction)


@pytest.mark.parametrize(
    ("x", "error_match"),
    [
        (torch.randn(2, 6), "must have shape"),
        (torch.randn(2, 6, 5), "feature dimension does not match"),
        (torch.randn(2, 0, 7), "history_len must be positive"),
    ],
)
def test_encode_history_rejects_invalid_input(
    x: torch.Tensor,
    error_match: str,
) -> None:
    model = LSTMCausalAttentionPredictor(feature_dim=7, future_len=2)

    with pytest.raises(ValueError, match=error_match):
        model.encode_history(x)


def test_encode_history_rejects_integer_input_dtype() -> None:
    model = LSTMCausalAttentionPredictor(feature_dim=7, future_len=2)
    x = torch.zeros(2, 5, 7, dtype=torch.int64)

    with pytest.raises(TypeError, match="floating point dtype"):
        model.encode_history(x)


def test_encode_history_default_returns_tensor_not_tuple() -> None:
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=2,
        num_heads=4,
    )
    x = torch.randn(2, 5, 7)

    encoded = model.encode_history(x)

    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape == (2, 5, 16)


def test_causal_encoding_earlier_outputs_ignore_future_input_changes() -> None:
    torch.manual_seed(7)
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=2,
        num_heads=4,
    )
    model.eval()
    x = torch.randn(1, 6, 7)
    changed = x.clone()
    changed[:, 4:, :] += 1000.0

    with torch.no_grad():
        original_encoding = model.encode_history(x)
        changed_encoding = model.encode_history(changed)

    assert torch.allclose(
        original_encoding[:, :4, :],
        changed_encoding[:, :4, :],
        atol=1e-6,
    )


def test_last_timestep_encoding_depends_on_early_history() -> None:
    torch.manual_seed(42)
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=2,
        num_heads=4,
    )
    model.eval()
    x = torch.randn(1, 6, 7)
    changed = x.clone()
    changed[:, 0, :] += 100.0

    with torch.no_grad():
        original_last = model.encode_history(x)[:, -1, :]
        changed_last = model.encode_history(changed)[:, -1, :]

    assert not torch.allclose(
        original_last, changed_last, atol=1e-5
    ), "Last timestep should depend on early history input"


def test_attention_weights_do_not_include_future_positions() -> None:
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=2,
        num_heads=4,
    )
    model.eval()
    x = torch.randn(2, 5, 7)

    with torch.no_grad():
        _, weights = model.encode_history(x, return_attention_weights=True)

    assert weights is not None
    assert weights.shape == (2, 5, 5)
    future_weights = torch.triu(weights, diagonal=1)
    assert torch.equal(
        future_weights,
        torch.zeros_like(future_weights),
    ), "Future attention weights must be exactly zero"


def test_non_causal_mask_produces_future_attention_weights() -> None:
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=2,
        num_heads=4,
    )
    model.eval()
    x = torch.zeros(2, 5, 7)

    with torch.no_grad():
        model.attention.in_proj_weight.zero_()
        if model.attention.in_proj_bias is not None:
            model.attention.in_proj_bias.zero_()
        encoded, _ = model.lstm(x)
        _, weights_no_mask = model.attention(
            encoded,
            encoded,
            encoded,
            need_weights=True,
        )

    assert weights_no_mask is not None
    future_weights = torch.triu(weights_no_mask, diagonal=1)
    assert (
        torch.count_nonzero(future_weights) > 0
    ), "Without causal mask, attention should attend to future positions"
