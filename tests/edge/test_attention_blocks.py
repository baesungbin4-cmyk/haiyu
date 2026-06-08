import torch

from edge.detect.attention_blocks import SEResidualAttention


def test_se_residual_attention_preserves_shape() -> None:
    block = SEResidualAttention(channels=8, reduction=4)
    x = torch.randn(2, 8, 16, 12)

    y = block(x)

    assert y.shape == x.shape


def test_se_residual_attention_backward_has_gradients() -> None:
    block = SEResidualAttention(channels=8, reduction=4)
    x = torch.randn(2, 8, 8, 8, requires_grad=True)

    loss = block(x).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    parameter_grads = [param.grad for param in block.parameters()]
    assert all(grad is not None for grad in parameter_grads)
    assert all(torch.isfinite(grad).all() for grad in parameter_grads)


def test_se_residual_attention_output_has_no_nan() -> None:
    block = SEResidualAttention(channels=4, reduction=8)
    x = torch.randn(1, 4, 5, 7)

    y = block(x)

    assert torch.isfinite(y).all()


def test_se_residual_attention_residual_path_can_act_as_identity() -> None:
    block = SEResidualAttention(channels=6, reduction=3, residual_scale=0.0)
    x = torch.randn(2, 6, 9, 9)

    y = block(x)

    assert torch.equal(y, x)


def test_se_residual_attention_residual_path_is_additive() -> None:
    block = SEResidualAttention(channels=8, reduction=4, residual_scale=1.0)
    x = torch.randn(2, 8, 16, 16)

    y = block(x)

    assert not torch.equal(y, x)
    with torch.no_grad():
        weights = block.attention(block.pool(x))
        expected_attended = x * weights
        assert torch.allclose(y - x, expected_attended, atol=1e-6)


def test_se_residual_attention_stability_with_large_input() -> None:
    block = SEResidualAttention(channels=16, reduction=4)
    x = torch.randn(1, 16, 32, 32) * 100.0

    y = block(x)

    assert torch.isfinite(y).all()
    assert y.abs().max() < 1e6


def test_se_residual_attention_hidden_channels_edge_cases() -> None:
    small = SEResidualAttention(channels=1, reduction=16)
    narrow = SEResidualAttention(channels=16, reduction=16)
    wide = SEResidualAttention(channels=1024, reduction=16)

    assert small.hidden_channels == 1
    assert narrow.hidden_channels == 4
    assert wide.hidden_channels == 64
    assert small(torch.randn(1, 1, 3, 3)).shape == (1, 1, 3, 3)
    assert narrow(torch.randn(1, 16, 4, 4)).shape == (1, 16, 4, 4)
    assert wide(torch.randn(1, 1024, 2, 2)).shape == (1, 1024, 2, 2)
