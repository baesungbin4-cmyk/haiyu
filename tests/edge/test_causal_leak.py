"""Causal leak tests for the trajectory predictor (contract §10 / review item S2).

Contract §10 验收标准 requires: "篡改未来帧输入，检查 ≤t 预测不变".
These tests verify that the predictor does not leak future information through
the causal attention mask into earlier timestep predictions.
"""

import torch

from edge.predict.lstm_causal_attention import (
    LSTMCausalAttentionPredictor,
)


def test_future_input_tampering_does_not_change_past_predictions() -> None:
    """Tamper with future timestep inputs; verify ≤t predictions are stable.

    For an input sequence of T timesteps, modifying inputs at positions
    [t+1 .. T-1] must not alter the model's encoding at position t.
    This is the contract §10 causal-leak prevention check.
    """
    torch.manual_seed(2024)
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=32,
        future_len=3,
        num_heads=4,
        num_layers=1,
    )
    model.eval()

    x = torch.randn(2, 8, 7)
    changed = x.clone()
    # Tamper with timesteps 4-7 (indices 4,5,6,7)
    changed[:, 4:, :] += 1000.0

    with torch.no_grad():
        original_enc = model.encode_history(x)  # [B, 8, hidden]
        changed_enc = model.encode_history(changed)

    # Positions 0-3 must be unchanged (they are ≤3, and we only
    # tampered with positions 4+).
    assert torch.allclose(
        original_enc[:, :4, :],
        changed_enc[:, :4, :],
        atol=1e-5,
    ), (
        "Causal leak detected: tampering with future timesteps "
        "changed the encoding of earlier timesteps"
    )

    # Positions 4-7 should have changed (they include the tampered input).
    assert not torch.allclose(
        original_enc[:, 4:, :],
        changed_enc[:, 4:, :],
        atol=1e-3,
    ), "Expected tampered timesteps to have different encodings"


def test_causal_attention_mask_is_strictly_lower_triangular() -> None:
    """Verify the causal mask prevents any attention to future positions.

    The prediction (``forward``) uses the last LSTM timestep's encoding;
    through the LSTM recurrence every timestep sees *all previous*
    history — that is correct behaviour, not a leak.  The causal
    guarantee is at the self-attention level: position i can only attend
    to positions ≤ i.
    """
    torch.manual_seed(2025)
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=32,
        future_len=3,
        num_heads=4,
        num_layers=1,
    )
    model.eval()

    for seq_len in [3, 5, 8, 12]:
        x = torch.randn(2, seq_len, 7)
        with torch.no_grad():
            _, attn_weights = model.encode_history(x, return_attention_weights=True)
        assert attn_weights is not None
        assert attn_weights.shape == (
            2,
            seq_len,
            seq_len,
        ), f"unexpected attention shape for seq_len={seq_len}"
        # All future-position weights must be exactly zero
        future_weights = torch.triu(attn_weights, diagonal=1)
        assert future_weights.sum() == 0.0, (
            f"Causal mask violation at seq_len={seq_len}: "
            f"non-zero attention to future positions"
        )


def test_causal_mask_single_timestep_edge_case() -> None:
    """With a single history step there are no future positions to leak."""
    torch.manual_seed(2026)
    model = LSTMCausalAttentionPredictor(
        feature_dim=7,
        hidden_dim=16,
        future_len=2,
        num_heads=4,
    )
    model.eval()

    x = torch.randn(1, 1, 7)
    y = model(x)
    assert y.shape == (1, 2, 2)
    assert torch.isfinite(y).all()
