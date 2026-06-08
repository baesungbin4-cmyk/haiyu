"""LSTM trajectory predictor with causal self-attention."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def create_causal_mask(
    seq_len: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return an additive mask where future positions are ``-inf``.

    The returned tensor has shape ``[seq_len, seq_len]``. Query position ``t``
    can attend only to keys ``<= t``.
    """

    if isinstance(seq_len, bool) or not isinstance(seq_len, int):
        raise TypeError("seq_len must be a positive integer")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating point dtype")

    blocked = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    return mask.masked_fill(blocked, float("-inf"))


class LSTMCausalAttentionPredictor(nn.Module):
    """Predict future ``x,y`` coordinates from historical trajectory features.

    Input shape is ``[batch, history_len, feature_dim]``. Output shape is
    ``[batch, future_len, 2]``.
    """

    def __init__(
        self,
        feature_dim: int,
        future_len: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self._validate_config(
            feature_dim=feature_dim,
            future_len=future_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feature_dim = feature_dim
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, future_len * 2),
        )

    @staticmethod
    def _validate_config(**values: Any) -> None:
        int_fields = (
            "feature_dim",
            "future_len",
            "hidden_dim",
            "num_layers",
            "num_heads",
        )
        for field in int_fields:
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{field} must be positive")

        dropout = values["dropout"]
        if not isinstance(dropout, float | int) or isinstance(dropout, bool):
            raise TypeError("dropout must be a number")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if values["hidden_dim"] % values["num_heads"] != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, history_len, feature]")
        if x.shape[-1] != self.feature_dim:
            raise ValueError("x feature dimension does not match model")
        if x.shape[1] <= 0:
            raise ValueError("history_len must be positive")
        if not torch.is_floating_point(x):
            raise TypeError("x must use a floating point dtype")

    def encode_history(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Encode history with a unidirectional LSTM and causal attention."""

        self._validate_input(x)
        encoded, _ = self.lstm(x)
        seq_len = encoded.shape[1]
        mask = create_causal_mask(
            seq_len,
            device=encoded.device,
            dtype=encoded.dtype,
        )
        attended, weights = self.attention(
            encoded,
            encoded,
            encoded,
            attn_mask=mask,
            need_weights=return_attention_weights,
        )
        contextual = self.norm(encoded + attended)
        if return_attention_weights:
            return contextual, weights
        return contextual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contextual = self.encode_history(x)
        last_context = contextual[:, -1, :]
        prediction = self.output_head(last_context)
        return prediction.view(x.shape[0], self.future_len, 2)


__all__ = ["LSTMCausalAttentionPredictor", "create_causal_mask"]
