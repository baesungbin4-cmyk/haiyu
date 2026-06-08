"""Closed-form TCPA/CPA utilities for 2D relative motion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import hypot, isfinite
from numbers import Real
from typing import TypeAlias

Vector2D: TypeAlias = tuple[float, float]

_ZERO_SPEED_SQUARED = 1e-12


@dataclass(frozen=True)
class TCPACPAResult:
    """Closest-point metrics for relative 2D motion."""

    tcpa_seconds: float
    cpa_distance_m: float


def coerce_vector2(value: Sequence[float], field_name: str) -> Vector2D:
    """Validate and coerce a finite 2D vector."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of two numbers")
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain two numbers")

    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"{field_name} must contain finite numbers")
        number = float(item)
        if not isfinite(number):
            raise ValueError(f"{field_name} values must be finite")
        result.append(number)
    return (result[0], result[1])


def vector_distance_m(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return Euclidean distance between two meter-space 2D positions."""

    first_x, first_y = coerce_vector2(first, "first")
    second_x, second_y = coerce_vector2(second, "second")
    return hypot(first_x - second_x, first_y - second_y)


def calculate_tcpa_cpa(
    relative_position_m: Sequence[float],
    relative_velocity_mps: Sequence[float],
) -> TCPACPAResult:
    """Calculate TCPA seconds and CPA meters for constant relative motion.

    ``relative_position_m`` is target position minus own position. Similarly,
    ``relative_velocity_mps`` is target velocity minus own velocity.
    """

    rel_x, rel_y = coerce_vector2(relative_position_m, "relative_position_m")
    vel_x, vel_y = coerce_vector2(
        relative_velocity_mps,
        "relative_velocity_mps",
    )

    speed_squared = vel_x * vel_x + vel_y * vel_y
    if speed_squared <= _ZERO_SPEED_SQUARED:
        return TCPACPAResult(
            tcpa_seconds=0.0,
            cpa_distance_m=hypot(rel_x, rel_y),
        )

    dot_product = rel_x * vel_x + rel_y * vel_y
    tcpa_seconds = -dot_product / speed_squared
    if abs(tcpa_seconds) <= _ZERO_SPEED_SQUARED:
        tcpa_seconds = 0.0
    cpa_x = rel_x + vel_x * tcpa_seconds
    cpa_y = rel_y + vel_y * tcpa_seconds
    return TCPACPAResult(
        tcpa_seconds=tcpa_seconds,
        cpa_distance_m=hypot(cpa_x, cpa_y),
    )


__all__ = [
    "TCPACPAResult",
    "Vector2D",
    "calculate_tcpa_cpa",
    "coerce_vector2",
    "vector_distance_m",
]
