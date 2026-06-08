import math

import pytest

from edge.risk.tcpa_cpa import calculate_tcpa_cpa


def test_tcpa_cpa_zero_relative_velocity_uses_current_distance() -> None:
    result = calculate_tcpa_cpa((30.0, 40.0), (0.0, 0.0))

    assert result.tcpa_seconds == 0.0
    assert result.cpa_distance_m == 50.0


def test_tcpa_cpa_approaching_target() -> None:
    result = calculate_tcpa_cpa((100.0, 0.0), (-5.0, 0.0))

    assert result.tcpa_seconds == 20.0
    assert result.cpa_distance_m == 0.0


def test_tcpa_cpa_leaving_target_has_negative_tcpa() -> None:
    result = calculate_tcpa_cpa((100.0, 0.0), (5.0, 0.0))

    assert result.tcpa_seconds == -20.0
    assert result.cpa_distance_m == 0.0


def test_tcpa_cpa_crossing_target() -> None:
    result = calculate_tcpa_cpa((100.0, 10.0), (-5.0, 0.0))

    assert result.tcpa_seconds == 20.0
    assert result.cpa_distance_m == 10.0


def test_tcpa_cpa_already_passed_closest_point() -> None:
    result = calculate_tcpa_cpa((0.0, 10.0), (0.0, 2.0))

    assert result.tcpa_seconds == -5.0
    assert result.cpa_distance_m == 0.0


def test_tcpa_cpa_validates_vector_shape_and_finiteness() -> None:
    with pytest.raises(ValueError, match="two numbers"):
        calculate_tcpa_cpa((1.0,), (0.0, 0.0))

    with pytest.raises(ValueError, match="finite"):
        calculate_tcpa_cpa((math.inf, 0.0), (0.0, 0.0))
