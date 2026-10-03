"""
Basic calibration utilities for the THERMO-X thermal model.

Physical prototype data can later be used to estimate effective
thermal parameters instead of relying entirely on simulation assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Summary of a basic thermal calibration estimate."""

    average_temperature_rise_c: float

    estimated_thermal_mass_j_per_k: float | None

    notes: str


def estimate_thermal_mass_from_step_test(
    power_w: Sequence[float],
    temperature_rate_c_per_s: Sequence[float],
) -> CalibrationResult:
    """
    Estimate an effective thermal mass from heating data.

    Simplified relationship:

        C_th ≈ P / (dT/dt)

    This approximation is most meaningful during a period where
    cooling losses are relatively small.

    For final physical calibration, we will use a more complete
    energy-balance identification procedure.
    """

    if (
        len(power_w)
        != len(temperature_rate_c_per_s)
        or not power_w
    ):
        raise ValueError(
            "Input sequences must have equal non-zero length."
        )

    valid_estimates = [
        p / rate
        for p, rate in zip(
            power_w,
            temperature_rate_c_per_s,
        )
        if p > 0 and rate > 0
    ]

    if not valid_estimates:
        return CalibrationResult(
            average_temperature_rise_c=0.0,
            estimated_thermal_mass_j_per_k=None,
            notes=(
                "No valid positive heating-rate samples "
                "were available."
            ),
        )

    median_mass = median(
        valid_estimates
    )

    average_rate = mean(
        temperature_rate_c_per_s
    )

    return CalibrationResult(
        average_temperature_rise_c=average_rate,
        estimated_thermal_mass_j_per_k=median_mass,
        notes=(
            "First-pass estimate only. "
            "Use measured system losses and a full energy "
            "balance for final calibration."
        ),
    )