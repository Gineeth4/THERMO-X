"""
Reusable thermal and electrical engineering calculations for THERMO-X.
"""

from __future__ import annotations

from math import isfinite


def clamp(value: float, lower: float, upper: float) -> float:
    """
    Restrict a value to a specified range.

    Example:

        clamp(120, 0, 100) -> 100
    """

    return max(lower, min(value, upper))


def liters_per_minute_to_m3_per_s(flow_lpm: float) -> float:
    """Convert L/min to m^3/s."""

    if flow_lpm < 0:
        raise ValueError("Flow rate cannot be negative.")

    return flow_lpm / 1000.0 / 60.0


def mass_flow_rate_kg_s(
    flow_lpm: float,
    density_kg_m3: float,
) -> float:
    """
    Convert volumetric flow to mass flow.

    m_dot = volumetric_flow × density
    """

    if density_kg_m3 <= 0:
        raise ValueError("Fluid density must be positive.")

    return (
        liters_per_minute_to_m3_per_s(flow_lpm)
        * density_kg_m3
    )


def heat_transfer_rate_w(
    flow_lpm: float,
    density_kg_m3: float,
    specific_heat_j_kg_k: float,
    delta_t_c: float,
) -> float:
    """
    Estimate coolant heat-transfer rate.

    Q_dot = m_dot × Cp × DeltaT

    IMPORTANT:
    The caller must supply properties for the actual coolant.

    Do not automatically use water properties when the physical
    prototype uses another coolant.
    """

    if specific_heat_j_kg_k <= 0:
        raise ValueError("Specific heat must be positive.")

    mass_flow = mass_flow_rate_kg_s(
        flow_lpm,
        density_kg_m3,
    )

    heat_w = (
        mass_flow
        * specific_heat_j_kg_k
        * delta_t_c
    )

    if not isfinite(heat_w):
        raise ValueError(
            "Heat-transfer calculation produced a non-finite value."
        )

    return heat_w


def electrical_power_w(
    voltage_v: float,
    current_a: float,
) -> float:
    """Calculate electrical power from voltage and current."""

    if voltage_v < 0 or current_a < 0:
        raise ValueError(
            "Voltage and current cannot be negative."
        )

    return voltage_v * current_a


def energy_from_power_wh(
    power_w: float,
    duration_s: float,
) -> float:
    """
    Convert power over time into watt-hours.

    Energy = Power × time
    """

    if power_w < 0 or duration_s < 0:
        raise ValueError(
            "Power and duration cannot be negative."
        )

    return power_w * duration_s / 3600.0