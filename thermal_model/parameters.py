"""
THERMO-X simulation parameters.

IMPORTANT
---------
These values are SOFTWARE-SIMULATION PARAMETERS.

They are not physical prototype specifications.

When the physical prototype is built, these values must be
replaced/calibrated using actual component datasheets and
experimental measurements.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FluidProperties:
    """Thermophysical properties of the simulated coolant."""

    # Approximate water-like properties for simulation.
    #
    # The actual physical prototype must use measured or
    # manufacturer-specified fluid properties.
    density_kg_m3: float = 997.0

    specific_heat_j_kg_k: float = 4180.0


@dataclass(frozen=True, slots=True)
class ThermalModuleParameters:
    """
    Parameters describing one simulated thermal source.

    The values are deliberately chosen to create a meaningful
    thermal transient rather than an almost-flat temperature curve.
    """

    # Effective thermal mass of:
    #
    # heater + thermal block + interface
    #
    # A larger thermal mass produces thermal inertia.
    thermal_mass_j_per_k: float = 850.0

    # Effective cold-plate heat-transfer coefficient.
    #
    # This is intentionally much lower than the earlier value
    # so that a 20–100 W thermal load produces visible thermal
    # dynamics.
    cold_plate_ua_w_per_k: float = 4.5

    # Maximum simulated heater power.
    max_heater_power_w: float = 110.0

    # Initial thermal-source temperature.
    initial_temperature_c: float = 34.0


@dataclass(frozen=True, slots=True)
class CoolingSystemParameters:
    """Parameters for the simulated shared cooling system."""

    ambient_temperature_c: float = 30.0

    # Total simulated coolant-flow range.
    minimum_total_flow_lpm: float = 0.15

    maximum_total_flow_lpm: float = 2.80

    # Effective radiator heat-rejection coefficient.
    radiator_ua_w_per_k: float = 65.0

    # Effective shared coolant thermal capacity.
    coolant_heat_capacity_j_per_k: float = 3500.0

    # Initial coolant temperature.
    initial_coolant_temperature_c: float = 30.0

    # Software estimates only.
    pump_power_at_100_pct_w: float = 30.0

    fan_power_at_100_pct_w: float = 20.0