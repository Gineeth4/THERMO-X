"""
Lumped thermal model for THERMO-X.

This is a simplified engineering model, NOT a CFD simulation.

The model contains:

    simulated heater
        ↓
    thermal mass
        ↓
    cold plate
        ↓
    coolant loop
        ↓
    radiator
        ↓
    ambient

The model is intentionally understandable and calibratable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from core.models import ActuatorCommand
from core.thermal_calculations import (
    clamp,
    mass_flow_rate_kg_s,
)

from .parameters import (
    CoolingSystemParameters,
    FluidProperties,
    ThermalModuleParameters,
)


@dataclass(slots=True)
class ThermalModuleState:
    """Dynamic state of one simulated thermal module."""

    temperature_c: float

    heater_power_w: float = 0.0

    heat_removed_w: float = 0.0

    coolant_inlet_c: float = 30.0

    coolant_outlet_c: float = 30.0

    branch_flow_lpm: float = 0.0


class ThermalSystemModel:
    """
    Simplified multi-zone THERMO-X thermal plant.
    """

    def __init__(
        self,
        server_ids: list[str],
        module_params: ThermalModuleParameters | None = None,
        cooling_params: CoolingSystemParameters | None = None,
        fluid: FluidProperties | None = None,
    ) -> None:

        self.server_ids = server_ids

        self.module_params = (
            module_params
            or ThermalModuleParameters()
        )

        self.cooling_params = (
            cooling_params
            or CoolingSystemParameters()
        )

        self.fluid = fluid or FluidProperties()

        # Create thermal state for every simulated zone.
        self.modules: dict[
            str,
            ThermalModuleState,
        ] = {
            server_id: ThermalModuleState(
                temperature_c=(
                    self.module_params
                    .initial_temperature_c
                ),
                coolant_inlet_c=(
                    self.cooling_params
                    .initial_coolant_temperature_c
                ),
                coolant_outlet_c=(
                    self.cooling_params
                    .initial_coolant_temperature_c
                ),
            )
            for server_id in server_ids
        }

        # Shared coolant/reservoir temperature.
        self.coolant_temperature_c = (
            self.cooling_params
            .initial_coolant_temperature_c
        )

    def _total_flow_lpm(
        self,
        pump_speed_pct: float,
    ) -> float:
        """
        Convert pump command into simulated total flow.

        This is a simple linear pump model for the first
        software simulation.
        """

        cooling = self.cooling_params

        speed = (
            clamp(
                pump_speed_pct,
                0.0,
                100.0,
            )
            / 100.0
        )

        return (
            cooling.minimum_total_flow_lpm
            + (
                cooling.maximum_total_flow_lpm
                - cooling.minimum_total_flow_lpm
            )
            * speed
        )

    def _branch_flows(
        self,
        total_flow_lpm: float,
        valve_positions_pct: Mapping[
            str,
            float,
        ],
    ) -> dict[str, float]:
        """
        Split shared pump flow between zones.

        Flow allocation is proportional to valve demand.
        """

        if not self.server_ids:
            return {}

        demands = {
            server_id: max(
                0.0,
                float(
                    valve_positions_pct.get(
                        server_id,
                        0.0,
                    )
                ),
            )
            for server_id in self.server_ids
        }

        total_demand = sum(
            demands.values()
        )

        # If there is no valve demand, distribute flow
        # equally so the simulation remains stable.
        if total_demand <= 0:
            equal_share = (
                total_flow_lpm
                / len(self.server_ids)
            )

            return {
                server_id: equal_share
                for server_id in self.server_ids
            }

        return {
            server_id: (
                total_flow_lpm
                * demands[server_id]
                / total_demand
            )
            for server_id in self.server_ids
        }

    def step(
        self,
        heater_powers_w: Mapping[str, float],
        command: ActuatorCommand,
        dt_s: float,
    ) -> dict[str, ThermalModuleState]:
        """
        Advance the simulation by dt_s seconds.

        This implements the simplified thermal energy balance.
        """

        if dt_s <= 0:
            raise ValueError(
                "Simulation time step must be positive."
            )

        module_params = self.module_params
        cooling_params = self.cooling_params

        # -----------------------------------------
        # Calculate total pump flow.
        # -----------------------------------------

        total_flow_lpm = self._total_flow_lpm(
            command.pump_speed_pct
        )

        # -----------------------------------------
        # Allocate flow to branches.
        # -----------------------------------------

        branches = self._branch_flows(
            total_flow_lpm,
            command.valve_positions_pct or {},
        )

        total_heat_removed_w = 0.0

        # Used to calculate mixed coolant return temperature.
        return_temperature_numerator = 0.0
        return_flow_total_lpm = 0.0

        # -----------------------------------------
        # Simulate each thermal module.
        # -----------------------------------------

        for server_id in self.server_ids:

            state = self.modules[server_id]

            configured_power = max(
                0.0,
                float(
                    heater_powers_w.get(
                        server_id,
                        0.0,
                    )
                ),
            )

            # Safety controller may disable the heat source.
            if command.heater_enabled:
                heater_power = min(
                    configured_power,
                    module_params.max_heater_power_w,
                )
            else:
                heater_power = 0.0

            branch_flow_lpm = branches.get(
                server_id,
                0.0,
            )

            # -------------------------------------
            # Model the effect of branch flow
            # on cold-plate heat transfer.
            # -------------------------------------

            flow_fraction = clamp(
                branch_flow_lpm
                / cooling_params.maximum_total_flow_lpm,
                0.0,
                1.0,
            )

            # At very low flow the cold plate still has
            # some conduction, but its effectiveness is reduced.
            ua_factor = (
                0.35
                + 0.65 * flow_fraction
            )

            effective_ua = (
                module_params.cold_plate_ua_w_per_k
                * ua_factor
            )

            # Temperature driving force between the
            # thermal module and coolant.
            temperature_difference = max(
                0.0,
                state.temperature_c
                - self.coolant_temperature_c,
            )

            potential_heat_removal_w = (
                effective_ua
                * temperature_difference
            )

            # During this simplified step, the coolant
            # cannot remove more thermal input than the
            # heater is currently generating.
            heat_removed_w = min(
                heater_power,
                potential_heat_removal_w,
            )

            # -------------------------------------
            # Calculate coolant temperature rise.
            # -------------------------------------

            mass_flow_kg_s = (
                mass_flow_rate_kg_s(
                    branch_flow_lpm,
                    self.fluid.density_kg_m3,
                )
            )

            if mass_flow_kg_s > 0:
                coolant_rise_c = (
                    heat_removed_w
                    / (
                        mass_flow_kg_s
                        * self.fluid
                        .specific_heat_j_kg_k
                    )
                )
            else:
                coolant_rise_c = 0.0

            coolant_outlet_c = (
                self.coolant_temperature_c
                + coolant_rise_c
            )

            # -------------------------------------
            # Thermal mass equation.
            #
            # C_th dT/dt =
            #     P_heat - Q_cooling
            # -------------------------------------

            net_heat_w = (
                heater_power
                - heat_removed_w
            )

            temperature_change_c = (
                net_heat_w
                * dt_s
                / module_params
                .thermal_mass_j_per_k
            )

            state.temperature_c += (
                temperature_change_c
            )

            # Store current state.
            state.heater_power_w = (
                heater_power
            )

            state.heat_removed_w = (
                heat_removed_w
            )

            state.coolant_inlet_c = (
                self.coolant_temperature_c
            )

            state.coolant_outlet_c = (
                coolant_outlet_c
            )

            state.branch_flow_lpm = (
                branch_flow_lpm
            )

            # Accumulate values for mixed return.
            total_heat_removed_w += (
                heat_removed_w
            )

            return_temperature_numerator += (
                coolant_outlet_c
                * branch_flow_lpm
            )

            return_flow_total_lpm += (
                branch_flow_lpm
            )

        # -----------------------------------------
        # Calculate mixed coolant return.
        # -----------------------------------------

        if return_flow_total_lpm > 0:
            mixed_return_c = (
                return_temperature_numerator
                / return_flow_total_lpm
            )
        else:
            mixed_return_c = (
                self.coolant_temperature_c
            )

        # -----------------------------------------
        # Radiator heat rejection.
        # -----------------------------------------

        fan_fraction = (
            clamp(
                command.fan_speed_pct,
                0.0,
                100.0,
            )
            / 100.0
        )

        radiator_heat_rejection_w = max(
            0.0,
            cooling_params.radiator_ua_w_per_k
            * fan_fraction
            * max(
                0.0,
                mixed_return_c
                - cooling_params
                .ambient_temperature_c,
            ),
        )

        # Don't reject more heat than entered the coolant
        # during this simple time step.
        radiator_heat_rejection_w = min(
            radiator_heat_rejection_w,
            max(
                0.0,
                total_heat_removed_w,
            ),
        )

        # -----------------------------------------
        # Update shared coolant thermal energy.
        # -----------------------------------------

        net_coolant_heat_w = (
            total_heat_removed_w
            - radiator_heat_rejection_w
        )

        self.coolant_temperature_c += (
            net_coolant_heat_w
            * dt_s
            / cooling_params
            .coolant_heat_capacity_j_per_k
        )

        # The simplified radiator is not allowed to
        # cool the fluid below ambient temperature.
        self.coolant_temperature_c = max(
            self.coolant_temperature_c,
            cooling_params.ambient_temperature_c,
        )

        # Return copies so callers do not accidentally
        # mutate the internal model state.
        return {
            server_id: ThermalModuleState(
                temperature_c=state.temperature_c,
                heater_power_w=state.heater_power_w,
                heat_removed_w=state.heat_removed_w,
                coolant_inlet_c=state.coolant_inlet_c,
                coolant_outlet_c=state.coolant_outlet_c,
                branch_flow_lpm=state.branch_flow_lpm,
            )
            for server_id, state in self.modules.items()
        }