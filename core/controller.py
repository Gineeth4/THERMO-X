"""
Level-2 adaptive thermal controller for THERMO-X.

The controller calculates a Thermal Demand Index (TDI)
from several thermal indicators and maps that demand to:

    - pump speed
    - fan speed
    - branch valve positions

A hard safety layer is evaluated afterwards and can override
the adaptive command.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from .models import (
    ActuatorCommand,
    CoolingMode,
    SensorSnapshot,
    ThermalMetrics,
)
from .safety import (
    SafetyController,
    apply_safety_override,
)


@dataclass(slots=True)
class ControllerConfig:
    """Tunable parameters for THERMO-X adaptive control."""

    # Simulation-control target.
    #
    # NOT a physical safety specification.
    target_temperature_c: float = 47.0

    warning_temperature_c: float = 58.0

    temperature_scale_c: float = 15.0

    rise_rate_scale_c_per_s: float = 0.08

    workload_weight: float = 0.30

    temperature_weight: float = 0.40

    rise_rate_weight: float = 0.20

    ambient_weight: float = 0.10

    minimum_pump_pct: float = 20.0

    minimum_fan_pct: float = 20.0


class AdaptiveController:
    """
    THERMO-X rule-based adaptive controller.

    Important characteristics:

    1. Uses multiple inputs.
    2. Produces continuous rather than only LOW/MEDIUM/HIGH commands.
    3. Supports multiple thermal zones.
    4. Safety evaluation always happens after optimization.
    """

    def __init__(
        self,
        config: ControllerConfig | None = None,
        safety_controller: SafetyController | None = None,
    ) -> None:

        self.config = config or ControllerConfig()

        self.safety = (
            safety_controller
            or SafetyController()
        )

        # Previous temperature samples.
        #
        # Used to estimate dT/dt.
        #
        # server_id -> (timestamp, temperature)
        self._previous_temperatures: dict[
            str,
            tuple[float, float],
        ] = {}

    def _temperature_rise_rate(
        self,
        snapshot: SensorSnapshot,
    ) -> float:
        """
        Estimate dT/dt for one thermal module.

        dT/dt = (current_T - previous_T) / elapsed_time
        """

        previous = self._previous_temperatures.get(
            snapshot.server_id
        )

        # No historical data on first sample.
        if previous is None:
            return 0.0

        previous_time, previous_temp = previous

        dt = (
            snapshot.timestamp_s
            - previous_time
        )

        if dt <= 0:
            return 0.0

        return (
            snapshot.temperature_c
            - previous_temp
        ) / dt

    def thermal_demand_index(
        self,
        snapshot: SensorSnapshot,
    ) -> tuple[float, float]:
        """
        Calculate:

            TDI
            temperature rise rate

        TDI is normalized to approximately 0-1.
        """

        config = self.config

        rise_rate = self._temperature_rise_rate(
            snapshot
        )

        # -------------------------------
        # 1. Workload contribution
        # -------------------------------

        workload_term = max(
            0.0,
            min(
                snapshot.workload_pct / 100.0,
                1.0,
            ),
        )

        # -------------------------------
        # 2. Temperature contribution
        # -------------------------------

        temperature_term = max(
            0.0,
            min(
                (
                    snapshot.temperature_c
                    - config.target_temperature_c
                )
                / config.temperature_scale_c,
                1.0,
            ),
        )

        # -------------------------------
        # 3. Temperature-rise contribution
        # -------------------------------

        rise_rate_term = max(
            0.0,
            min(
                rise_rate
                / config.rise_rate_scale_c_per_s,
                1.0,
            ),
        )

        # -------------------------------
        # 4. Ambient contribution
        # -------------------------------

        # Initial simulation normalization:
        # 20 °C -> 0
        # 40 °C -> 1
        #
        # This is a tunable model assumption.
        ambient_term = max(
            0.0,
            min(
                (
                    snapshot.ambient_temperature_c
                    - 20.0
                ) / 20.0,
                1.0,
            ),
        )

        # -------------------------------
        # Weighted combination
        # -------------------------------

        tdi = (
            config.workload_weight * workload_term
            + config.temperature_weight * temperature_term
            + config.rise_rate_weight * rise_rate_term
            + config.ambient_weight * ambient_term
        )

        tdi = max(
            0.0,
            min(tdi, 1.0),
        )

        if not isfinite(tdi):
            raise ValueError(
                "TDI calculation produced a non-finite value."
            )

        return tdi, rise_rate

    def control_step(
        self,
        snapshots: Sequence[SensorSnapshot],
    ) -> tuple[
        ActuatorCommand,
        list[ThermalMetrics],
    ]:
        """
        Generate the next adaptive cooling command.

        Returns:
            ActuatorCommand
            list of thermal metrics
        """

        # No data -> safe shutdown.
        if not snapshots:

            command = ActuatorCommand(
                pump_speed_pct=0.0,
                fan_speed_pct=0.0,
                heater_enabled=False,
                cooling_mode=CoolingMode.SHUTDOWN,
                reason="No sensor data.",
            )

            decision = self.safety.evaluate(
                snapshots
            )

            return (
                apply_safety_override(
                    command,
                    decision,
                ),
                [],
            )

        metrics: list[ThermalMetrics] = []
        tdi_values: list[float] = []

        # Calculate thermal demand for every zone.
        for snapshot in snapshots:

            tdi, rise_rate = (
                self.thermal_demand_index(
                    snapshot
                )
            )

            metrics.append(
                ThermalMetrics(
                    coolant_delta_t_c=(
                        snapshot.coolant_delta_t_c
                    ),
                    thermal_demand_index=tdi,
                )
            )

            tdi_values.append(tdi)

            # Save this sample for the next dT/dt calculation.
            self._previous_temperatures[
                snapshot.server_id
            ] = (
                snapshot.timestamp_s,
                snapshot.temperature_c,
            )

        # --------------------------------
        # Shared pump/fan demand
        # --------------------------------

        # A shared loop must accommodate the zone
        # with the greatest current demand.
        max_tdi = max(tdi_values)

        config = self.config

        pump_speed = (
            config.minimum_pump_pct
            + (
                100.0
                - config.minimum_pump_pct
            )
            * max_tdi
        )

        fan_speed = (
            config.minimum_fan_pct
            + (
                100.0
                - config.minimum_fan_pct
            )
            * max_tdi
        )

        # --------------------------------
        # Individual branch control
        # --------------------------------

        valve_positions = {}

        for snapshot, tdi in zip(
            snapshots,
            tdi_values,
        ):

            valve_positions[
                snapshot.server_id
            ] = (
                config.minimum_pump_pct
                + (
                    100.0
                    - config.minimum_pump_pct
                )
                * tdi
            )

        command = ActuatorCommand(
            pump_speed_pct=pump_speed,
            fan_speed_pct=fan_speed,
            valve_positions_pct=valve_positions,
            heater_enabled=True,
            cooling_mode=CoolingMode.ADAPTIVE,
            reason="Adaptive controller active.",
        )

        # --------------------------------
        # Safety gets final authority
        # --------------------------------

        decision = self.safety.evaluate(
            snapshots
        )

        command = apply_safety_override(
            command,
            decision,
        )

        return command, metrics