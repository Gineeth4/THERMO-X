"""
Virtual THERMO-X system.

This class connects:

    workload
        ↓
    thermal model
        ↓
    virtual sensors
        ↓
    adaptive controller
        ↓
    virtual actuators

It acts as the software equivalent of the future physical
ESP32-controlled thermal prototype.
"""

from __future__ import annotations

from typing import Sequence

from core.controller import AdaptiveController
from core.models import (
    ActuatorCommand,
    SensorSnapshot,
)

from thermal_model.model import ThermalSystemModel

from .virtual_actuators import VirtualActuatorBank
from .virtual_sensors import VirtualSensorBank
from .workload_generator import WorkloadProfile


class VirtualThermoXSystem:
    """Complete software representation of the THERMO-X MVP."""

    def __init__(
        self,
        server_ids: list[str],
        workload_profile: WorkloadProfile,
        model: ThermalSystemModel,
        controller: AdaptiveController,
        sensors: VirtualSensorBank,
        actuators: VirtualActuatorBank,
    ) -> None:

        self.server_ids = server_ids

        self.workload_profile = (
            workload_profile
        )

        self.model = model

        self.controller = controller

        self.sensors = sensors

        self.actuators = actuators

        # Start with moderate cooling and equal
        # branch valve opening.
        self.actuators.apply(
            ActuatorCommand(
                pump_speed_pct=30.0,
                fan_speed_pct=30.0,
                valve_positions_pct={
                    server_id: 50.0
                    for server_id
                    in server_ids
                },
            )
        )

    def step(
        self,
        timestamp_s: float,
        dt_s: float,
    ) -> tuple[
        Sequence[SensorSnapshot],
        ActuatorCommand,
    ]:
        """
        Execute one complete control cycle.

        Order:

            1. Determine workload
            2. Apply current actuator command to model
            3. Advance thermal state
            4. Read sensors
            5. Calculate next control command
            6. Apply next command
        """

        # -----------------------------------------
        # 1. Determine current workload.
        # -----------------------------------------

        workload_by_server = {
            server_id: (
                self.workload_profile
                .get_workload(
                    timestamp_s,
                    server_id,
                )
            )
            for server_id
            in self.server_ids
        }

        heater_powers = {
            server_id: values[1]
            for (
                server_id,
                values,
            ) in workload_by_server.items()
        }

        # -----------------------------------------
        # 2. Advance thermal plant.
        #
        # The current actuator command controls
        # this simulation step.
        # -----------------------------------------

        self.model.step(
            heater_powers_w=heater_powers,
            command=self.actuators.command,
            dt_s=dt_s,
        )

        # -----------------------------------------
        # 3. Read simulated sensors.
        # -----------------------------------------

        snapshots = self.sensors.read(
            timestamp_s=timestamp_s,
            workload_by_server=(
                workload_by_server
            ),
        )

        # -----------------------------------------
        # 4. Calculate the next controller command.
        # -----------------------------------------

        command, _metrics = (
            self.controller.control_step(
                snapshots
            )
        )

        # -----------------------------------------
        # 5. Apply command for the next step.
        # -----------------------------------------

        self.actuators.apply(
            command
        )

        return snapshots, command