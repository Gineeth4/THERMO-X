"""
Simulator adapter for THERMO-X.

This provides a small abstraction around the existing
VirtualThermoXSystem.

The purpose is to make the experiment layer independent
from the internal simulator implementation.

Future architecture:

    SimulatorAdapter
          OR
    SerialAdapter
          OR
    MQTTAdapter
          ↓
    THERMO-X application
"""

from __future__ import annotations

from typing import Sequence

from core.models import (
    ActuatorCommand,
    SensorSnapshot,
)

from simulator.thermal_system import (
    VirtualThermoXSystem,
)


class SimulatorAdapter:
    """
    Adapter around the software-only THERMO-X plant.
    """

    def __init__(
        self,
        system: VirtualThermoXSystem,
    ) -> None:

        self.system = system

    def step(
        self,
        timestamp_s: float,
        dt_s: float,
    ) -> tuple[
        Sequence[SensorSnapshot],
        ActuatorCommand,
    ]:
        """
        Execute one simulator cycle.

        The simulator internally performs:

            workload
                ↓
            thermal model
                ↓
            sensors
                ↓
            controller
                ↓
            actuators
        """

        return self.system.step(
            timestamp_s=timestamp_s,
            dt_s=dt_s,
        )