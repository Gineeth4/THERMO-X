"""
Hardware-independent interfaces for THERMO-X.

The purpose of this file is to prevent the control system from
depending directly on a specific hardware implementation.

Therefore:

    Simulator ---> Interface ---> Control Engine

can later become:

    ESP32 ------> Interface ---> Control Engine
"""

from __future__ import annotations

from typing import Protocol, Sequence

from .models import ActuatorCommand, SensorSnapshot


class SensorProvider(Protocol):
    """
    Interface for anything capable of providing sensor measurements.

    Both the virtual sensor system and future ESP32 communication
    layer can implement this interface.
    """

    def read(self, timestamp_s: float) -> Sequence[SensorSnapshot]:
        """Return the latest sensor snapshots."""
        ...


class ActuatorInterface(Protocol):
    """
    Interface for anything capable of accepting actuator commands.
    """

    def apply(self, command: ActuatorCommand) -> None:
        """Apply a control command."""
        ...


class WorkloadProvider(Protocol):
    """
    Interface for workload/thermal-load generation.
    """

    def get_workload(
        self,
        timestamp_s: float,
        server_id: str,
    ) -> tuple[float, float]:
        """
        Return:

            workload_percent
            heater_power_w

        Heater power is an explicitly configured experimental
        thermal load. It is NOT a claim about real GPU power.
        """
        ...