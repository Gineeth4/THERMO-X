"""
Virtual actuators for the THERMO-X simulation.

These represent:

    pump
    fan
    branch valves
    heater enable/disable

No physical electrical output is produced.
"""

from __future__ import annotations

from core.models import ActuatorCommand


class VirtualActuatorBank:
    """Stores the latest command sent to the simulated hardware."""

    def __init__(
        self,
        initial_command: ActuatorCommand | None = None,
    ) -> None:

        self.command = (
            initial_command
            or ActuatorCommand()
        )

    def apply(
        self,
        command: ActuatorCommand,
    ) -> None:
        """
        Accept a controller command.

        Values are clamped to valid actuator ranges.
        """

        # ------------------------------------
        # Shared pump
        # ------------------------------------

        command.pump_speed_pct = max(
            0.0,
            min(
                command.pump_speed_pct,
                100.0,
            ),
        )

        # ------------------------------------
        # Shared fan
        # ------------------------------------

        command.fan_speed_pct = max(
            0.0,
            min(
                command.fan_speed_pct,
                100.0,
            ),
        )

        # ------------------------------------
        # Branch valves
        # ------------------------------------

        if (
            command.valve_positions_pct
            is None
        ):
            command.valve_positions_pct = {}

        for (
            server_id,
            position,
        ) in list(
            command.valve_positions_pct.items()
        ):

            command.valve_positions_pct[
                server_id
            ] = max(
                0.0,
                min(
                    float(position),
                    100.0,
                ),
            )

        # Store validated command.
        self.command = command