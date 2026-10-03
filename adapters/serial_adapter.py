"""
Serial adapter for future THERMO-X ESP32 communication.

Expected communication style:

    Laptop
       ↕
    USB Serial
       ↕
    ESP32

The protocol is intentionally simple JSON Lines:

    One JSON object per line.

Sensor message example:

    {
        "type": "sensor_batch",
        "sensors": [...]
    }

Actuator message example:

    {
        "type": "actuator_command",
        ...
    }

The exact ESP32 firmware protocol will be finalized later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from typing import Sequence

import serial

from core.models import (
    ActuatorCommand,
    SensorSnapshot,
)


class SerialAdapter:
    """
    USB serial communication layer for THERMO-X.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        timeout_s: float = 1.0,
    ) -> None:

        self.port = port

        self.baud_rate = baud_rate

        self.timeout_s = timeout_s

        self.connection: serial.Serial | None = None

    def connect(self) -> None:
        """Open the serial connection."""

        if self.connection is not None:
            return

        self.connection = serial.Serial(
            port=self.port,
            baudrate=self.baud_rate,
            timeout=self.timeout_s,
        )

    def close(self) -> None:
        """Close the serial connection."""

        if self.connection is not None:

            self.connection.close()

            self.connection = None

    def _require_connection(self) -> serial.Serial:
        """Return the active serial connection."""

        if self.connection is None:
            raise RuntimeError(
                "Serial connection is not open."
            )

        return self.connection

    @staticmethod
    def _snapshot_from_dict(
        payload: dict,
    ) -> SensorSnapshot:
        """
        Convert a JSON dictionary into SensorSnapshot.

        Only known SensorSnapshot fields are accepted.
        """

        valid_fields = {
            field.name
            for field
            in fields(SensorSnapshot)
        }

        clean_payload = {
            key: value
            for key, value
            in payload.items()
            if key in valid_fields
        }

        return SensorSnapshot(
            **clean_payload
        )

    def read(
        self,
        timestamp_s: float,
    ) -> Sequence[SensorSnapshot]:
        """
        Read one sensor batch from the ESP32.

        The supplied timestamp is used only as a fallback if
        the incoming data doesn't contain a timestamp.
        """

        connection = (
            self._require_connection()
        )

        raw_line = connection.readline()

        if not raw_line:
            return []

        try:
            message = json.loads(
                raw_line.decode(
                    "utf-8"
                ).strip()
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Received invalid JSON from ESP32."
            ) from exc

        if message.get("type") != "sensor_batch":
            return []

        sensor_payloads = (
            message.get(
                "sensors",
                [],
            )
        )

        snapshots: list[
            SensorSnapshot
        ] = []

        for payload in sensor_payloads:

            if "timestamp_s" not in payload:
                payload = {
                    **payload,
                    "timestamp_s": timestamp_s,
                }

            snapshots.append(
                self._snapshot_from_dict(
                    payload
                )
            )

        return snapshots

    def apply(
        self,
        command: ActuatorCommand,
    ) -> None:
        """
        Send an actuator command to the ESP32.
        """

        connection = (
            self._require_connection()
        )

        # Enum values must be converted to strings
        # before JSON serialization.
        payload = {
            "type": "actuator_command",

            "pump_speed_pct":
                command.pump_speed_pct,

            "fan_speed_pct":
                command.fan_speed_pct,

            "valve_positions_pct":
                command.valve_positions_pct
                or {},

            "heater_enabled":
                command.heater_enabled,

            "cooling_mode":
                command.cooling_mode.value,

            "system_state":
                command.system_state.value,

            "reason":
                command.reason,
        }

        encoded = (
            json.dumps(
                payload
            )
            + "\n"
        ).encode("utf-8")

        connection.write(
            encoded
        )