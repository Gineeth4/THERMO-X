"""
MQTT adapter for THERMO-X.

Possible future architecture:

    ESP32
      │
      │ Wi-Fi
      ▼
    MQTT Broker
      │
      ▼
    THERMO-X Laptop

This file is scaffolding for later hardware integration.
It is not required for the current simulator.
"""

from __future__ import annotations

import json
from typing import Sequence

import paho.mqtt.client as mqtt

from core.models import (
    ActuatorCommand,
    SensorSnapshot,
)


class MQTTAdapter:
    """
    MQTT communication layer for THERMO-X.
    """

    def __init__(
        self,
        broker_host: str,
        broker_port: int = 1883,
        sensor_topic: str = "thermo-x/sensors",
        command_topic: str = "thermo-x/commands",
        client_id: str = "thermo-x-laptop",
    ) -> None:

        self.broker_host = broker_host

        self.broker_port = broker_port

        self.sensor_topic = sensor_topic

        self.command_topic = command_topic

        self.client = mqtt.Client(
            client_id=client_id
        )

        self.latest_snapshots: dict[
            str,
            SensorSnapshot,
        ] = {}

        self.client.on_message = (
            self._on_message
        )

    def connect(self) -> None:
        """Connect to the MQTT broker."""

        self.client.connect(
            self.broker_host,
            self.broker_port,
        )

        self.client.subscribe(
            self.sensor_topic
        )

        self.client.loop_start()

    def close(self) -> None:
        """Disconnect from the MQTT broker."""

        self.client.loop_stop()

        self.client.disconnect()

    def _on_message(
        self,
        client,
        userdata,
        message,
    ) -> None:
        """Receive and decode sensor messages."""

        try:
            payload = json.loads(
                message.payload.decode(
                    "utf-8"
                )
            )

        except json.JSONDecodeError:
            return

        if payload.get("type") != "sensor_batch":
            return

        for sensor_data in payload.get(
            "sensors",
            [],
        ):

            try:
                snapshot = (
                    SensorSnapshot(
                        **sensor_data
                    )
                )

            except TypeError:
                # Ignore malformed sensor records.
                continue

            self.latest_snapshots[
                snapshot.server_id
            ] = snapshot

    def read(
        self,
        timestamp_s: float = 0.0,
    ) -> Sequence[SensorSnapshot]:
        """
        Return the latest received sensor values.
        """

        return list(
            self.latest_snapshots.values()
        )

    def apply(
        self,
        command: ActuatorCommand,
    ) -> None:
        """Publish an actuator command."""

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

        self.client.publish(
            self.command_topic,
            json.dumps(payload),
        )