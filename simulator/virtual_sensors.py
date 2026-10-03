"""
Virtual sensors for the THERMO-X software prototype.

These generate noisy measurements from the thermal model
so that the controller experiences conditions closer to
real sensor data rather than perfect mathematical values.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from core.models import SensorSnapshot
from thermal_model.model import ThermalSystemModel


class VirtualSensorConfig:
    """Noise parameters for simulated sensors."""

    # Small temperature measurement noise.
    temperature_noise_c: float = 0.15

    # Simulated flow-meter noise.
    flow_noise_lpm: float = 0.02

    # Simulated pressure-sensor noise.
    pressure_noise_kpa: float = 1.0


class VirtualSensorBank:
    """
    Generates SensorSnapshot objects from the thermal model.
    """

    def __init__(
        self,
        model: ThermalSystemModel,
        config: VirtualSensorConfig | None = None,
        seed: int = 42,
    ) -> None:

        self.model = model

        self.config = (
            config
            or VirtualSensorConfig()
        )

        # Fixed seed makes the simulation reproducible.
        self._random = random.Random(seed)

    def _noise(
        self,
        amplitude: float,
    ) -> float:
        """Generate bounded random sensor noise."""

        return self._random.uniform(
            -amplitude,
            amplitude,
        )

    def read(
        self,
        timestamp_s: float,
        workload_by_server: Mapping[
            str,
            tuple[float, float],
        ],
    ) -> Sequence[SensorSnapshot]:
        """
        Convert internal model state into virtual sensor readings.
        """

        snapshots: list[
            SensorSnapshot
        ] = []

        for (
            server_id,
            state,
        ) in self.model.modules.items():

            workload_pct, heater_power_w = (
                workload_by_server[
                    server_id
                ]
            )

            # A simplified pressure relationship
            # for the software model.
            pressure_kpa = (
                80.0
                + 20.0 * state.branch_flow_lpm
            )

            snapshot = SensorSnapshot(
                timestamp_s=timestamp_s,

                server_id=server_id,

                workload_pct=workload_pct,

                heater_power_w=heater_power_w,

                temperature_c=(
                    state.temperature_c
                    + self._noise(
                        self.config
                        .temperature_noise_c
                    )
                ),

                coolant_inlet_c=(
                    self.model.coolant_temperature_c
                ),

                coolant_outlet_c=(
                    state.coolant_outlet_c
                    + self._noise(
                        self.config
                        .temperature_noise_c
                    )
                ),

                flow_rate_lpm=max(
                    0.0,
                    state.branch_flow_lpm
                    + self._noise(
                        self.config
                        .flow_noise_lpm
                    ),
                ),

                pressure_kpa=max(
                    0.0,
                    pressure_kpa
                    + self._noise(
                        self.config
                        .pressure_noise_kpa
                    ),
                ),

                ambient_temperature_c=(
                    self.model
                    .cooling_params
                    .ambient_temperature_c
                ),

                leak_detected=False,

                sensors_valid=True,
            )

            snapshots.append(
                snapshot
            )

        return snapshots