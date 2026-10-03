"""
THERMO-X application service layer.

This file provides reusable functions for:

    - building the virtual THERMO-X system
    - running a simulation
    - converting simulation output into a DataFrame
    - loading saved experiment data

The dashboard imports these functions instead of directly
constructing the thermal model itself.

This separation will become important when the simulator is
eventually replaced by the real ESP32 adapter.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd

from core.controller import AdaptiveController
from core.models import SensorSnapshot

from experiments.scenario_generator import (
    PlantVariation,
    ScenarioSpec,
    simulate_scenario,
)

from simulator.virtual_actuators import (
    VirtualActuatorBank,
)

from simulator.virtual_sensors import (
    VirtualSensorBank,
)

from simulator.workload_generator import (
    create_demo_profile,
)

from thermal_model.model import (
    ThermalSystemModel,
)

from simulator.thermal_system import (
    VirtualThermoXSystem,
)

from .config import (
    DEFAULT_SENSOR_SEED,
    SERVER_IDS,
)

def run_interactive_scenario(
    pattern: str,
    duration_s: float,
    ambient_temperature_c: float,
    imbalance_pct: float,
    peak_intensity_pct: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run one judge-friendly interactive thermal scenario.

    This is a software simulation, not a physical hardware run.
    """

    plant = PlantVariation(
        thermal_mass_j_per_k=850.0,

        cold_plate_ua_w_per_k=4.5,

        radiator_ua_w_per_k=65.0,

        maximum_flow_lpm=2.8,
    )

    spec = ScenarioSpec(

        scenario_id=(
            "INTERACTIVE_SCENARIO"
        ),

        pattern=pattern,

        duration_s=duration_s,

        ambient_temperature_c=(
            ambient_temperature_c
        ),

        imbalance_pct=(
            imbalance_pct
        ),

        peak_intensity_pct=(
            peak_intensity_pct
        ),

        seed=seed,

        max_power_server_a_w=100.0,

        max_power_server_b_w=100.0,

        plant=plant,
    )

    return simulate_scenario(
        spec
    )

def build_virtual_system(
    sensor_seed: int = DEFAULT_SENSOR_SEED,
) -> VirtualThermoXSystem:
    """
    Build a fresh virtual THERMO-X system.

    The same function is used every time a new simulation
    is started, ensuring each run begins from a known state.
    """

    server_ids = list(
        SERVER_IDS
    )

    # Workload generator.
    workload_profile = (
        create_demo_profile(
            server_ids
        )
    )

    # Thermal plant.
    thermal_model = (
        ThermalSystemModel(
            server_ids=server_ids
        )
    )

    # Adaptive controller.
    controller = (
        AdaptiveController()
    )

    # Virtual sensors.
    sensors = (
        VirtualSensorBank(
            model=thermal_model,
            seed=sensor_seed,
        )
    )

    # Virtual actuators.
    actuators = (
        VirtualActuatorBank()
    )

    return VirtualThermoXSystem(
        server_ids=server_ids,
        workload_profile=workload_profile,
        model=thermal_model,
        controller=controller,
        sensors=sensors,
        actuators=actuators,
    )


def snapshot_to_record(
    snapshot: SensorSnapshot,
    pump_speed_pct: float,
    fan_speed_pct: float,
    valve_position_pct: float,
    heater_enabled: bool,
    cooling_mode: str,
    system_state: str,
    system_reason: str,
) -> dict:
    """
    Convert one SensorSnapshot into a dictionary suitable
    for a pandas DataFrame.
    """

    return {
        "timestamp_s": snapshot.timestamp_s,

        "server_id": snapshot.server_id,

        "workload_pct": snapshot.workload_pct,

        "heater_power_w": snapshot.heater_power_w,

        "temperature_c": snapshot.temperature_c,

        "coolant_inlet_c": (
            snapshot.coolant_inlet_c
        ),

        "coolant_outlet_c": (
            snapshot.coolant_outlet_c
        ),

        "flow_rate_lpm": (
            snapshot.flow_rate_lpm
        ),

        "pressure_kpa": (
            snapshot.pressure_kpa
        ),

        "ambient_temperature_c": (
            snapshot.ambient_temperature_c
        ),

        "delta_t_c": (
            snapshot.coolant_delta_t_c
        ),

        "pump_speed_pct": pump_speed_pct,

        "fan_speed_pct": fan_speed_pct,

        "valve_position_pct": (
            valve_position_pct
        ),

        "heater_enabled": heater_enabled,

        "cooling_mode": cooling_mode,

        "system_state": system_state,

        "system_reason": system_reason,
    }


def run_live_simulation(
    duration_s: float,
    dt_s: float = 1.0,
    sensor_seed: int = DEFAULT_SENSOR_SEED,
    realtime_speed: float = 20.0,
    update_callback: Callable[
        [pd.DataFrame],
        None,
    ]
    | None = None,
) -> pd.DataFrame:
    """
    Run an adaptive THERMO-X simulation.

    Parameters
    ----------
    duration_s:
        Total simulated time.

    dt_s:
        Simulation timestep.

    sensor_seed:
        Seed used by the virtual sensors.

    realtime_speed:
        Controls how quickly the simulation is displayed.

        1.0  = approximately real time
        20.0 = approximately 20x faster

    update_callback:
        Optional callback used by the Streamlit dashboard
        to display intermediate results.

    Returns
    -------
    pandas.DataFrame
        Complete simulation dataset.
    """

    if duration_s <= 0:
        raise ValueError(
            "Simulation duration must be positive."
        )

    if dt_s <= 0:
        raise ValueError(
            "Simulation timestep must be positive."
        )

    if realtime_speed <= 0:
        raise ValueError(
            "Realtime speed must be positive."
        )

    system = build_virtual_system(
        sensor_seed=sensor_seed
    )

    records: list[dict] = []

    timestamp_s = 0.0

    while timestamp_s < duration_s:

        # Save the actuator command that is currently
        # controlling this simulation interval.
        #
        # This is the command already applied to the
        # virtual plant.
        applied_command = (
            system.actuators.command
        )

        snapshots, next_command = (
            system.step(
                timestamp_s=timestamp_s,
                dt_s=dt_s,
            )
        )

        # The snapshots describe all virtual thermal zones.
        for snapshot in snapshots:

            valve_position = (
                (
                    applied_command
                    .valve_positions_pct
                    or {}
                )
                .get(
                    snapshot.server_id,
                    0.0,
                )
            )

            records.append(
                snapshot_to_record(
                    snapshot=snapshot,

                    pump_speed_pct=(
                        applied_command
                        .pump_speed_pct
                    ),

                    fan_speed_pct=(
                        applied_command
                        .fan_speed_pct
                    ),

                    valve_position_pct=(
                        valve_position
                    ),

                    heater_enabled=(
                        applied_command
                        .heater_enabled
                    ),

                    cooling_mode=(
                        applied_command
                        .cooling_mode.value
                    ),

                    system_state=(
                        applied_command
                        .system_state.value
                    ),

                    system_reason=(
                        applied_command
                        .reason
                    ),
                )
            )

        # Create a DataFrame containing everything generated
        # so far. This is used only when the dashboard asks
        # for intermediate display updates.
        # ----------------------------------------------------------
        # Dashboard update throttling
        # ----------------------------------------------------------
        #
        # The thermal simulation still runs every dt_s.
        # We simply do not redraw the Streamlit UI for every single
        # sample. This makes longer simulations much smoother.
        # ----------------------------------------------------------

        should_update_dashboard = (
            update_callback is not None
            and (
                len(records) == 0
                or int(timestamp_s) % 5 == 0
                or timestamp_s + dt_s >= duration_s
            )
        )

        if should_update_dashboard:

            current_dataframe = (
                pd.DataFrame(
                    records
                )
            )

            update_callback(
                current_dataframe
            )

        # Convert simulated seconds into actual waiting time.
        #
        # Example:
        # 20x speed + 1 second simulation timestep
        # = approximately 0.05 seconds of real waiting.
        time.sleep(
            dt_s / realtime_speed
        )

        timestamp_s += dt_s

    return pd.DataFrame(
        records
    )


def load_experiment_csv(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load a THERMO-X experiment CSV.

    Raises a clear error if the file doesn't exist.
    """

    csv_path = Path(
        path
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Experiment file not found: {csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path
    )

    if dataframe.empty:
        raise ValueError(
            f"Experiment file is empty: {csv_path}"
        )

    return dataframe