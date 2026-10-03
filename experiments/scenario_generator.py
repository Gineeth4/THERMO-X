"""
THERMO-X synthetic thermal-scenario generator.

Purpose
-------
Generate a large and diverse thermal dataset for the ML layer.

The scenarios represent controlled SOFTWARE EXPERIMENTS.

They are not measurements from a real data center.

Scenario variation includes:

    - ambient temperature
    - workload patterns
    - sudden thermal spikes
    - gradual ramps
    - high sustained load
    - thermal-zone imbalance
    - cooling-system parameter variation

This gives the ML models substantially more thermal diversity
than the original single 240-second workload.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.controller import (
    AdaptiveController,
)

from core.models import (
    ActuatorCommand,
)

from simulator.workload_generator import (
    WorkloadProfile,
    WorkloadSegment,
)

from simulator.virtual_sensors import (
    VirtualSensorBank,
)

from thermal_model.model import (
    ThermalSystemModel,
)

from thermal_model.parameters import (
    CoolingSystemParameters,
    ThermalModuleParameters,
)


# ============================================================
# SCENARIO DEFINITIONS
# ============================================================


@dataclass(frozen=True, slots=True)
class PlantVariation:
    """
    Simulation-only variation of thermal-system behaviour.
    """

    thermal_mass_j_per_k: float

    cold_plate_ua_w_per_k: float

    radiator_ua_w_per_k: float

    maximum_flow_lpm: float

    pump_power_at_100_pct_w: float = 30.0

    fan_power_at_100_pct_w: float = 20.0


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Description of one thermal experiment."""

    scenario_id: str

    pattern: str

    duration_s: float

    ambient_temperature_c: float

    imbalance_pct: float

    peak_intensity_pct: float

    seed: int

    max_power_server_a_w: float

    max_power_server_b_w: float

    plant: PlantVariation


# ============================================================
# RANDOM PLANT
# ============================================================


def random_plant(
    rng: random.Random,
) -> PlantVariation:
    """
    Generate plausible simulation variation.

    These are not hardware specifications.
    """

    return PlantVariation(
        thermal_mass_j_per_k=(
            rng.uniform(
                650.0,
                1000.0,
            )
        ),

        cold_plate_ua_w_per_k=(
            rng.uniform(
                3.5,
                5.5,
            )
        ),

        radiator_ua_w_per_k=(
            rng.uniform(
                52.0,
                82.0,
            )
        ),

        maximum_flow_lpm=(
            rng.uniform(
                2.2,
                3.3,
            )
        ),
    )


# ============================================================
# WORKLOAD GENERATION
# ============================================================


def _clip_workload(
    value: float,
) -> float:
    """Keep workload inside 10–100%."""

    return float(
        np.clip(
            value,
            10.0,
            100.0,
        )
    )


def _pattern_workload(
    pattern: str,
    segment_index: int,
    segment_count: int,
    rng: random.Random,
    peak: float,
) -> float:
    """Generate workload for one segment."""

    if pattern == "Spike":

        if (
            segment_index
            == segment_count // 2
        ):

            return peak

        return rng.uniform(
            20.0,
            42.0,
        )

    if pattern == "Ramp":

        phase = (
            segment_index
            / max(
                1,
                segment_count - 1,
            )
        )

        triangle = (
            1.0
            - abs(
                2.0 * phase
                - 1.0
            )
        )

        return (
            20.0
            + triangle
            * (
                peak
                - 20.0
            )
        )

    if pattern == "High Load":

        return rng.uniform(
            max(
                60.0,
                peak - 15.0,
            ),
            peak,
        )

    if pattern == "Unbalanced":

        return rng.uniform(
            20.0,
            peak,
        )

    # Mixed.

    choices = [
        rng.uniform(
            15.0,
            40.0,
        ),

        rng.uniform(
            35.0,
            65.0,
        ),

        rng.uniform(
            55.0,
            peak,
        ),
    ]

    return rng.choice(
        choices
    )


def create_workload_profile(
    spec: ScenarioSpec,
) -> WorkloadProfile:
    """
    Create independent server workload schedules.

    The two servers intentionally do not always experience
    identical workloads.
    """

    server_ids = [
        "SERVER_A",
        "SERVER_B",
    ]

    segment_count = max(
        4,
        int(
            spec.duration_s
            / 30.0
        ),
    )

    segment_duration = (
        spec.duration_s
        / segment_count
    )

    schedules = {}

    for server_index, server_id in enumerate(
        server_ids
    ):

        segments = []

        # Apply opposite bias to the two zones.
        if server_index == 0:

            bias = (
                spec.imbalance_pct
                / 2.0
            )

        else:

            bias = (
                -spec.imbalance_pct
                / 2.0
            )

        for segment_index in range(
            segment_count
        ):

            start = (
                segment_index
                * segment_duration
            )

            end = (
                spec.duration_s
                if segment_index
                == segment_count - 1
                else (
                    segment_index
                    + 1
                )
                * segment_duration
            )

            workload = (
                _pattern_workload(
                    pattern=spec.pattern,
                    segment_index=segment_index,
                    segment_count=segment_count,
                    rng=random.Random(
                        spec.seed
                        + server_index * 1000
                        + segment_index,
                    ),
                    peak=spec.peak_intensity_pct,
                )
            )

            # Apply zone imbalance.
            if spec.pattern != "Unbalanced":
                workload += (
                    bias
                )

            elif server_index == 0:
                workload += abs(
                    bias
                )

            else:
                workload -= abs(
                    bias
                )

            workload = (
                _clip_workload(
                    workload
                )
            )

            max_power = (
                spec.max_power_server_a_w
                if server_id
                == "SERVER_A"
                else spec.max_power_server_b_w
            )

            heater_power = (
                workload
                / 100.0
                * max_power
            )

            segments.append(
                WorkloadSegment(
                    start_s=start,
                    end_s=end,
                    workload_pct=workload,
                    heater_power_w=heater_power,
                )
            )

        schedules[
            server_id
        ] = segments

    return WorkloadProfile(
        schedules
    )


# ============================================================
# SIMULATION
# ============================================================


def simulate_scenario(
    spec: ScenarioSpec,
    dt_s: float = 1.0,
    sensor_noise_seed: int | None = None,
) -> pd.DataFrame:
    """
    Run one complete software thermal experiment.
    """

    if sensor_noise_seed is None:

        sensor_noise_seed = (
            spec.seed
        )

    server_ids = [
        "SERVER_A",
        "SERVER_B",
    ]

    module_parameters = (
        ThermalModuleParameters(

            thermal_mass_j_per_k=(
                spec.plant
                .thermal_mass_j_per_k
            ),

            cold_plate_ua_w_per_k=(
                spec.plant
                .cold_plate_ua_w_per_k
            ),

            max_heater_power_w=110.0,

            initial_temperature_c=(
                spec.ambient_temperature_c
                + 4.0
            ),
        )
    )

    cooling_parameters = (
        CoolingSystemParameters(

            ambient_temperature_c=(
                spec.ambient_temperature_c
            ),

            minimum_total_flow_lpm=0.15,

            maximum_total_flow_lpm=(
                spec.plant
                .maximum_flow_lpm
            ),

            radiator_ua_w_per_k=(
                spec.plant
                .radiator_ua_w_per_k
            ),

            coolant_heat_capacity_j_per_k=(
                3500.0
            ),

            initial_coolant_temperature_c=(
                spec.ambient_temperature_c
            ),

            pump_power_at_100_pct_w=(
                spec.plant
                .pump_power_at_100_pct_w
            ),

            fan_power_at_100_pct_w=(
                spec.plant
                .fan_power_at_100_pct_w
            ),
        )
    )

    model = ThermalSystemModel(
        server_ids=server_ids,
        module_params=module_parameters,
        cooling_params=cooling_parameters,
    )

    workload_profile = (
        create_workload_profile(
            spec
        )
    )

    sensors = VirtualSensorBank(
        model=model,
        seed=sensor_noise_seed,
    )

    controller = AdaptiveController()

    current_command = ActuatorCommand(
        pump_speed_pct=30.0,

        fan_speed_pct=30.0,

        valve_positions_pct={
            server_id: 50.0
            for server_id
            in server_ids
        },

        heater_enabled=True,
    )

    records = []

    timestamp_s = 0.0

    while timestamp_s < spec.duration_s:

        workload = {
            server_id:
                workload_profile.get_workload(
                    timestamp_s,
                    server_id,
                )
            for server_id
            in server_ids
        }

        heater_powers = {
            server_id:
                values[1]
            for (
                server_id,
                values,
            ) in workload.items()
        }

        # Current command controls this interval.
        model.step(
            heater_powers_w=heater_powers,
            command=current_command,
            dt_s=dt_s,
        )

        snapshots = sensors.read(
            timestamp_s=timestamp_s,
            workload_by_server=workload,
        )

        next_command, metrics = (
            controller.control_step(
                snapshots
            )
        )

        metric_by_server = {
            snapshot.server_id:
                metric
            for snapshot, metric
            in zip(
                snapshots,
                metrics,
            )
        }

        for snapshot in snapshots:

            thermal_metric = (
                metric_by_server[
                    snapshot.server_id
                ]
            )

            valve_position = (
                (
                    current_command
                    .valve_positions_pct
                    or {}
                )
                .get(
                    snapshot.server_id,
                    0.0,
                )
            )

            records.append(
                {
                    "scenario_id":
                        spec.scenario_id,

                    "scenario_type":
                        spec.pattern,

                    "timestamp_s":
                        snapshot.timestamp_s,

                    "server_id":
                        snapshot.server_id,

                    "workload_pct":
                        snapshot.workload_pct,

                    "heater_power_w":
                        snapshot.heater_power_w,

                    "temperature_c":
                        snapshot.temperature_c,

                    "coolant_inlet_c":
                        snapshot.coolant_inlet_c,

                    "coolant_outlet_c":
                        snapshot.coolant_outlet_c,

                    "flow_rate_lpm":
                        snapshot.flow_rate_lpm,

                    "pressure_kpa":
                        snapshot.pressure_kpa,

                    "ambient_temperature_c":
                        snapshot.ambient_temperature_c,

                    "pump_speed_pct":
                        current_command
                        .pump_speed_pct,

                    "fan_speed_pct":
                        current_command
                        .fan_speed_pct,

                    "valve_position_pct":
                        valve_position,

                    "thermal_demand_index":
                        thermal_metric
                        .thermal_demand_index,

                    "heater_enabled":
                        current_command
                        .heater_enabled,

                    "cooling_mode":
                        current_command
                        .cooling_mode.value,

                    "system_state":
                        current_command
                        .system_state.value,

                    "system_reason":
                        current_command
                        .reason,
                }
            )

        current_command = (
            next_command
        )

        timestamp_s += dt_s

    return pd.DataFrame(
        records
    )


# ============================================================
# RANDOM TRAINING SCENARIO
# ============================================================


def create_random_training_spec(
    index: int,
    master_seed: int,
    duration_s: float,
) -> ScenarioSpec:
    """Create a randomized training experiment."""

    rng = random.Random(
        master_seed
        + index
    )

    pattern = rng.choice(
        [
            "Mixed",
            "Spike",
            "Ramp",
            "High Load",
            "Unbalanced",
        ]
    )

    ambient = rng.uniform(
        24.0,
        38.0,
    )

    imbalance = rng.uniform(
        0.0,
        60.0,
    )

    peak = rng.uniform(
        65.0,
        100.0,
    )

    return ScenarioSpec(

        scenario_id=(
            f"SCENARIO_{index + 1:03d}"
        ),

        pattern=pattern,

        duration_s=duration_s,

        ambient_temperature_c=ambient,

        imbalance_pct=imbalance,

        peak_intensity_pct=peak,

        seed=(
            master_seed
            + index
            * 7919
        ),

        max_power_server_a_w=(
            rng.uniform(
                75.0,
                110.0,
            )
        ),

        max_power_server_b_w=(
            rng.uniform(
                75.0,
                110.0,
            )
        ),

        plant=random_plant(
            rng
        ),
    )


# ============================================================
# TRAINING DATASET
# ============================================================


def generate_training_dataset(
    scenario_count: int = 80,
    duration_s: float = 240.0,
    master_seed: int = 42,
    output_path: str | Path = (
        "data/processed/"
        "thermo_x_training_dataset.csv"
    ),
) -> Path:
    """Generate and save a large ML dataset."""

    if scenario_count < 10:

        raise ValueError(
            "Use at least 10 scenarios."
        )

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = []

    print()

    print(
        "THERMO-X SCENARIO GENERATION"
    )

    print(
        "============================"
    )

    for index in range(
        scenario_count
    ):

        spec = (
            create_random_training_spec(
                index=index,
                master_seed=master_seed,
                duration_s=duration_s,
            )
        )

        frame = simulate_scenario(
            spec
        )

        frames.append(
            frame
        )

        if (
            (index + 1) % 5 == 0
            or index
            == scenario_count - 1
        ):

            print(
                f"Generated "
                f"{index + 1}/"
                f"{scenario_count} scenarios"
            )

    dataset = pd.concat(
        frames,
        ignore_index=True,
    )

    dataset.to_csv(
        output,
        index=False,
    )

    print()

    print(
        f"Rows generated: "
        f"{len(dataset):,}"
    )

    print(
        f"Scenarios: "
        f"{dataset['scenario_id'].nunique()}"
    )

    print(
        f"Saved to: "
        f"{output}"
    )

    return output


# ============================================================
# CLI
# ============================================================


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the THERMO-X ML training dataset."
        )
    )

    parser.add_argument(
        "--scenarios",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=240.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/"
            "thermo_x_training_dataset.csv"
        ),
    )

    args = parser.parse_args()

    generate_training_dataset(
        scenario_count=args.scenarios,
        duration_s=args.duration,
        master_seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()