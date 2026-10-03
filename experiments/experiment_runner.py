"""
THERMO-X fixed-vs-adaptive experiment runner.

This script performs two controlled simulations:

    1. FIXED COOLING
    2. THERMO-X ADAPTIVE COOLING

Both experiments use the same workload profile and thermal
model parameters.

No hardware is required.

The resulting datasets are saved as CSV files.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import pandas as pd

from analytics.comparison import (
    compare_runs,
)
from analytics.plots import (
    plot_fixed_vs_adaptive_temperature,
    plot_temperature_vs_time,
)
from analytics.metrics import (
    calculate_run_metrics,
)

from core.controller import (
    AdaptiveController,
)
from core.models import (
    ActuatorCommand,
    CoolingMode,
)
from core.safety import (
    SafetyController,
    apply_safety_override,
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

from .experiment_config import (
    ExperimentConfig,
)


def write_csv_header(
    writer: csv.writer,
) -> None:
    """Write the common THERMO-X experiment columns."""

    writer.writerow(
        [
            "timestamp_s",
            "server_id",
            "workload_pct",
            "heater_power_w",
            "temperature_c",
            "coolant_inlet_c",
            "coolant_outlet_c",
            "flow_rate_lpm",
            "pressure_kpa",
            "ambient_temperature_c",
            "pump_speed_pct",
            "fan_speed_pct",
            "valve_position_pct",
            "heater_enabled",
            "cooling_mode",
            "system_state",
            "system_reason",
        ]
    )


def write_snapshot_rows(
    writer: csv.writer,
    snapshots,
    command: ActuatorCommand,
) -> None:
    """Write one CSV row for every thermal zone."""

    for snapshot in snapshots:

        writer.writerow(
            [
                f"{snapshot.timestamp_s:.2f}",

                snapshot.server_id,

                f"{snapshot.workload_pct:.2f}",

                f"{snapshot.heater_power_w:.2f}",

                f"{snapshot.temperature_c:.4f}",

                f"{snapshot.coolant_inlet_c:.4f}",

                f"{snapshot.coolant_outlet_c:.4f}",

                f"{snapshot.flow_rate_lpm:.4f}",

                f"{snapshot.pressure_kpa:.4f}",

                f"{snapshot.ambient_temperature_c:.4f}",

                f"{command.pump_speed_pct:.2f}",

                f"{command.fan_speed_pct:.2f}",

                f"{(command.valve_positions_pct or {}).get(snapshot.server_id, 0.0):.2f}",

                command.heater_enabled,

                command.cooling_mode.value,

                command.system_state.value,

                command.reason,
            ]
        )


def make_fixed_command(
    config: ExperimentConfig,
) -> ActuatorCommand:
    """
    Create the fixed-cooling baseline command.

    The same command is used throughout the run unless the
    safety controller needs to override it.
    """

    return ActuatorCommand(
        pump_speed_pct=(
            config.fixed_pump_speed_pct
        ),

        fan_speed_pct=(
            config.fixed_fan_speed_pct
        ),

        valve_positions_pct={
            server_id:
                config.fixed_valve_position_pct
            for server_id
            in config.server_ids
        },

        heater_enabled=True,

        cooling_mode=CoolingMode.FIXED,

        reason=(
            "Fixed cooling baseline."
        ),
    )


def run_single_mode(
    config: ExperimentConfig,
    mode: str,
    output_path: Path,
) -> None:
    """
    Execute one controlled experiment.

    mode must be either:

        "fixed"
        "adaptive"
    """

    if mode not in {
        "fixed",
        "adaptive",
    }:
        raise ValueError(
            "mode must be 'fixed' or 'adaptive'."
        )

    # -----------------------------------------
    # Create a fresh thermal plant.
    # -----------------------------------------

    model = ThermalSystemModel(
        server_ids=list(
            config.server_ids
        )
    )

    # -----------------------------------------
    # Same workload profile for both tests.
    # -----------------------------------------

    workload_profile = (
        create_demo_profile(
            list(
                config.server_ids
            )
        )
    )

    # -----------------------------------------
    # Same sensor noise seed for both tests.
    #
    # This improves reproducibility and makes the
    # comparison more controlled.
    # -----------------------------------------

    sensors = VirtualSensorBank(
        model=model,
        seed=config.sensor_seed,
    )

    actuators = (
        VirtualActuatorBank()
    )

    safety = SafetyController()

    controller = (
        AdaptiveController(
            safety_controller=safety
        )
        if mode == "adaptive"
        else None
    )

    # -----------------------------------------
    # Initial actuator command.
    # -----------------------------------------

    if mode == "fixed":

        current_command = (
            make_fixed_command(
                config
            )
        )

    else:

        current_command = (
            ActuatorCommand(
                pump_speed_pct=30.0,
                fan_speed_pct=30.0,
                valve_positions_pct={
                    server_id: 50.0
                    for server_id
                    in config.server_ids
                },
                heater_enabled=True,
                cooling_mode=(
                    CoolingMode.ADAPTIVE
                ),
                reason=(
                    "Initial adaptive "
                    "cooling command."
                ),
            )
        )

    actuators.apply(
        current_command
    )

    # -----------------------------------------
    # Prepare output.
    # -----------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        write_csv_header(
            writer
        )

        timestamp_s = 0.0

        while timestamp_s < config.duration_s:

            # ---------------------------------
            # Determine workload.
            # ---------------------------------

            workload = {
                server_id:
                    workload_profile.get_workload(
                        timestamp_s,
                        server_id,
                    )
                for server_id
                in config.server_ids
            }

            heater_powers = {
                server_id:
                    values[1]
                for server_id, values
                in workload.items()
            }

            # ---------------------------------
            # Apply current cooling command
            # to the thermal plant.
            # ---------------------------------

            model.step(
                heater_powers_w=(
                    heater_powers
                ),

                command=(
                    actuators.command
                ),

                dt_s=config.dt_s,
            )

            # ---------------------------------
            # Read simulated sensors.
            # ---------------------------------

            snapshots = sensors.read(
                timestamp_s=timestamp_s,
                workload_by_server=workload,
            )

            # ---------------------------------
            # Determine the NEXT actuator command.
            # ---------------------------------

            if mode == "adaptive":

                next_command, _ = (
                    controller.control_step(
                        snapshots
                    )
                )

            else:

                # Create a new command object so
                # a safety override doesn't alter
                # the baseline template.
                next_command = copy.deepcopy(
                    current_command
                )

                safety_decision = (
                    safety.evaluate(
                        snapshots
                    )
                )

                next_command = (
                    apply_safety_override(
                        next_command,
                        safety_decision,
                    )
                )

            # ---------------------------------
            # Record the state that produced
            # this sensor sample.
            # ---------------------------------

            write_snapshot_rows(
                writer=writer,
                snapshots=list(
                    snapshots
                ),
                command=(
                    actuators.command
                ),
            )

            # ---------------------------------
            # Apply next command.
            # ---------------------------------

            actuators.apply(
                next_command
            )

            current_command = (
                next_command
            )

            # ---------------------------------
            # Periodic console output.
            # ---------------------------------

            if abs(
                timestamp_s % 20.0
            ) < 1e-9:

                average_temperature = (
                    sum(
                        snapshot.temperature_c
                        for snapshot
                        in snapshots
                    )
                    / len(snapshots)
                )

                print(
                    f"[{mode.upper():8}] "
                    f"t={timestamp_s:6.1f}s | "
                    f"avg_temp="
                    f"{average_temperature:6.2f} C | "
                    f"pump="
                    f"{actuators.command.pump_speed_pct:6.2f}% | "
                    f"fan="
                    f"{actuators.command.fan_speed_pct:6.2f}% | "
                    f"state="
                    f"{actuators.command.system_state.value}"
                )

            timestamp_s += (
                config.dt_s
            )


def run_experiment(
    config: ExperimentConfig,
) -> None:
    """Run fixed and adaptive experiments and compare them."""

    config.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=========================================="
    )
    print(
        "THERMO-X FIXED vs ADAPTIVE EXPERIMENT"
    )
    print(
        "=========================================="
    )
    print()

    # -----------------------------------------
    # Experiment A: Fixed cooling
    # -----------------------------------------

    print(
        "Running FIXED cooling experiment..."
    )

    run_single_mode(
        config=config,
        mode="fixed",
        output_path=(
            config.fixed_csv_path
        ),
    )

    # -----------------------------------------
    # Experiment B: Adaptive cooling
    # -----------------------------------------

    print()
    print(
        "Running THERMO-X ADAPTIVE experiment..."
    )

    run_single_mode(
        config=config,
        mode="adaptive",
        output_path=(
            config.adaptive_csv_path
        ),
    )

    # -----------------------------------------
    # Load both datasets.
    # -----------------------------------------

    fixed_dataframe = (
        pd.read_csv(
            config.fixed_csv_path
        )
    )

    adaptive_dataframe = (
        pd.read_csv(
            config.adaptive_csv_path
        )
    )

    # -----------------------------------------
    # Calculate comparison.
    # -----------------------------------------

    comparison = compare_runs(
        fixed_dataframe=fixed_dataframe,
        adaptive_dataframe=adaptive_dataframe,
    )

    comparison.to_csv(
        config.comparison_csv_path,
        index=False,
    )

    # -----------------------------------------
    # Generate useful plots.
    # -----------------------------------------

    plot_temperature_vs_time(
        fixed_dataframe,
        config.plot_directory,
        filename="fixed_temperature_vs_time.png",
    )

    plot_temperature_vs_time(
        adaptive_dataframe,
        config.plot_directory,
        filename="adaptive_temperature_vs_time.png",
    )

    plot_fixed_vs_adaptive_temperature(
        fixed_dataframe,
        adaptive_dataframe,
        config.plot_directory,
    )

    # -----------------------------------------
    # Print summary metrics.
    # -----------------------------------------

    fixed_metrics = (
        calculate_run_metrics(
            fixed_dataframe
        ).iloc[0]
    )

    adaptive_metrics = (
        calculate_run_metrics(
            adaptive_dataframe
        ).iloc[0]
    )

    print()
    print(
        "=========================================="
    )
    print(
        "EXPERIMENT COMPLETE"
    )
    print(
        "=========================================="
    )

    print()
    print(
        "Fixed cooling metrics:"
    )

    print(
        f"  Maximum temperature : "
        f"{fixed_metrics['maximum_temperature_c']:.3f} °C"
    )

    print(
        f"  Average temperature : "
        f"{fixed_metrics['average_temperature_c']:.3f} °C"
    )

    print(
        f"  Cooling energy      : "
        f"{fixed_metrics['cooling_energy_wh']:.3f} Wh"
    )

    print()
    print(
        "Adaptive cooling metrics:"
    )

    print(
        f"  Maximum temperature : "
        f"{adaptive_metrics['maximum_temperature_c']:.3f} °C"
    )

    print(
        f"  Average temperature : "
        f"{adaptive_metrics['average_temperature_c']:.3f} °C"
    )

    print(
        f"  Cooling energy      : "
        f"{adaptive_metrics['cooling_energy_wh']:.3f} Wh"
    )

    print()
    print(
        "Output files:"
    )

    print(
        f"  Fixed CSV      : "
        f"{config.fixed_csv_path}"
    )

    print(
        f"  Adaptive CSV   : "
        f"{config.adaptive_csv_path}"
    )

    print(
        f"  Comparison CSV : "
        f"{config.comparison_csv_path}"
    )

    print(
        f"  Plots          : "
        f"{config.plot_directory}"
    )


def main() -> None:
    """Command-line entry point."""

    config = (
        ExperimentConfig()
    )

    run_experiment(
        config
    )


if __name__ == "__main__":
    main()