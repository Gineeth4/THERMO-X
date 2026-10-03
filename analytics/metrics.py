"""
Engineering metrics for THERMO-X experimental data.

The functions in this file operate on the CSV datasets produced
by the simulator and, later, by the physical prototype.

Important distinction:

The current cooling-power calculation is a SIMULATION ESTIMATE
based on configured actuator ratings.

When physical hardware exists, measured electrical power should
replace this estimate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp_s",
    "server_id",
    "workload_pct",
    "heater_power_w",
    "temperature_c",
    "coolant_inlet_c",
    "coolant_outlet_c",
    "flow_rate_lpm",
    "pump_speed_pct",
    "fan_speed_pct",
}


def validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Verify that the dataset contains the fields required
    for THERMO-X analysis.
    """

    missing = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )


def _integrate_power_wh(
    timestamp_s: np.ndarray,
    power_w: np.ndarray,
) -> float:
    """
    Numerically integrate power over time.

    A simple left-rectangle integration is used:

        Energy = Σ(P_i × Δt_i)

    Result is returned in watt-hours.
    """

    if len(timestamp_s) == 0:
        return 0.0

    if len(timestamp_s) != len(power_w):
        raise ValueError(
            "Time and power arrays must have equal length."
        )

    if len(timestamp_s) == 1:
        return 0.0

    dt = np.diff(
        timestamp_s,
        prepend=timestamp_s[0],
    )

    # Prevent accidental negative intervals from corrupting energy.
    dt = np.maximum(
        dt,
        0.0,
    )

    energy_wh = np.sum(
        power_w * dt
    ) / 3600.0

    return float(
        energy_wh
    )


def _add_derived_columns(
    dataframe: pd.DataFrame,
    pump_rated_power_w: float,
    fan_rated_power_w: float,
) -> pd.DataFrame:
    """
    Add quantities derived from the recorded measurements.
    """

    validate_dataframe(
        dataframe
    )

    df = dataframe.copy()

    # Coolant temperature rise.
    df["delta_t_c"] = (
        df["coolant_outlet_c"]
        - df["coolant_inlet_c"]
    )

    # Heat transfer estimate.
    #
    # Current simulation assumes water-like properties.
    # These values must be replaced with actual fluid properties
    # when another coolant is selected for the physical prototype.
    density_kg_m3 = 997.0
    cp_j_kg_k = 4180.0

    # Convert L/min → kg/s.
    mass_flow_kg_s = (
        df["flow_rate_lpm"]
        / 1000.0
        / 60.0
        * density_kg_m3
    )

    df["heat_removed_w"] = (
        mass_flow_kg_s
        * cp_j_kg_k
        * df["delta_t_c"]
    ).clip(
        lower=0.0
    )

    # -----------------------------------------
    # Simulation electrical-power estimates
    # -----------------------------------------

    df["pump_power_est_w"] = (
        df["pump_speed_pct"]
        / 100.0
        * pump_rated_power_w
    )

    df["fan_power_est_w"] = (
        df["fan_speed_pct"]
        / 100.0
        * fan_rated_power_w
    )

    df["cooling_power_est_w"] = (
        df["pump_power_est_w"]
        + df["fan_power_est_w"]
    )

    return df


def calculate_thermal_response_time(
    dataframe: pd.DataFrame,
    workload_step_threshold_pct: float = 10.0,
) -> float:
    """
    Estimate thermal response time.

    Definition used here:

        A workload increase of at least the specified threshold
        starts a thermal event.

        Response time =
        time from workload increase to the local temperature peak
        before the next workload change.

    This is an engineering definition for the current experiment.
    It can later be refined using measured prototype behavior.

    Returns the average response time in seconds.
    """

    validate_dataframe(
        dataframe
    )

    response_times: list[
        float
    ] = []

    for _, group in dataframe.groupby(
        "server_id"
    ):

        group = (
            group.sort_values(
                "timestamp_s"
            )
            .reset_index(drop=True)
        )

        workload_change = (
            group["workload_pct"]
            .diff()
            .fillna(0.0)
            .to_numpy()
        )

        change_positions = np.where(
            workload_change
            >= workload_step_threshold_pct
        )[0]

        for position in change_positions:

            start_time = float(
                group.loc[
                    position,
                    "timestamp_s",
                ]
            )

            # Find next workload change.
            next_changes = (
                change_positions[
                    change_positions
                    > position
                ]
            )

            if len(next_changes):
                end_position = int(
                    next_changes[0]
                )
            else:
                end_position = (
                    len(group) - 1
                )

            segment = group.iloc[
                position:
                end_position + 1
            ]

            if segment.empty:
                continue

            peak_position = (
                segment[
                    "temperature_c"
                ].idxmax()
            )

            peak_time = float(
                group.loc[
                    peak_position,
                    "timestamp_s",
                ]
            )

            response = (
                peak_time
                - start_time
            )

            if response >= 0:
                response_times.append(
                    response
                )

    if not response_times:
        return float("nan")

    return float(
        np.mean(
            response_times
        )
    )


def calculate_run_metrics(
    dataframe: pd.DataFrame,
    pump_rated_power_w: float = 30.0,
    fan_rated_power_w: float = 20.0,
) -> pd.DataFrame:
    """
    Calculate one summary row for an entire THERMO-X run.

    The defaults match the initial software simulation parameters.

    They are NOT physical hardware specifications.
    """

    validate_dataframe(
        dataframe
    )

    if dataframe.empty:
        raise ValueError(
            "Cannot calculate metrics from an empty dataset."
        )

    df = _add_derived_columns(
        dataframe=dataframe,
        pump_rated_power_w=pump_rated_power_w,
        fan_rated_power_w=fan_rated_power_w,
    )

    # -----------------------------------------
    # Temperature metrics
    # -----------------------------------------

    maximum_temperature = float(
        df["temperature_c"].max()
    )

    average_temperature = float(
        df["temperature_c"].mean()
    )

    temperature_std = float(
        df["temperature_c"].std(
            ddof=0
        )
    )

    temperature_range = float(
        df["temperature_c"].max()
        - df["temperature_c"].min()
    )

    # -----------------------------------------
    # Cooling-energy metrics
    # -----------------------------------------
    #
    # Pump/fan are shared actuators.
    # Therefore we MUST NOT sum one row per server,
    # or a two-server system would count their energy twice.
    # -----------------------------------------

    unique_control = (
        df[
            [
                "timestamp_s",
                "pump_power_est_w",
                "fan_power_est_w",
                "cooling_power_est_w",
            ]
        ]
        .drop_duplicates(
            subset=["timestamp_s"]
        )
        .sort_values(
            "timestamp_s"
        )
    )

    timestamps = (
        unique_control[
            "timestamp_s"
        ]
        .to_numpy(
            dtype=float
        )
    )

    cooling_power = (
        unique_control[
            "cooling_power_est_w"
        ]
        .to_numpy(
            dtype=float
        )
    )

    cooling_energy_wh = (
        _integrate_power_wh(
            timestamps,
            cooling_power,
        )
    )

    average_cooling_power_w = float(
        np.mean(
            cooling_power
        )
    )

    # -----------------------------------------
    # Heat removed
    # -----------------------------------------

    heat_removed_wh = 0.0

    for _, group in df.groupby(
        "server_id"
    ):

        group = (
            group.sort_values(
                "timestamp_s"
            )
        )

        heat_removed_wh += (
            _integrate_power_wh(
                group[
                    "timestamp_s"
                ].to_numpy(
                    dtype=float
                ),
                group[
                    "heat_removed_w"
                ].to_numpy(
                    dtype=float
                ),
            )
        )

    # -----------------------------------------
    # Flow metrics
    # -----------------------------------------

    average_flow_lpm = float(
        df["flow_rate_lpm"].mean()
    )

    maximum_flow_lpm = float(
        df["flow_rate_lpm"].max()
    )

    average_coolant_delta_t_c = float(
        df["delta_t_c"].mean()
    )

    response_time_s = (
        calculate_thermal_response_time(
            df
        )
    )

    return pd.DataFrame(
        [
            {
                "maximum_temperature_c":
                    maximum_temperature,

                "average_temperature_c":
                    average_temperature,

                "temperature_std_c":
                    temperature_std,

                "temperature_range_c":
                    temperature_range,

                "thermal_response_time_s":
                    response_time_s,

                "average_flow_lpm":
                    average_flow_lpm,

                "maximum_flow_lpm":
                    maximum_flow_lpm,

                "average_coolant_delta_t_c":
                    average_coolant_delta_t_c,

                "average_cooling_power_w":
                    average_cooling_power_w,

                "cooling_energy_wh":
                    cooling_energy_wh,

                "heat_removed_wh":
                    heat_removed_wh,
            }
        ]
    )


def calculate_metrics_from_csv(
    csv_path: str,
    pump_rated_power_w: float = 30.0,
    fan_rated_power_w: float = 20.0,
) -> pd.DataFrame:
    """Load a CSV and calculate the THERMO-X run metrics."""

    dataframe = pd.read_csv(
        csv_path
    )

    return calculate_run_metrics(
        dataframe,
        pump_rated_power_w=pump_rated_power_w,
        fan_rated_power_w=fan_rated_power_w,
    )