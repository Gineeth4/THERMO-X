"""
Engineering plots for THERMO-X experiments.

Plots are saved to disk rather than displayed interactively.
The Streamlit dashboard will later use similar data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_directory(
    output_directory: str | Path,
) -> Path:
    """Create the plot output directory."""

    directory = Path(
        output_directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def plot_temperature_vs_time(
    dataframe: pd.DataFrame,
    output_directory: str | Path,
    filename: str = "temperature_vs_time.png",
) -> Path:
    """Plot temperature versus simulation time."""

    output_directory = (
        _prepare_output_directory(
            output_directory
        )
    )

    fig, ax = plt.subplots()

    for server_id, group in dataframe.groupby(
        "server_id"
    ):

        group = group.sort_values(
            "timestamp_s"
        )

        ax.plot(
            group["timestamp_s"],
            group["temperature_c"],
            label=server_id,
        )

    ax.set_title(
        "THERMO-X Temperature vs Time"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Temperature (°C)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_directory
        / filename
    )

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)

    return path


def plot_workload_vs_temperature(
    dataframe: pd.DataFrame,
    output_directory: str | Path,
    filename: str = "workload_vs_temperature.png",
) -> Path:
    """Plot workload against thermal response."""

    output_directory = (
        _prepare_output_directory(
            output_directory
        )
    )

    fig, ax = plt.subplots()

    ax.scatter(
        dataframe["workload_pct"],
        dataframe["temperature_c"],
        s=12,
        alpha=0.6,
    )

    ax.set_title(
        "THERMO-X Workload vs Temperature"
    )

    ax.set_xlabel(
        "Workload (%)"
    )

    ax.set_ylabel(
        "Temperature (°C)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    path = (
        output_directory
        / filename
    )

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)

    return path


def plot_cooling_power(
    dataframe: pd.DataFrame,
    output_directory: str | Path,
    pump_rated_power_w: float = 30.0,
    fan_rated_power_w: float = 20.0,
    filename: str = "cooling_power_vs_time.png",
) -> Path:
    """Plot estimated pump/fan/cooling power."""

    output_directory = (
        _prepare_output_directory(
            output_directory
        )
    )

    df = dataframe[
        [
            "timestamp_s",
            "pump_speed_pct",
            "fan_speed_pct",
        ]
    ].drop_duplicates(
        subset=[
            "timestamp_s"
        ]
    ).sort_values(
        "timestamp_s"
    )

    df["pump_power_w"] = (
        df["pump_speed_pct"]
        / 100.0
        * pump_rated_power_w
    )

    df["fan_power_w"] = (
        df["fan_speed_pct"]
        / 100.0
        * fan_rated_power_w
    )

    df["cooling_power_w"] = (
        df["pump_power_w"]
        + df["fan_power_w"]
    )

    fig, ax = plt.subplots()

    ax.plot(
        df["timestamp_s"],
        df["cooling_power_w"],
        label="Total Cooling Power",
    )

    ax.set_title(
        "THERMO-X Cooling Power vs Time"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Power (W)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_directory
        / filename
    )

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)

    return path


def plot_fixed_vs_adaptive_temperature(
    fixed_dataframe: pd.DataFrame,
    adaptive_dataframe: pd.DataFrame,
    output_directory: str | Path,
    filename: str = "fixed_vs_adaptive_temperature.png",
) -> Path:
    """
    Compare average temperature of the thermal zones
    between fixed and adaptive runs.
    """

    output_directory = (
        _prepare_output_directory(
            output_directory
        )
    )

    fixed = (
        fixed_dataframe
        .groupby("timestamp_s")[
            "temperature_c"
        ]
        .mean()
        .reset_index()
    )

    adaptive = (
        adaptive_dataframe
        .groupby("timestamp_s")[
            "temperature_c"
        ]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots()

    ax.plot(
        fixed["timestamp_s"],
        fixed["temperature_c"],
        label="Fixed Cooling",
    )

    ax.plot(
        adaptive["timestamp_s"],
        adaptive["temperature_c"],
        label="Adaptive Cooling",
    )

    ax.set_title(
        "THERMO-X Fixed vs Adaptive Cooling"
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Average Temperature (°C)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_directory
        / filename
    )

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)

    return path