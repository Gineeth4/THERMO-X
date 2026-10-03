"""
Tests for THERMO-X engineering analytics.
"""

import pandas as pd

from analytics.comparison import (
    compare_runs,
)

from analytics.metrics import (
    calculate_run_metrics,
    validate_dataframe,
)


def create_test_dataframe() -> pd.DataFrame:
    """Create a small valid THERMO-X telemetry dataset."""

    return pd.DataFrame(
        [
            {
                "timestamp_s": 0.0,
                "server_id": "SERVER_A",
                "workload_pct": 20.0,
                "heater_power_w": 20.0,
                "temperature_c": 40.0,
                "coolant_inlet_c": 30.0,
                "coolant_outlet_c": 31.0,
                "flow_rate_lpm": 1.0,
                "pressure_kpa": 80.0,
                "ambient_temperature_c": 30.0,
                "pump_speed_pct": 50.0,
                "fan_speed_pct": 50.0,
            },

            {
                "timestamp_s": 1.0,
                "server_id": "SERVER_A",
                "workload_pct": 50.0,
                "heater_power_w": 50.0,
                "temperature_c": 42.0,
                "coolant_inlet_c": 30.0,
                "coolant_outlet_c": 32.0,
                "flow_rate_lpm": 1.0,
                "pressure_kpa": 80.0,
                "ambient_temperature_c": 30.0,
                "pump_speed_pct": 70.0,
                "fan_speed_pct": 70.0,
            },

            {
                "timestamp_s": 0.0,
                "server_id": "SERVER_B",
                "workload_pct": 20.0,
                "heater_power_w": 20.0,
                "temperature_c": 41.0,
                "coolant_inlet_c": 30.0,
                "coolant_outlet_c": 31.0,
                "flow_rate_lpm": 1.0,
                "pressure_kpa": 80.0,
                "ambient_temperature_c": 30.0,
                "pump_speed_pct": 50.0,
                "fan_speed_pct": 50.0,
            },

            {
                "timestamp_s": 1.0,
                "server_id": "SERVER_B",
                "workload_pct": 50.0,
                "heater_power_w": 50.0,
                "temperature_c": 43.0,
                "coolant_inlet_c": 30.0,
                "coolant_outlet_c": 32.0,
                "flow_rate_lpm": 1.0,
                "pressure_kpa": 80.0,
                "ambient_temperature_c": 30.0,
                "pump_speed_pct": 70.0,
                "fan_speed_pct": 70.0,
            },
        ]
    )


# ============================================================
# VALIDATION
# ============================================================


def test_valid_dataframe_is_accepted() -> None:

    dataframe = (
        create_test_dataframe()
    )

    # Should not raise an exception.
    validate_dataframe(
        dataframe
    )


# ============================================================
# METRICS
# ============================================================


def test_metrics_calculation() -> None:

    dataframe = (
        create_test_dataframe()
    )

    metrics = (
        calculate_run_metrics(
            dataframe
        )
    )

    assert not metrics.empty

    row = metrics.iloc[0]

    assert (
        row[
            "maximum_temperature_c"
        ]
        == 43.0
    )

    assert (
        row[
            "average_temperature_c"
        ]
        == 41.5
    )

    assert (
        row[
            "temperature_range_c"
        ]
        == 3.0
    )

    assert (
        row[
            "average_flow_lpm"
        ]
        == 1.0
    )


def test_metrics_contain_required_outputs() -> None:

    dataframe = (
        create_test_dataframe()
    )

    metrics = (
        calculate_run_metrics(
            dataframe
        )
    )

    required_metrics = {
        "maximum_temperature_c",
        "average_temperature_c",
        "temperature_std_c",
        "temperature_range_c",
        "thermal_response_time_s",
        "average_flow_lpm",
        "maximum_flow_lpm",
        "average_coolant_delta_t_c",
        "average_cooling_power_w",
        "cooling_energy_wh",
        "heat_removed_wh",
    }

    assert required_metrics.issubset(
        set(
            metrics.columns
        )
    )


# ============================================================
# COMPARISON
# ============================================================


def test_fixed_vs_adaptive_comparison() -> None:

    fixed = (
        create_test_dataframe()
    )

    adaptive = (
        create_test_dataframe()
        .copy()
    )

    # Modify adaptive temperature slightly so the
    # comparison has a real numerical difference.
    adaptive[
        "temperature_c"
    ] = (
        adaptive[
            "temperature_c"
        ]
        + 1.0
    )

    comparison = (
        compare_runs(
            fixed_dataframe=fixed,
            adaptive_dataframe=adaptive,
        )
    )

    assert not comparison.empty

    assert (
        "metric"
        in comparison.columns
    )

    assert (
        "fixed_value"
        in comparison.columns
    )

    assert (
        "adaptive_value"
        in comparison.columns
    )

    assert (
        "percentage_difference"
        in comparison.columns
    )

    temperature_row = (
        comparison[
            comparison[
                "metric"
            ]
            == "maximum_temperature_c"
        ]
    )

    assert len(
        temperature_row
    ) == 1