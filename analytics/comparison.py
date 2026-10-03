"""
Fixed-vs-adaptive comparison for THERMO-X.

The comparison produces numerical differences and percentage
changes only.

It does NOT assume that adaptive cooling must perform better.

The experiment determines the result.
"""

from __future__ import annotations

import pandas as pd

from .metrics import (
    calculate_run_metrics,
)


def _percentage_change(
    baseline: float,
    comparison: float,
) -> float:
    """
    Calculate:

        (baseline - comparison) / baseline × 100

    This is useful for metrics where a lower value may indicate
    reduced resource use.

    Returns NaN when a percentage change is undefined.
    """

    if baseline == 0:
        return float("nan")

    return (
        (
            baseline
            - comparison
        )
        / baseline
        * 100.0
    )


def compare_runs(
    fixed_dataframe: pd.DataFrame,
    adaptive_dataframe: pd.DataFrame,
    pump_rated_power_w: float = 30.0,
    fan_rated_power_w: float = 20.0,
) -> pd.DataFrame:
    """
    Compare two THERMO-X runs.

    The expected usage is:

        fixed = same workload, fixed cooling
        adaptive = same workload, adaptive cooling
    """

    fixed_metrics = (
        calculate_run_metrics(
            fixed_dataframe,
            pump_rated_power_w=pump_rated_power_w,
            fan_rated_power_w=fan_rated_power_w,
        )
        .iloc[0]
    )

    adaptive_metrics = (
        calculate_run_metrics(
            adaptive_dataframe,
            pump_rated_power_w=pump_rated_power_w,
            fan_rated_power_w=fan_rated_power_w,
        )
        .iloc[0]
    )

    rows = []

    lower_is_better_metrics = {
        "maximum_temperature_c",
        "average_temperature_c",
        "temperature_std_c",
        "temperature_range_c",
        "thermal_response_time_s",
        "average_cooling_power_w",
        "cooling_energy_wh",
    }

    for metric in fixed_metrics.index:

        fixed_value = float(
            fixed_metrics[metric]
        )

        adaptive_value = float(
            adaptive_metrics[metric]
        )

        # Percentage reduction.
        #
        # For metrics such as heat removed, a lower value isn't
        # automatically better, so the percentage is still provided
        # as a mathematical comparison rather than a verdict.
        percentage_difference = (
            _percentage_change(
                fixed_value,
                adaptive_value,
            )
        )

        rows.append(
            {
                "metric": metric,

                "fixed_value": fixed_value,

                "adaptive_value": adaptive_value,

                "percentage_difference": (
                    percentage_difference
                ),

                "lower_is_better_metric": (
                    metric
                    in lower_is_better_metrics
                ),
            }
        )

    return pd.DataFrame(
        rows
    )