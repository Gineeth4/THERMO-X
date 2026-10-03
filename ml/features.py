"""
THERMO-X ML feature engineering.

The ML system predicts:

    FUTURE TEMPERATURE CHANGE

rather than directly predicting absolute temperature.

This allows the models to learn:

    current thermal state
            ↓
    expected thermal change
            ↓
    future temperature
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# BASE FEATURES
# ============================================================

BASE_FEATURE_COLUMNS = [

    "temperature_c",

    "workload_pct",

    "heater_power_w",

    "coolant_inlet_c",

    "coolant_outlet_c",

    "delta_t_c",

    "flow_rate_lpm",

    "pressure_kpa",

    "ambient_temperature_c",

    "pump_speed_pct",

    "fan_speed_pct",

    "valve_position_pct",
]


# ============================================================
# TEMPORAL FEATURES
# ============================================================

TEMPORAL_FEATURE_COLUMNS = [

    "temperature_rate_c_per_s",

    "temperature_lag_1s_c",

    "temperature_lag_5s_c",

    "temperature_lag_15s_c",

    "workload_lag_1s_pct",

    "flow_lag_1s_lpm",

    "temperature_rolling_mean_5s_c",

    "temperature_rolling_mean_15s_c",

    "workload_rolling_mean_5s_pct",

    "flow_rolling_mean_5s_lpm",
]


FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + TEMPORAL_FEATURE_COLUMNS
)


# ============================================================
# GROUPING
# ============================================================


def get_group_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Determine how telemetry should be grouped.

    Scenario ID prevents temporal features from crossing
    from one independent experiment into another.
    """

    if "scenario_id" in dataframe.columns:

        return [
            "scenario_id",
            "server_id",
        ]

    return [
        "server_id"
    ]


# ============================================================
# VALIDATION
# ============================================================


def validate_input_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """Validate required telemetry columns."""

    required = {

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
    }

    missing = (
        required
        - set(dataframe.columns)
    )

    if missing:

        raise ValueError(
            "Missing required ML columns: "
            + ", ".join(
                sorted(missing)
            )
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def add_feature_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create physical + temporal ML features."""

    validate_input_dataframe(
        dataframe
    )

    group_columns = (
        get_group_columns(
            dataframe
        )
    )

    df = (
        dataframe
        .copy()
        .sort_values(
            [
                *group_columns,
                "timestamp_s",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    grouped = df.groupby(
        group_columns,
        sort=False,
    )

    # --------------------------------------------------------
    # Coolant temperature rise.
    # --------------------------------------------------------

    df["delta_t_c"] = (
        df["coolant_outlet_c"]
        - df["coolant_inlet_c"]
    )

    # --------------------------------------------------------
    # Temperature rate of change.
    # --------------------------------------------------------

    dt = grouped[
        "timestamp_s"
    ].diff()

    d_temperature = grouped[
        "temperature_c"
    ].diff()

    df[
        "temperature_rate_c_per_s"
    ] = (
        d_temperature
        / dt.replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Lag features.
    # --------------------------------------------------------

    df[
        "temperature_lag_1s_c"
    ] = grouped[
        "temperature_c"
    ].shift(1)

    df[
        "temperature_lag_5s_c"
    ] = grouped[
        "temperature_c"
    ].shift(5)

    df[
        "temperature_lag_15s_c"
    ] = grouped[
        "temperature_c"
    ].shift(15)

    df[
        "workload_lag_1s_pct"
    ] = grouped[
        "workload_pct"
    ].shift(1)

    df[
        "flow_lag_1s_lpm"
    ] = grouped[
        "flow_rate_lpm"
    ].shift(1)

    # --------------------------------------------------------
    # Rolling thermal history.
    # --------------------------------------------------------

    df[
        "temperature_rolling_mean_5s_c"
    ] = grouped[
        "temperature_c"
    ].transform(
        lambda series:
        series.rolling(
            window=5,
            min_periods=1,
        ).mean()
    )

    df[
        "temperature_rolling_mean_15s_c"
    ] = grouped[
        "temperature_c"
    ].transform(
        lambda series:
        series.rolling(
            window=15,
            min_periods=1,
        ).mean()
    )

    df[
        "workload_rolling_mean_5s_pct"
    ] = grouped[
        "workload_pct"
    ].transform(
        lambda series:
        series.rolling(
            window=5,
            min_periods=1,
        ).mean()
    )

    df[
        "flow_rolling_mean_5s_lpm"
    ] = grouped[
        "flow_rate_lpm"
    ].transform(
        lambda series:
        series.rolling(
            window=5,
            min_periods=1,
        ).mean()
    )

    # --------------------------------------------------------
    # Handle invalid numerical values.
    # --------------------------------------------------------

    numeric_columns = (
        FEATURE_COLUMNS
    )

    df[
        numeric_columns
    ] = (
        df[
            numeric_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    df[
        numeric_columns
    ] = (
        df
        .groupby(
            group_columns,
            sort=False,
        )[
            numeric_columns
        ]
        .transform(
            lambda group:
            group
            .bfill()
            .ffill()
        )
        .fillna(0.0)
    )

    return df


# ============================================================
# SUPERVISED DATASET
# ============================================================


def create_prediction_dataset(
    dataframe: pd.DataFrame,
    horizon_s: float = 30.0,
) -> pd.DataFrame:
    """
    Create supervised learning samples.

    Target:

        future temperature change

    The future sample is always searched inside the same
    scenario and thermal zone.
    """

    if horizon_s <= 0:

        raise ValueError(
            "Prediction horizon must be positive."
        )

    df = add_feature_columns(
        dataframe
    )

    group_columns = (
        get_group_columns(
            df
        )
    )

    output_frames = []

    for (
        group_key,
        group,
    ) in df.groupby(
        group_columns,
        sort=False,
    ):

        group = (
            group
            .sort_values(
                "timestamp_s"
            )
            .reset_index(
                drop=True
            )
        )

        timestamps = (
            group[
                "timestamp_s"
            ]
            .to_numpy(
                dtype=float
            )
        )

        current_temperatures = (
            group[
                "temperature_c"
            ]
            .to_numpy(
                dtype=float
            )
        )

        future_temperatures = []

        valid = []

        for current_time in timestamps:

            target_time = (
                current_time
                + horizon_s
            )

            future_index = int(
                np.searchsorted(
                    timestamps,
                    target_time,
                    side="left",
                )
            )

            if (
                future_index
                >= len(timestamps)
            ):

                future_temperatures.append(
                    np.nan
                )

                valid.append(
                    False
                )

            else:

                future_temperatures.append(
                    current_temperatures[
                        future_index
                    ]
                )

                valid.append(
                    True
                )

        group[
            "target_future_temperature_c"
        ] = future_temperatures

        group[
            "_valid_target"
        ] = valid

        group[
            "target_delta_temperature_c"
        ] = (
            group[
                "target_future_temperature_c"
            ]
            - group[
                "temperature_c"
            ]
        )

        valid_group = group[
            group[
                "_valid_target"
            ]
        ].copy()

        if not valid_group.empty:

            output_frames.append(
                valid_group
            )

    if not output_frames:

        raise ValueError(
            "No valid future-temperature samples were created."
        )

    result = (
        pd.concat(
            output_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                *group_columns,
                "timestamp_s",
            ]
        )
        .reset_index(
            drop=True,
        )
    )

    return result[
        [
            *(
                [
                    "scenario_id"
                ]
                if "scenario_id"
                in result.columns
                else []
            ),

            "timestamp_s",

            "server_id",

            *FEATURE_COLUMNS,

            "target_future_temperature_c",

            "target_delta_temperature_c",
        ]
    ]