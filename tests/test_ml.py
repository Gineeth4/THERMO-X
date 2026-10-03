"""
Tests for the THERMO-X machine-learning layer.

The test dataset intentionally contains:

    - scenario_id
    - server_id
    - valve_position_pct
    - thermal telemetry
    - temporal variation

This matches the current four-model ensemble architecture.
"""

import numpy as np
import pandas as pd

from ml.features import (
    FEATURE_COLUMNS,
    add_feature_columns,
    create_prediction_dataset,
)

from ml.ensemble import (
    optimize_ensemble_weights,
    split_by_scenario,
)


# ============================================================
# TEST DATASET
# ============================================================


def create_small_dataset(
    scenario_count: int = 4,
    duration_s: int = 80,
) -> pd.DataFrame:
    """
    Create a deterministic synthetic ML test dataset.

    The dataset is deliberately small so tests execute quickly.

    There are enough samples for a 30-second prediction horizon.
    """

    rows = []

    for scenario_index in range(
        scenario_count
    ):

        scenario_id = (
            f"SCENARIO_{scenario_index + 1:03d}"
        )

        # Slightly different starting conditions for each
        # scenario so that the dataset is not completely identical.
        ambient = (
            28.0
            + scenario_index
        )

        for server_index, server_id in enumerate(
            [
                "SERVER_A",
                "SERVER_B",
            ]
        ):

            for second in range(
                duration_s
            ):

                # Create gradual thermal behaviour.
                temperature = (
                    35.0

                    + 0.04
                    * second

                    + 0.50
                    * server_index

                    + 0.15
                    * scenario_index
                )

                workload = (
                    30.0
                    + (
                        (
                            second
                            // 10
                        )
                        % 4
                    )
                    * 20.0
                )

                heater_power = (
                    workload
                    / 100.0
                    * 100.0
                )

                coolant_inlet = (
                    ambient
                )

                coolant_outlet = (
                    ambient
                    + 0.5
                    + 0.01
                    * workload
                )

                flow_rate = (
                    0.8
                    + 0.01
                    * (
                        second
                        % 10
                    )
                )

                pump_speed = (
                    35.0
                    + 0.25
                    * workload
                )

                fan_speed = (
                    35.0
                    + 0.20
                    * workload
                )

                valve_position = (
                    40.0
                    + 0.30
                    * workload
                    + 2.0
                    * server_index
                )

                pressure = (
                    80.0
                    + 10.0
                    * flow_rate
                )

                rows.append(
                    {
                        "scenario_id":
                            scenario_id,

                        "timestamp_s":
                            float(
                                second
                            ),

                        "server_id":
                            server_id,

                        "workload_pct":
                            workload,

                        "heater_power_w":
                            heater_power,

                        "temperature_c":
                            temperature,

                        "coolant_inlet_c":
                            coolant_inlet,

                        "coolant_outlet_c":
                            coolant_outlet,

                        "flow_rate_lpm":
                            flow_rate,

                        "pressure_kpa":
                            pressure,

                        "ambient_temperature_c":
                            ambient,

                        "pump_speed_pct":
                            pump_speed,

                        "fan_speed_pct":
                            fan_speed,

                        "valve_position_pct":
                            valve_position,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FEATURE TESTS
# ============================================================


def test_feature_creation() -> None:

    dataframe = (
        create_small_dataset()
    )

    result = (
        add_feature_columns(
            dataframe
        )
    )

    # Existing physical features.
    assert (
        "delta_t_c"
        in result.columns
    )

    # Temporal features.
    assert (
        "temperature_rate_c_per_s"
        in result.columns
    )

    assert (
        "temperature_lag_1s_c"
        in result.columns
    )

    assert (
        "temperature_lag_5s_c"
        in result.columns
    )

    assert (
        "temperature_lag_15s_c"
        in result.columns
    )

    assert (
        "temperature_rolling_mean_5s_c"
        in result.columns
    )

    assert (
        "temperature_rolling_mean_15s_c"
        in result.columns
    )


def test_all_model_features_exist() -> None:

    dataframe = (
        create_small_dataset()
    )

    result = (
        add_feature_columns(
            dataframe
        )
    )

    for feature in (
        FEATURE_COLUMNS
    ):

        assert feature in (
            result.columns
        )


def test_feature_values_are_finite() -> None:

    dataframe = (
        create_small_dataset()
    )

    result = (
        add_feature_columns(
            dataframe
        )
    )

    numeric_features = (
        result[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )

    assert np.isfinite(
        numeric_features
    ).all()


# ============================================================
# SUPERVISED TARGET
# ============================================================


def test_prediction_dataset_has_target() -> None:

    dataframe = (
        create_small_dataset()
    )

    result = (
        create_prediction_dataset(
            dataframe,
            horizon_s=30.0,
        )
    )

    assert (
        "target_future_temperature_c"
        in result.columns
    )

    assert (
        "target_delta_temperature_c"
        in result.columns
    )

    assert len(
        result
    ) > 0


def test_prediction_target_is_future_temperature() -> None:

    dataframe = (
        create_small_dataset(
            scenario_count=1
        )
    )

    result = (
        create_prediction_dataset(
            dataframe,
            horizon_s=30.0,
        )
    )

    first_row = (
        result
        .sort_values(
            [
                "server_id",
                "timestamp_s",
            ]
        )
        .iloc[0]
    )

    # Current temperature at t=0:
    #
    # 35.0 C for SERVER_A.
    #
    # Future temperature at t=30:
    #
    # 35 + 0.04 × 30 = 36.2 C.
    expected_future = (
        35.0
        + 0.04
        * 30
    )

    assert abs(
        first_row[
            "target_future_temperature_c"
        ]
        - expected_future
    ) < 1e-6


def test_future_delta_target_is_correct() -> None:

    dataframe = (
        create_small_dataset(
            scenario_count=1
        )
    )

    result = (
        create_prediction_dataset(
            dataframe,
            horizon_s=30.0,
        )
    )

    first_row = (
        result[
            result[
                "server_id"
            ]
            == "SERVER_A"
        ]
        .sort_values(
            "timestamp_s"
        )
        .iloc[0]
    )

    current = (
        first_row[
            "temperature_c"
        ]
    )

    future = (
        first_row[
            "target_future_temperature_c"
        ]
    )

    delta = (
        first_row[
            "target_delta_temperature_c"
        ]
    )

    assert abs(
        (
            future
            - current
        )
        - delta
    ) < 1e-9


# ============================================================
# SCENARIO SPLITTING
# ============================================================


def test_scenario_split_does_not_overlap() -> None:
    """
    Train/validation/test scenario IDs must not overlap.

    This protects the ensemble from scenario-level leakage.
    """

    dataframe = (
        create_small_dataset(
            scenario_count=20
        )
    )

    dataset = (
        create_prediction_dataset(
            dataframe,
            horizon_s=30.0,
        )
    )

    train, validation, test = (
        split_by_scenario(
            dataset,
            seed=42,
        )
    )

    train_ids = set(
        train[
            "scenario_id"
        ]
        .unique()
    )

    validation_ids = set(
        validation[
            "scenario_id"
        ]
        .unique()
    )

    test_ids = set(
        test[
            "scenario_id"
        ]
        .unique()
    )

    assert (
        train_ids.isdisjoint(
            validation_ids
        )
    )

    assert (
        train_ids.isdisjoint(
            test_ids
        )
    )

    assert (
        validation_ids.isdisjoint(
            test_ids
        )
    )


def test_scenario_split_contains_all_data() -> None:

    dataframe = (
        create_small_dataset(
            scenario_count=20
        )
    )

    dataset = (
        create_prediction_dataset(
            dataframe,
            horizon_s=30.0,
        )
    )

    train, validation, test = (
        split_by_scenario(
            dataset,
            seed=42,
        )
    )

    assert (
        len(train)
        + len(validation)
        + len(test)
        == len(dataset)
    )


# ============================================================
# ENSEMBLE WEIGHTS
# ============================================================


def test_ensemble_weights_sum_to_one() -> None:
    """
    The four model weights must form a valid convex combination.
    """

    validation = pd.DataFrame(
        {
            "temperature_c": [
                40.0,
                41.0,
                42.0,
                43.0,
                44.0,
            ],

            "target_future_temperature_c": [
                42.0,
                43.0,
                44.0,
                45.0,
                46.0,
            ],

            "Random Forest": [
                2.0,
                2.0,
                2.0,
                2.0,
                2.0,
            ],

            "XGBoost": [
                2.1,
                1.9,
                2.0,
                2.1,
                2.0,
            ],

            "Deep MLP": [
                1.7,
                2.2,
                1.8,
                2.3,
                1.9,
            ],

            "LSTM": [
                2.4,
                1.6,
                2.3,
                1.7,
                2.2,
            ],
        }
    )

    weights = (
        optimize_ensemble_weights(
            validation
        )
    )

    assert set(
        weights.keys()
    ) == {
        "Random Forest",
        "XGBoost",
        "Deep MLP",
        "LSTM",
    }

    assert abs(
        sum(
            weights.values()
        )
        - 1.0
    ) < 1e-8


def test_each_ensemble_model_has_minimum_weight() -> None:

    validation = pd.DataFrame(
        {
            "temperature_c": np.linspace(
                40.0,
                50.0,
                10,
            ),

            "target_future_temperature_c":
                np.linspace(
                    42.0,
                    52.0,
                    10,
                ),

            "Random Forest":
                np.full(
                    10,
                    2.0,
                ),

            "XGBoost":
                np.full(
                    10,
                    2.05,
                ),

            "Deep MLP":
                np.full(
                    10,
                    1.95,
                ),

            "LSTM":
                np.full(
                    10,
                    2.10,
                ),
        }
    )

    weights = (
        optimize_ensemble_weights(
            validation
        )
    )

    for weight in (
        weights.values()
    ):

        assert weight >= 0.05