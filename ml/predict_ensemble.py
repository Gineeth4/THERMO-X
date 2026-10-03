"""
THERMO-X Four-Model Ensemble Inference.

Used for:

    - dashboard predictions
    - new simulation scenarios
    - future ESP32 telemetry

The same trained ensemble can therefore eventually consume
real sensor data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .ensemble import (
    create_lstm_sequences,
)

from .features import (
    FEATURE_COLUMNS,
    add_feature_columns,
)

from .neural_models import (
    DEVICE,
    ThermalLSTM,
    ThermalMLP,
)


def load_ensemble_bundle(
    path: str | Path = (
        "models/"
        "thermo_x_ensemble.joblib"
    ),
) -> dict:

    model_path = Path(
        path
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Ensemble model not found: "
            f"{model_path}"
        )

    return joblib.load(
        model_path
    )


def _build_models(
    bundle: dict,
) -> tuple:

    mlp = ThermalMLP(
        input_size=bundle[
            "input_size"
        ]
    ).to(
        DEVICE
    )

    mlp.load_state_dict(
        bundle[
            "mlp_state_dict"
        ]
    )

    mlp.eval()

    lstm = ThermalLSTM(
        input_size=bundle[
            "input_size"
        ],
        hidden_size=64,
        num_layers=2,
        dropout=0.15,
    ).to(
        DEVICE
    )

    lstm.load_state_dict(
        bundle[
            "lstm_state_dict"
        ]
    )

    lstm.eval()

    return (
        mlp,
        lstm,
    )


def _predict_live_lstm(
    bundle: dict,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict for live/new telemetry.

    Unlike training, there may be no target column.
    """

    sequence_length = int(
        bundle[
            "sequence_length"
        ]
    )

    feature_df = (
        add_feature_columns(
            dataframe
        )
    )

    group_columns = []

    if "scenario_id" in feature_df.columns:

        group_columns.append(
            "scenario_id"
        )

    group_columns.append(
        "server_id"
    )

    sequences = []

    metadata = []

    for (
        group_key,
        group,
    ) in feature_df.groupby(
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

        values = (
            group[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        for end_index in range(
            sequence_length - 1,
            len(group),
        ):

            start_index = (
                end_index
                - sequence_length
                + 1
            )

            sequences.append(
                values[
                    start_index:
                    end_index + 1
                ]
            )

            metadata.append(
                {
                    "scenario_id":
                        (
                            group.loc[
                                end_index,
                                "scenario_id"
                            ]
                            if
                            "scenario_id"
                            in group.columns
                            else "LIVE"
                        ),

                    "server_id":
                        group.loc[
                            end_index,
                            "server_id"
                        ],

                    "timestamp_s":
                        float(
                            group.loc[
                                end_index,
                                "timestamp_s"
                            ]
                        ),
                }
            )

    if not sequences:

        return pd.DataFrame(
            columns=[
                "scenario_id",
                "server_id",
                "timestamp_s",
                "LSTM",
            ]
        )

    sequences = np.asarray(
        sequences,
        dtype=np.float32,
    )

    scaler_x = bundle[
        "lstm_scaler_x"
    ]

    scaler_y = bundle[
        "lstm_scaler_y"
    ]

    X_scaled = (
        scaler_x.transform(
            sequences.reshape(
                -1,
                len(
                    FEATURE_COLUMNS
                ),
            )
        )
        .reshape(
            sequences.shape
        )
        .astype(
            np.float32
        )
    )

    _, lstm = _build_models(
        bundle
    )

    with torch.no_grad():

        prediction_scaled = (
            lstm(
                torch.tensor(
                    X_scaled,
                    dtype=torch.float32,
                    device=DEVICE,
                )
            )
            .cpu()
            .numpy()
        )

    prediction_delta = (
        scaler_y
        .inverse_transform(
            prediction_scaled
        )
        .reshape(
            -1
        )
    )

    output = pd.DataFrame(
        metadata
    )

    output[
        "LSTM"
    ] = prediction_delta

    return output


def predict_dataframe(
    dataframe: pd.DataFrame,
    bundle: dict | None = None,
) -> pd.DataFrame:
    """
    Generate four-model and ensemble predictions.
    """

    if bundle is None:

        bundle = (
            load_ensemble_bundle()
        )

    feature_df = (
        add_feature_columns(
            dataframe
        )
    )

    rf = bundle[
        "random_forest"
    ]

    xgb = bundle[
        "xgboost"
    ]

    mlp, _ = _build_models(
        bundle
    )

    scaler_x = bundle[
        "mlp_scaler_x"
    ]

    scaler_y = bundle[
        "mlp_scaler_y"
    ]

    X = (
        feature_df[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # Tree models
    # --------------------------------------------------------

    rf_delta = rf.predict(
        feature_df[
            FEATURE_COLUMNS
        ]
    )

    xgb_delta = xgb.predict(
        feature_df[
            FEATURE_COLUMNS
        ]
    )

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    X_scaled = (
        scaler_x
        .transform(
            X
        )
        .astype(
            np.float32
        )
    )

    with torch.no_grad():

        mlp_scaled = (
            mlp(
                torch.tensor(
                    X_scaled,
                    dtype=torch.float32,
                    device=DEVICE,
                )
            )
            .cpu()
            .numpy()
        )

    mlp_delta = (
        scaler_y
        .inverse_transform(
            mlp_scaled
        )
        .reshape(
            -1
        )
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm_predictions = (
        _predict_live_lstm(
            bundle,
            dataframe,
        )
    )

    # --------------------------------------------------------
    # Base prediction table
    # --------------------------------------------------------

    result = feature_df[
        [
            *(
                [
                    "scenario_id"
                ]
                if
                "scenario_id"
                in feature_df.columns
                else []
            ),

            "timestamp_s",

            "server_id",

            "temperature_c",
        ]
    ].copy()

    result[
        "Random Forest"
    ] = rf_delta

    result[
        "XGBoost"
    ] = xgb_delta

    result[
        "Deep MLP"
    ] = mlp_delta

    result = (
        result
        .merge(
            lstm_predictions,
            on=[
                *(
                    [
                        "scenario_id"
                    ]
                    if
                    "scenario_id"
                    in result.columns
                    else []
                ),

                "server_id",

                "timestamp_s",
            ],
            how="left",
        )
    )

    weights = bundle[
        "weights"
    ]

    model_columns = [
        "Random Forest",
        "XGBoost",
        "Deep MLP",
        "LSTM",
    ]

    model_matrix = (
        result[
            model_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    weight_matrix = np.array(
        [
            weights[
                model_name
            ]
            for model_name
            in model_columns
        ]
    )

    # --------------------------------------------------------
    # Adaptive weighting when LSTM history is unavailable.
    # --------------------------------------------------------

    ensemble_delta = []

    spread = []

    for row in model_matrix:

        finite = np.isfinite(
            row
        )

        if not finite.any():

            ensemble_delta.append(
                np.nan
            )

            spread.append(
                np.nan
            )

            continue

        active_weights = (
            weight_matrix[
                finite
            ]
        )

        active_predictions = (
            row[
                finite
            ]
        )

        active_weights = (
            active_weights
            / active_weights.sum()
        )

        prediction = float(
            active_predictions
            @ active_weights
        )

        ensemble_delta.append(
            prediction
        )

        spread.append(
            float(
                np.std(
                    active_predictions
                )
            )
        )

    result[
        "ensemble_delta_c"
    ] = ensemble_delta

    result[
        "ensemble_future_temperature_c"
    ] = (
        result[
            "temperature_c"
        ]
        + result[
            "ensemble_delta_c"
        ]
    )

    result[
        "ensemble_spread_c"
    ] = spread

    # Individual future temperatures.
    for model_name in model_columns:

        result[
            f"{model_name}_future_temperature_c"
        ] = (
            result[
                "temperature_c"
            ]
            + result[
                model_name
            ]
        )

    return result


def predict_csv(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Run ensemble prediction on a CSV file."""

    dataframe = pd.read_csv(
        input_path
    )

    bundle = (
        load_ensemble_bundle()
    )

    predictions = (
        predict_dataframe(
            dataframe,
            bundle,
        )
    )

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output,
        index=False,
    )

    return output


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run THERMO-X four-model ensemble prediction."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    output = predict_csv(
        args.input,
        args.output,
    )

    print(
        f"Prediction file saved to: {output}"
    )


if __name__ == "__main__":
    main()