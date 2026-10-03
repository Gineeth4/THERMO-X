"""
Train the THERMO-X future-temperature prediction model.

Usage:

    python -m ml.train ^
        --input data/experiments/adaptive_cooling.csv ^
        --horizon 30

The model uses a time-ordered train/test split.

This is important because randomly mixing future and past
samples can produce an unrealistically optimistic evaluation
for a time-dependent thermal system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .features import (
    FEATURE_COLUMNS,
    create_prediction_dataset,
)

from .model_manager import (
    save_model_bundle,
)


def train_model(
    dataframe: pd.DataFrame,
    horizon_s: float,
    test_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[
    dict,
    pd.DataFrame,
]:
    """
    Train and evaluate a future-temperature predictor.

    The last portion of the time-ordered dataset is used
    as the test set.
    """

    dataset = create_prediction_dataset(
        dataframe,
        horizon_s=horizon_s,
    )

    if len(dataset) < 20:
        raise ValueError(
            "Not enough valid samples for ML training. "
            "Run a longer experiment."
        )

    if not 0.05 <= test_fraction < 0.5:
        raise ValueError(
            "test_fraction must be between 0.05 and 0.49."
        )

    split_index = int(
        len(dataset)
        * (1.0 - test_fraction)
    )

    if split_index <= 0:
        raise ValueError(
            "Training split is empty."
        )

    if split_index >= len(dataset):
        raise ValueError(
            "Test split is empty."
        )

    train = dataset.iloc[
        :split_index
    ].copy()

    test = dataset.iloc[
        split_index:
    ].copy()

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "target_future_temperature_c"
    ]

    X_test = test[
        FEATURE_COLUMNS
    ]

    y_test = test[
        "target_future_temperature_c"
    ]

    model = RandomForestRegressor(
        n_estimators=250,

        max_depth=None,

        min_samples_leaf=2,

        random_state=random_state,

        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = (
        model.predict(
            X_test
        )
    )

    mae = float(
        mean_absolute_error(
            y_test,
            predictions,
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_test,
                predictions,
            )
        )
    )

    r2 = float(
        r2_score(
            y_test,
            predictions,
        )
    )

    prediction_dataframe = (
        test[
            [
                "timestamp_s",
                "server_id",
                "temperature_c",
                "target_future_temperature_c",
            ]
        ]
        .copy()
    )

    prediction_dataframe[
        "predicted_future_temperature_c"
    ] = predictions

    prediction_dataframe[
        "prediction_error_c"
    ] = (
        prediction_dataframe[
            "predicted_future_temperature_c"
        ]
        - prediction_dataframe[
            "target_future_temperature_c"
        ]
    )

    metrics = {
        "horizon_s": horizon_s,

        "sample_count": len(
            dataset
        ),

        "training_samples": len(
            train
        ),

        "test_samples": len(
            test
        ),

        "mae_c": mae,

        "rmse_c": rmse,

        "r2": r2,

        "model": (
            "RandomForestRegressor"
        ),

        "feature_columns": (
            FEATURE_COLUMNS
        ),
    }

    bundle = {
        "model": model,

        "feature_columns": (
            FEATURE_COLUMNS
        ),

        "horizon_s": horizon_s,

        "metrics": metrics,
    }

    return (
        bundle,
        prediction_dataframe,
    )


def main() -> None:
    """Command-line training entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the THERMO-X future-temperature predictor."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input THERMO-X telemetry CSV.",
    )

    parser.add_argument(
        "--horizon",
        type=float,
        default=30.0,
        help=(
            "Prediction horizon in seconds. "
            "Default: 30."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/thermal_predictor.joblib"
        ),
        help="Output model path.",
    )

    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(
            "models/thermal_predictor_metrics.json"
        ),
        help="Output metrics JSON.",
    )

    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path(
            "data/experiments/"
            "adaptive_predictions.csv"
        ),
        help="Output test predictions CSV.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {args.input}"
        )

    dataframe = pd.read_csv(
        args.input
    )

    (
        bundle,
        predictions,
    ) = train_model(
        dataframe=dataframe,
        horizon_s=args.horizon,
    )

    model_path = (
        save_model_bundle(
            bundle,
            args.output,
        )
    )

    args.metrics_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.metrics_output.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            bundle["metrics"],
            file,
            indent=2,
        )

    args.predictions_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        args.predictions_output,
        index=False,
    )

    print()
    print(
        "=========================================="
    )

    print(
        "THERMO-X ML TRAINING COMPLETE"
    )

    print(
        "=========================================="
    )

    print()

    print(
        f"Prediction horizon : "
        f"{bundle['horizon_s']:.1f} s"
    )

    print(
        f"Samples            : "
        f"{bundle['metrics']['sample_count']}"
    )

    print(
        f"Training samples   : "
        f"{bundle['metrics']['training_samples']}"
    )

    print(
        f"Test samples       : "
        f"{bundle['metrics']['test_samples']}"
    )

    print(
        f"MAE                : "
        f"{bundle['metrics']['mae_c']:.4f} °C"
    )

    print(
        f"RMSE               : "
        f"{bundle['metrics']['rmse_c']:.4f} °C"
    )

    print(
        f"R²                 : "
        f"{bundle['metrics']['r2']:.4f}"
    )

    print()

    print(
        f"Model saved to     : "
        f"{model_path}"
    )

    print(
        f"Metrics saved to   : "
        f"{args.metrics_output}"
    )

    print(
        f"Predictions saved  : "
        f"{args.predictions_output}"
    )


if __name__ == "__main__":
    main()