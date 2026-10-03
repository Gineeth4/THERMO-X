"""
THERMO-X Four-Model Thermal Ensemble.

Models:

    1. Random Forest
    2. XGBoost
    3. Deep MLP
    4. LSTM

The target is future temperature change.

The final prediction is a validation-optimized weighted
combination of all four models.

The weights are learned from a separate validation set.

The final test set remains completely unseen during
ensemble-weight optimization.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from scipy.optimize import minimize

from sklearn.ensemble import (
    RandomForestRegressor,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.preprocessing import (
    StandardScaler,
)

from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from xgboost import (
    XGBRegressor,
)

from .features import (
    FEATURE_COLUMNS,
    create_prediction_dataset,
    get_group_columns,
)

from .neural_models import (
    DEVICE,
    ThermalLSTM,
    ThermalMLP,
)


MODEL_NAMES = [
    "Random Forest",
    "XGBoost",
    "Deep MLP",
    "LSTM",
]


SEQUENCE_LENGTH = 20


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


# ============================================================
# DATA SPLIT
# ============================================================


def split_by_scenario(
    dataframe: pd.DataFrame,
    seed: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split by scenario rather than by individual rows.

    This is essential for testing generalization to unseen
    thermal experiments.
    """

    if "scenario_id" not in dataframe.columns:

        raise ValueError(
            "scenario_id is required for ensemble training."
        )

    scenario_ids = (
        dataframe[
            "scenario_id"
        ]
        .drop_duplicates()
        .to_list()
    )

    if len(
        scenario_ids
    ) < 20:

        raise ValueError(
            "At least 20 independent scenarios are recommended."
        )

    rng = np.random.default_rng(
        seed
    )

    rng.shuffle(
        scenario_ids
    )

    total = len(
        scenario_ids
    )

    train_end = int(
        total * 0.70
    )

    validation_end = int(
        total * 0.85
    )

    train_ids = set(
        scenario_ids[
            :train_end
        ]
    )

    validation_ids = set(
        scenario_ids[
            train_end:
            validation_end
        ]
    )

    test_ids = set(
        scenario_ids[
            validation_end:
        ]
    )

    train = (
        dataframe[
            dataframe[
                "scenario_id"
            ].isin(
                train_ids
            )
        ]
        .copy()
    )

    validation = (
        dataframe[
            dataframe[
                "scenario_id"
            ].isin(
                validation_ids
            )
        ]
        .copy()
    )

    test = (
        dataframe[
            dataframe[
                "scenario_id"
            ].isin(
                test_ids
            )
        ]
        .copy()
    )

    return (
        train,
        validation,
        test,
    )


# ============================================================
# METRICS
# ============================================================


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:

    return {

        "MAE_C":
            float(
                mean_absolute_error(
                    actual,
                    predicted,
                )
            ),

        "RMSE_C":
            float(
                np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted,
                    )
                )
            ),

        "R2":
            float(
                r2_score(
                    actual,
                    predicted,
                )
            ),
    }


# ============================================================
# LSTM SEQUENCES
# ============================================================


def create_lstm_sequences(
    dataframe: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Build sequences independently for every scenario and server.
    """

    group_columns = (
        get_group_columns(
            dataframe
        )
    )

    sequences = []

    targets = []

    metadata = []

    for (
        group_key,
        group,
    ) in dataframe.groupby(
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

        target = (
            group[
                "target_delta_temperature_c"
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

            targets.append(
                target[
                    end_index
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
                            else "UNKNOWN"
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

                    "temperature_c":
                        float(
                            group.loc[
                                end_index,
                                "temperature_c"
                            ]
                        ),
                }
            )

    if not sequences:

        raise ValueError(
            "No LSTM sequences could be created."
        )

    return (
        np.asarray(
            sequences,
            dtype=np.float32,
        ),

        np.asarray(
            targets,
            dtype=np.float32,
        ),

        pd.DataFrame(
            metadata
        ),
    )


# ============================================================
# MLP TRAINING
# ============================================================


def train_mlp(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> tuple[
    ThermalMLP,
    StandardScaler,
    StandardScaler,
    np.ndarray,
    int,
]:
    """Train Deep MLP with validation-based early stopping."""

    X_train = train[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    y_train = train[
        "target_delta_temperature_c"
    ].to_numpy(
        dtype=np.float32
    ).reshape(
        -1,
        1,
    )

    X_val = validation[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    scaler_x = StandardScaler()

    X_train_scaled = (
        scaler_x
        .fit_transform(
            X_train
        )
        .astype(
            np.float32
        )
    )

    X_val_scaled = (
        scaler_x
        .transform(
            X_val
        )
        .astype(
            np.float32
        )
    )

    scaler_y = StandardScaler()

    y_train_scaled = (
        scaler_y
        .fit_transform(
            y_train
        )
        .astype(
            np.float32
        )
    )

    model = ThermalMLP(
        input_size=len(
            FEATURE_COLUMNS
        )
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    loss_function = nn.HuberLoss()

    train_x = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    train_y = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    val_x = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    best_state = (
        copy.deepcopy(
            model.state_dict()
        )
    )

    best_val_loss = float(
        "inf"
    )

    patience = 10

    patience_counter = 0

    best_epoch = 1

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    loader = DataLoader(
        TensorDataset(
            train_x,
            train_y,
        ),
        batch_size=256,
        shuffle=True,
        generator=generator,
    )

    for epoch in range(
        1,
        61,
    ):

        model.train()

        for batch_x, batch_y in loader:

            optimizer.zero_grad()

            prediction = model(
                batch_x
            )

            loss = loss_function(
                prediction,
                batch_y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

        model.eval()

        with torch.no_grad():

            validation_prediction = model(
                val_x
            )

            validation_target = torch.tensor(
                scaler_y.transform(
                    validation[
                        "target_delta_temperature_c"
                    ].to_numpy(
                        dtype=np.float32
                    ).reshape(
                        -1,
                        1,
                    )
                ),
                dtype=torch.float32,
                device=DEVICE,
            )

            validation_loss = float(
                loss_function(
                    validation_prediction,
                    validation_target,
                ).item()
            )

        if (
            validation_loss
            <
            best_val_loss
        ):

            best_val_loss = (
                validation_loss
            )

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            best_epoch = (
                epoch
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if (
                patience_counter
                >= patience
            ):

                break

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():

        val_scaled_prediction = (
            model(
                val_x
            )
            .cpu()
            .numpy()
        )

    val_prediction = (
        scaler_y
        .inverse_transform(
            val_scaled_prediction
        )
        .reshape(
            -1
        )
    )

    return (
        model,
        scaler_x,
        scaler_y,
        val_prediction,
        best_epoch,
    )


# ============================================================
# LSTM TRAINING
# ============================================================


def train_lstm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
) -> tuple[
    ThermalLSTM,
    StandardScaler,
    StandardScaler,
    pd.DataFrame,
    int,
]:
    """Train the temporal LSTM model."""

    (
        X_train_raw,
        y_train,
        _,
    ) = create_lstm_sequences(
        train
    )

    (
        X_val_raw,
        y_val,
        val_metadata,
    ) = create_lstm_sequences(
        validation
    )

    scaler_x = StandardScaler()

    scaler_x.fit(
        X_train_raw.reshape(
            -1,
            len(
                FEATURE_COLUMNS
            ),
        )
    )

    def scale_sequences(
        sequences: np.ndarray,
    ) -> np.ndarray:

        scaled = (
            scaler_x
            .transform(
                sequences.reshape(
                    -1,
                    len(
                        FEATURE_COLUMNS
                    ),
                )
            )
        )

        return scaled.reshape(
            sequences.shape
        ).astype(
            np.float32
        )

    X_train = scale_sequences(
        X_train_raw
    )

    X_val = scale_sequences(
        X_val_raw
    )

    scaler_y = StandardScaler()

    y_train_scaled = (
        scaler_y
        .fit_transform(
            y_train.reshape(
                -1,
                1,
            )
        )
        .astype(
            np.float32
        )
    )

    model = ThermalLSTM(
        input_size=len(
            FEATURE_COLUMNS
        ),
        hidden_size=64,
        num_layers=2,
        dropout=0.15,
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    loss_function = nn.HuberLoss()

    train_x = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=DEVICE,
    )

    train_y = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    val_x = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=DEVICE,
    )

    val_y_scaled = (
        scaler_y.transform(
            y_val.reshape(
                -1,
                1,
            )
        )
        .astype(
            np.float32
        )
    )

    val_y = torch.tensor(
        val_y_scaled,
        dtype=torch.float32,
        device=DEVICE,
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    loader = DataLoader(
        TensorDataset(
            train_x,
            train_y,
        ),
        batch_size=128,
        shuffle=True,
        generator=generator,
    )

    best_state = (
        copy.deepcopy(
            model.state_dict()
        )
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = 1

    patience = 8

    patience_counter = 0

    for epoch in range(
        1,
        41,
    ):

        model.train()

        for batch_x, batch_y in loader:

            optimizer.zero_grad()

            prediction = model(
                batch_x
            )

            loss = loss_function(
                prediction,
                batch_y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

        model.eval()

        with torch.no_grad():

            validation_prediction = model(
                val_x
            )

            validation_loss = float(
                loss_function(
                    validation_prediction,
                    val_y,
                ).item()
            )

        if (
            validation_loss
            <
            best_val_loss
        ):

            best_val_loss = (
                validation_loss
            )

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

            best_epoch = (
                epoch
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if (
                patience_counter
                >= patience
            ):

                break

    model.load_state_dict(
        best_state
    )

    model.eval()

    with torch.no_grad():

        val_scaled_prediction = (
            model(
                val_x
            )
            .cpu()
            .numpy()
        )

    val_prediction = (
        scaler_y
        .inverse_transform(
            val_scaled_prediction
        )
        .reshape(
            -1
        )
    )

    val_predictions = (
        val_metadata.copy()
    )

    val_predictions[
        "predicted_delta"
    ] = val_prediction

    return (
        model,
        scaler_x,
        scaler_y,
        val_predictions,
        best_epoch,
    )


# ============================================================
# LSTM PREDICTION
# ============================================================


def predict_lstm(
    model: ThermalLSTM,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Generate LSTM predictions with metadata."""

    (
        X_raw,
        _,
        metadata,
    ) = create_lstm_sequences(
        dataframe
    )

    X_scaled = (
        scaler_x.transform(
            X_raw.reshape(
                -1,
                len(
                    FEATURE_COLUMNS
                ),
            )
        )
        .reshape(
            X_raw.shape
        )
        .astype(
            np.float32
        )
    )

    model.eval()

    with torch.no_grad():

        prediction_scaled = (
            model(
                torch.tensor(
                    X_scaled,
                    dtype=torch.float32,
                    device=DEVICE,
                )
            )
            .cpu()
            .numpy()
        )

    prediction = (
        scaler_y
        .inverse_transform(
            prediction_scaled
        )
        .reshape(
            -1
        )
    )

    result = (
        metadata.copy()
    )

    result[
        "predicted_delta"
    ] = prediction

    return result


# ============================================================
# TABULAR PREDICTION
# ============================================================


def predict_tabular_models(
    rf,
    xgb,
    mlp,
    mlp_scaler_x,
    mlp_scaler_y,
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe[
        [
            "scenario_id",
            "server_id",
            "timestamp_s",
            "temperature_c",
            "target_future_temperature_c",
        ]
    ].copy()

    rf_delta = rf.predict(
        dataframe[
            FEATURE_COLUMNS
        ]
    )

    xgb_delta = xgb.predict(
        dataframe[
            FEATURE_COLUMNS
        ]
    )

    X = dataframe[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    X_scaled = (
        mlp_scaler_x
        .transform(
            X
        )
        .astype(
            np.float32
        )
    )

    mlp.eval()

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
        mlp_scaler_y
        .inverse_transform(
            mlp_scaled
        )
        .reshape(
            -1
        )
    )

    result[
        "Random Forest"
    ] = rf_delta

    result[
        "XGBoost"
    ] = xgb_delta

    result[
        "Deep MLP"
    ] = mlp_delta

    return result


# ============================================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================


def optimize_ensemble_weights(
    validation_predictions: pd.DataFrame,
) -> dict[str, float]:
    """
    Learn the contribution of all four models.

    Every model has a minimum 5% contribution.
    """

    model_columns = [
        "Random Forest",
        "XGBoost",
        "Deep MLP",
        "LSTM",
    ]

    matrix = (
        validation_predictions[
            model_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    current_temperature = (
        validation_predictions[
            "temperature_c"
        ]
        .to_numpy(
            dtype=float
        )
    )

    actual_future = (
        validation_predictions[
            "target_future_temperature_c"
        ]
        .to_numpy(
            dtype=float
        )
    )

    def objective(
        weights: np.ndarray,
    ) -> float:

        predicted_delta = (
            matrix @ weights
        )

        predicted_future = (
            current_temperature
            + predicted_delta
        )

        error = (
            predicted_future
            - actual_future
        )

        return float(
            np.mean(
                error ** 2
            )
        )

    starting_weights = (
        np.ones(
            len(
                model_columns
            )
        )
        / len(
            model_columns
        )
    )

    result = minimize(
        objective,

        starting_weights,

        method="SLSQP",

        bounds=[
            (
                0.05,
                0.80,
            )
            for _
            in model_columns
        ],

        constraints=[
            {
                "type":
                    "eq",

                "fun":
                    lambda weights:
                    np.sum(
                        weights
                    )
                    - 1.0,
            }
        ],

        options={
            "maxiter":
                500,

            "ftol":
                1e-10,
        },
    )

    if not result.success:

        weights = (
            starting_weights
        )

    else:

        weights = result.x

    weights = np.maximum(
        weights,
        0.05,
    )

    weights /= np.sum(
        weights
    )

    return {
        model:
            float(
                weight
            )
        for model, weight
        in zip(
            model_columns,
            weights,
        )
    }


# ============================================================
# COMPARISON
# ============================================================


def build_metrics_table(
    dataframe: pd.DataFrame,
    include_persistence: bool = True,
) -> pd.DataFrame:

    rows = []

    actual = (
        dataframe[
            "target_future_temperature_c"
        ]
        .to_numpy(
            dtype=float
        )
    )

    current = (
        dataframe[
            "temperature_c"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if include_persistence:

        persistence = (
            current
        )

        persistence_metrics = (
            regression_metrics(
                actual,
                persistence,
            )
        )

        rows.append(
            {
                "Model":
                    "Persistence Baseline",

                **persistence_metrics,
            }
        )

    for model in MODEL_NAMES:

        prediction = (
            current
            + dataframe[
                model
            ].to_numpy(
                dtype=float
            )
        )

        metrics = (
            regression_metrics(
                actual,
                prediction,
            )
        )

        rows.append(
            {
                "Model":
                    model,

                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================


def train_ensemble(
    dataframe: pd.DataFrame,
    horizon_s: float = 30.0,
    seed: int = 42,
) -> tuple[
    Path,
    pd.DataFrame,
]:
    """Train, validate and save the four-model ensemble."""

    set_seed(
        seed
    )

    dataset = (
        create_prediction_dataset(
            dataframe,
            horizon_s=horizon_s,
        )
    )

    (
        train,
        validation,
        test,
    ) = split_by_scenario(
        dataset,
        seed=seed,
    )

    print()
    print(
        "THERMO-X FOUR-MODEL ENSEMBLE"
    )
    print(
        "============================"
    )

    print(
        f"Train rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    rf = RandomForestRegressor(
        n_estimators=250,

        max_depth=14,

        min_samples_leaf=2,

        max_features="sqrt",

        random_state=seed,

        n_jobs=-1,
    )

    rf.fit(
        train[
            FEATURE_COLUMNS
        ],
        train[
            "target_delta_temperature_c"
        ],
    )

    rf_val = rf.predict(
        validation[
            FEATURE_COLUMNS
        ]
    )

    rf_test = rf.predict(
        test[
            FEATURE_COLUMNS
        ]
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    print(
        "Training XGBoost..."
    )

    xgb = XGBRegressor(
        n_estimators=350,

        learning_rate=0.035,

        max_depth=6,

        min_child_weight=2,

        subsample=0.88,

        colsample_bytree=0.88,

        reg_lambda=1.5,

        objective="reg:squarederror",

        eval_metric="rmse",

        random_state=seed,

        n_jobs=-1,

        tree_method="hist",
    )

    xgb.fit(
        train[
            FEATURE_COLUMNS
        ],
        train[
            "target_delta_temperature_c"
        ],
    )

    xgb_val = xgb.predict(
        validation[
            FEATURE_COLUMNS
        ]
    )

    xgb_test = xgb.predict(
        test[
            FEATURE_COLUMNS
        ]
    )

    # --------------------------------------------------------
    # Deep MLP
    # --------------------------------------------------------

    print(
        "Training Deep MLP..."
    )

    (
        mlp,
        mlp_scaler_x,
        mlp_scaler_y,
        mlp_val,
        mlp_epochs,
    ) = train_mlp(
        train,
        validation,
        seed,
    )

    X_test_mlp = (
        mlp_scaler_x.transform(
            test[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )
        .astype(
            np.float32
        )
    )

    mlp.eval()

    with torch.no_grad():

        mlp_test_scaled = (
            mlp(
                torch.tensor(
                    X_test_mlp,
                    dtype=torch.float32,
                    device=DEVICE,
                )
            )
            .cpu()
            .numpy()
        )

    mlp_test = (
        mlp_scaler_y
        .inverse_transform(
            mlp_test_scaled
        )
        .reshape(
            -1
        )
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    print(
        "Training LSTM..."
    )

    (
        lstm,
        lstm_scaler_x,
        lstm_scaler_y,
        lstm_val_predictions,
        lstm_epochs,
    ) = train_lstm(
        train,
        validation,
        seed,
    )

    # Test LSTM.
    lstm_test_predictions = (
        predict_lstm(
            lstm,
            lstm_scaler_x,
            lstm_scaler_y,
            test,
        )
    )

    # --------------------------------------------------------
    # Assemble validation predictions.
    # --------------------------------------------------------

    validation_predictions = (
        validation[
            [
                "scenario_id",
                "server_id",
                "timestamp_s",
                "temperature_c",
                "target_future_temperature_c",
            ]
        ]
        .copy()
    )

    validation_predictions[
        "Random Forest"
    ] = rf_val

    validation_predictions[
        "XGBoost"
    ] = xgb_val

    validation_predictions[
        "Deep MLP"
    ] = mlp_val

    validation_predictions = (
        validation_predictions
        .merge(
            lstm_val_predictions[
                [
                    "scenario_id",
                    "server_id",
                    "timestamp_s",
                    "predicted_delta",
                ]
            ].rename(
                columns={
                    "predicted_delta":
                        "LSTM"
                }
            ),
            on=[
                "scenario_id",
                "server_id",
                "timestamp_s",
            ],
            how="inner",
        )
    )

    # --------------------------------------------------------
    # Learn ensemble weights.
    # --------------------------------------------------------

    weights = (
        optimize_ensemble_weights(
            validation_predictions
        )
    )

    print()
    print(
        "Learned ensemble weights:"
    )

    for model_name, weight in (
        weights.items()
    ):

        print(
            f"  {model_name:15s}: "
            f"{weight * 100:6.2f}%"
        )

    # --------------------------------------------------------
    # Validation ensemble.
    # --------------------------------------------------------

    weight_vector = np.array(
        [
            weights[
                model_name
            ]
            for model_name
            in MODEL_NAMES
        ]
    )

    validation_matrix = (
        validation_predictions[
            MODEL_NAMES
        ]
        .to_numpy(
            dtype=float
        )
    )

    validation_predictions[
        "Ensemble"
    ] = (
        validation_matrix
        @ weight_vector
    )

    validation_metrics = (
        build_metrics_table(
            validation_predictions[
                [
                    "temperature_c",
                    "target_future_temperature_c",
                    *MODEL_NAMES,
                ]
            ]
        )
    )

    ensemble_val_future = (
        validation_predictions[
            "temperature_c"
        ].to_numpy()
        + validation_predictions[
            "Ensemble"
        ].to_numpy()
    )

    validation_ensemble_metrics = (
        regression_metrics(
            validation_predictions[
                "target_future_temperature_c"
            ].to_numpy(),
            ensemble_val_future,
        )
    )

    print()
    print(
        "Validation ensemble:"
    )

    print(
        validation_ensemble_metrics
    )

    # --------------------------------------------------------
    # Test predictions.
    #
    # LSTM only exists after sufficient history.
    # --------------------------------------------------------

    test_predictions = (
        test[
            [
                "scenario_id",
                "server_id",
                "timestamp_s",
                "temperature_c",
                "target_future_temperature_c",
            ]
        ]
        .copy()
    )

    test_predictions[
        "Random Forest"
    ] = rf_test

    test_predictions[
        "XGBoost"
    ] = xgb_test

    test_predictions[
        "Deep MLP"
    ] = mlp_test

    test_predictions = (
        test_predictions
        .merge(
            lstm_test_predictions[
                [
                    "scenario_id",
                    "server_id",
                    "timestamp_s",
                    "predicted_delta",
                ]
            ].rename(
                columns={
                    "predicted_delta":
                        "LSTM"
                }
            ),
            on=[
                "scenario_id",
                "server_id",
                "timestamp_s",
            ],
            how="inner",
        )
    )

    test_matrix = (
        test_predictions[
            MODEL_NAMES
        ]
        .to_numpy(
            dtype=float
        )
    )

    test_predictions[
        "Ensemble"
    ] = (
        test_matrix
        @ weight_vector
    )

    # --------------------------------------------------------
    # Calculate future temperatures.
    # --------------------------------------------------------

    for model_name in MODEL_NAMES:

        test_predictions[
            f"{model_name}_future_temperature_c"
        ] = (
            test_predictions[
                "temperature_c"
            ]
            + test_predictions[
                model_name
            ]
        )

    test_predictions[
        "ensemble_future_temperature_c"
    ] = (
        test_predictions[
            "temperature_c"
        ]
        + test_predictions[
            "Ensemble"
        ]
    )

    test_predictions[
        "ensemble_spread_c"
    ] = (
        test_predictions[
            [
                (
                    f"{model}_future_temperature_c"
                )
                for model in MODEL_NAMES
            ]
        ]
        .std(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Test metrics.
    # --------------------------------------------------------

    actual_test = (
        test_predictions[
            "target_future_temperature_c"
        ].to_numpy(
            dtype=float
        )
    )

    test_metrics_rows = []

    # Persistence.
    persistence_metrics = (
        regression_metrics(
            actual_test,
            test_predictions[
                "temperature_c"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    test_metrics_rows.append(
        {
            "Model":
                "Persistence Baseline",

            **persistence_metrics,
        }
    )

    for model_name in MODEL_NAMES:

        metrics = (
            regression_metrics(
                actual_test,
                test_predictions[
                    f"{model_name}_future_temperature_c"
                ].to_numpy(
                    dtype=float
                ),
            )
        )

        test_metrics_rows.append(
            {
                "Model":
                    model_name,

                **metrics,
            }
        )

    ensemble_metrics = (
        regression_metrics(
            actual_test,
            test_predictions[
                "ensemble_future_temperature_c"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    test_metrics_rows.append(
        {
            "Model":
                "Four-Model Ensemble",

            **ensemble_metrics,
        }
    )

    test_metrics = pd.DataFrame(
        test_metrics_rows
    )

    # --------------------------------------------------------
    # Save model bundle.
    # --------------------------------------------------------

    model_path = Path(
        "models/"
        "thermo_x_ensemble.joblib"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = {

        "model_version":
            "THERMO-X-ENSEMBLE-v1",

        "horizon_s":
            horizon_s,

        "feature_columns":
            FEATURE_COLUMNS,

        "sequence_length":
            SEQUENCE_LENGTH,

        "model_names":
            MODEL_NAMES,

        "weights":
            weights,

        "random_forest":
            rf,

        "xgboost":
            xgb,

        "mlp_state_dict":
            {
                key:
                    value.cpu()
                for key, value
                in mlp.state_dict().items()
            },

        "mlp_scaler_x":
            mlp_scaler_x,

        "mlp_scaler_y":
            mlp_scaler_y,

        "lstm_state_dict":
            {
                key:
                    value.cpu()
                for key, value
                in lstm.state_dict().items()
            },

        "lstm_scaler_x":
            lstm_scaler_x,

        "lstm_scaler_y":
            lstm_scaler_y,

        "input_size":
            len(
                FEATURE_COLUMNS
            ),

        "metrics":
            {
                "validation":
                    validation_ensemble_metrics,

                "test":
                    ensemble_metrics,

                "test_table":
                    test_metrics
                    .to_dict(
                        orient="records"
                    ),
            },

        "neural_epochs":
            {
                "Deep MLP":
                    mlp_epochs,

                "LSTM":
                    lstm_epochs,
            },
    }

    joblib.dump(
        bundle,
        model_path,
    )

    # --------------------------------------------------------
    # Save metrics.
    # --------------------------------------------------------

    metrics_path = Path(
        "models/"
        "thermo_x_ensemble_metrics.csv"
    )

    test_metrics.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save test predictions.
    # --------------------------------------------------------

    prediction_path = Path(
        "data/experiments/"
        "ensemble_test_predictions.csv"
    )

    prediction_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_predictions.to_csv(
        prediction_path,
        index=False,
    )

    metadata_path = Path(
        "models/"
        "thermo_x_ensemble_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "model_version":
                    bundle[
                        "model_version"
                    ],

                "horizon_s":
                    horizon_s,

                "weights":
                    weights,

                "validation":
                    validation_ensemble_metrics,

                "test":
                    ensemble_metrics,

                "test_rows":
                    len(
                        test_predictions
                    ),
            },
            file,
            indent=2,
        )

    print()
    print(
        "=========================================="
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=========================================="
    )

    print(
        test_metrics.to_string(
            index=False
        )
    )

    print()

    print(
        f"Ensemble model saved: "
        f"{model_path}"
    )

    print(
        f"Test predictions saved: "
        f"{prediction_path}"
    )

    return (
        model_path,
        test_metrics,
    )