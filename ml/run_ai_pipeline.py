"""
One-command THERMO-X AI pipeline.

Pipeline:

    1. Generate diverse thermal scenarios
    2. Build training dataset
    3. Train four models
    4. Learn ensemble weights
    5. Evaluate on unseen scenarios
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.scenario_generator import (
    generate_training_dataset,
)

from .ensemble import (
    train_ensemble,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete THERMO-X AI pipeline."
        )
    )

    parser.add_argument(
        "--scenarios",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=240.0,
    )

    parser.add_argument(
        "--horizon",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    dataset_path = (
        Path(
            "data/processed/"
            "thermo_x_training_dataset.csv"
        )
    )

    print()
    print(
        "=========================================="
    )

    print(
        "       THERMO-X FULL AI PIPELINE"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Step 1 — Synthetic thermal dataset.
    # --------------------------------------------------------

    generate_training_dataset(
        scenario_count=args.scenarios,
        duration_s=args.duration,
        master_seed=args.seed,
        output_path=dataset_path,
    )

    # --------------------------------------------------------
    # Step 2 — Train ensemble.
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        dataset_path
    )

    train_ensemble(
        dataframe=dataframe,
        horizon_s=args.horizon,
        seed=args.seed,
    )

    print()
    print(
        "=========================================="
    )

    print(
        "THERMO-X AI PIPELINE COMPLETE"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "Training dataset:"
    )

    print(
        f"  {dataset_path}"
    )

    print()

    print(
        "Final ensemble:"
    )

    print(
        "  models/thermo_x_ensemble.joblib"
    )

    print()

    print(
        "Prediction horizon:"
    )

    print(
        f"  {args.horizon:.0f} seconds"
    )


if __name__ == "__main__":
    main()