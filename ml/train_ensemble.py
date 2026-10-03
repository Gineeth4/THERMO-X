"""
Command-line interface for THERMO-X ensemble training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .ensemble import (
    train_ensemble,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train the THERMO-X four-model ensemble."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/"
            "thermo_x_training_dataset.csv"
        ),
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

    if not args.input.exists():

        raise FileNotFoundError(
            f"Dataset not found: {args.input}"
        )

    dataframe = pd.read_csv(
        args.input
    )

    train_ensemble(
        dataframe=dataframe,
        horizon_s=args.horizon,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()