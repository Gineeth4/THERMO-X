"""
Configuration for reproducible THERMO-X experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Settings for the first fixed-vs-adaptive experiment.

    These are SOFTWARE EXPERIMENT parameters.

    They are not physical prototype component specifications.
    """

    # -----------------------------
    # Thermal zones
    # -----------------------------

    server_ids: tuple[str, ...] = (
        "SERVER_A",
        "SERVER_B",
    )

    # -----------------------------
    # Time
    # -----------------------------

    duration_s: float = 240.0

    dt_s: float = 1.0

    # -----------------------------
    # Fixed cooling baseline
    # -----------------------------

    fixed_pump_speed_pct: float = 60.0

    fixed_fan_speed_pct: float = 60.0

    fixed_valve_position_pct: float = 50.0

    # -----------------------------
    # Reproducibility
    # -----------------------------

    sensor_seed: int = 42

    # -----------------------------
    # Output directory
    # -----------------------------

    output_directory: Path = Path(
        "data/experiments"
    )

    @property
    def fixed_csv_path(self) -> Path:
        """CSV produced by the fixed-cooling experiment."""

        return (
            self.output_directory
            / "fixed_cooling.csv"
        )

    @property
    def adaptive_csv_path(self) -> Path:
        """CSV produced by the adaptive experiment."""

        return (
            self.output_directory
            / "adaptive_cooling.csv"
        )

    @property
    def comparison_csv_path(self) -> Path:
        """CSV containing fixed-vs-adaptive metrics."""

        return (
            self.output_directory
            / "fixed_vs_adaptive_comparison.csv"
        )

    @property
    def plot_directory(self) -> Path:
        """Directory for experiment plots."""

        return (
            self.output_directory
            / "plots"
        )