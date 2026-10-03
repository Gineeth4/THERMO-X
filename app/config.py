"""
Central configuration for THERMO-X.

Keep project-level settings here rather than scattering them
throughout the application.

Important:
The numerical values in this file are simulation settings.
They are not physical hardware specifications.
"""

from __future__ import annotations

from pathlib import Path


# ------------------------------------------------------------
# Project directories
# ------------------------------------------------------------

# D:\Projects\THERMO-X
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXPERIMENT_DATA_DIR = DATA_DIR / "experiments"

MODEL_DIR = PROJECT_ROOT / "models"


# ------------------------------------------------------------
# Initial simulation configuration
# ------------------------------------------------------------

SERVER_IDS = (
    "SERVER_A",
    "SERVER_B",
)

DEFAULT_SIMULATION_DURATION_S = 240.0

DEFAULT_SIMULATION_DT_S = 1.0

DEFAULT_SENSOR_SEED = 42


# ------------------------------------------------------------
# Initial simulated actuator power assumptions
# ------------------------------------------------------------

# These values currently come from the initial software model.
#
# When the physical system is built, measured electrical power
# will replace these estimates.

PUMP_RATED_POWER_W = 30.0

FAN_RATED_POWER_W = 20.0


# ------------------------------------------------------------
# Dashboard configuration
# ------------------------------------------------------------

DASHBOARD_TITLE = (
    "THERMO-X — Adaptive Thermal Management"
)

DASHBOARD_ICON = "❄️"


# ------------------------------------------------------------
# Ensure important directories exist
# ------------------------------------------------------------

for directory in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXPERIMENT_DATA_DIR,
    MODEL_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )