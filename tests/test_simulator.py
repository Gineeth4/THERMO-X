"""
Tests for the THERMO-X virtual plant.

These tests verify that:

    workload
        ↓
    thermal model
        ↓
    virtual sensors
        ↓
    controller
        ↓
    virtual actuators

can operate together.
"""

from math import isfinite

from core.controller import (
    AdaptiveController,
)

from simulator.thermal_system import (
    VirtualThermoXSystem,
)

from simulator.virtual_actuators import (
    VirtualActuatorBank,
)

from simulator.virtual_sensors import (
    VirtualSensorBank,
)

from simulator.workload_generator import (
    create_demo_profile,
)

from thermal_model.model import (
    ThermalSystemModel,
)


def build_test_system() -> (
    VirtualThermoXSystem
):
    """Build a fresh two-zone virtual THERMO-X system."""

    server_ids = [
        "SERVER_A",
        "SERVER_B",
    ]

    model = ThermalSystemModel(
        server_ids=server_ids
    )

    workload = (
        create_demo_profile(
            server_ids
        )
    )

    sensors = (
        VirtualSensorBank(
            model=model,
            seed=42,
        )
    )

    actuators = (
        VirtualActuatorBank()
    )

    controller = (
        AdaptiveController()
    )

    return VirtualThermoXSystem(
        server_ids=server_ids,
        workload_profile=workload,
        model=model,
        controller=controller,
        sensors=sensors,
        actuators=actuators,
    )


# ============================================================
# BASIC OPERATION
# ============================================================


def test_virtual_system_returns_sensor_data() -> None:
    """One simulation step should return both server measurements."""

    system = (
        build_test_system()
    )

    snapshots, command = (
        system.step(
            timestamp_s=0.0,
            dt_s=1.0,
        )
    )

    assert len(
        snapshots
    ) == 2

    assert (
        command.pump_speed_pct
        >= 0.0
    )

    assert (
        command.fan_speed_pct
        >= 0.0
    )


def test_virtual_system_returns_expected_server_ids() -> None:
    """Both configured thermal zones should be present."""

    system = (
        build_test_system()
    )

    snapshots, _ = (
        system.step(
            timestamp_s=0.0,
            dt_s=1.0,
        )
    )

    server_ids = {
        snapshot.server_id
        for snapshot in snapshots
    }

    assert server_ids == {
        "SERVER_A",
        "SERVER_B",
    }


def test_virtual_system_values_are_finite() -> None:
    """Thermal telemetry must remain finite."""

    system = (
        build_test_system()
    )

    for second in range(
        10
    ):

        snapshots, _ = (
            system.step(
                timestamp_s=float(
                    second
                ),
                dt_s=1.0,
            )
        )

        for snapshot in snapshots:

            assert isfinite(
                snapshot.temperature_c
            )

            assert isfinite(
                snapshot.flow_rate_lpm
            )

            assert isfinite(
                snapshot.pressure_kpa
            )

            assert isfinite(
                snapshot.coolant_inlet_c
            )

            assert isfinite(
                snapshot.coolant_outlet_c
            )


def test_temperature_changes_under_load() -> None:
    """
    The thermal model should not remain perfectly static
    when thermal load is applied.
    """

    system = (
        build_test_system()
    )

    first_snapshots, _ = (
        system.step(
            timestamp_s=0.0,
            dt_s=1.0,
        )
    )

    first_temperature = (
        first_snapshots[0]
        .temperature_c
    )

    latest_temperature = (
        first_temperature
    )

    for second in range(
        1,
        30,
    ):

        snapshots, _ = (
            system.step(
                timestamp_s=float(
                    second
                ),
                dt_s=1.0,
            )
        )

        latest_temperature = (
            snapshots[0]
            .temperature_c
        )

    assert (
        abs(
            latest_temperature
            - first_temperature
        )
        > 1e-6
    )


# ============================================================
# ACTUATOR LIMITS
# ============================================================


def test_virtual_actuators_clamp_values() -> None:
    """Virtual actuator commands must remain inside 0–100%."""

    from core.models import (
        ActuatorCommand,
    )

    actuators = (
        VirtualActuatorBank()
    )

    command = ActuatorCommand(
        pump_speed_pct=150.0,
        fan_speed_pct=-20.0,
        valve_positions_pct={
            "SERVER_A": 120.0,
            "SERVER_B": -10.0,
        },
    )

    actuators.apply(
        command
    )

    assert (
        actuators.command
        .pump_speed_pct
        == 100.0
    )

    assert (
        actuators.command
        .fan_speed_pct
        == 0.0
    )

    assert (
        actuators.command
        .valve_positions_pct[
            "SERVER_A"
        ]
        == 100.0
    )

    assert (
        actuators.command
        .valve_positions_pct[
            "SERVER_B"
        ]
        == 0.0
    )