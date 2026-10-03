"""
Tests for the THERMO-X core layer.

These tests verify:

    - thermal calculations
    - sensor data models
    - safety logic
    - adaptive controller behaviour
"""

from core.controller import (
    AdaptiveController,
)

from core.models import (
    ActuatorCommand,
    CoolingMode,
    SensorSnapshot,
    SystemState,
)

from core.safety import (
    SafetyController,
    apply_safety_override,
)

from core.thermal_calculations import (
    electrical_power_w,
    heat_transfer_rate_w,
    mass_flow_rate_kg_s,
)


def make_snapshot(
    *,
    timestamp_s: float = 0.0,
    server_id: str = "SERVER_A",
    workload_pct: float = 50.0,
    heater_power_w: float = 50.0,
    temperature_c: float = 40.0,
    flow_rate_lpm: float = 1.0,
    pressure_kpa: float = 80.0,
    ambient_temperature_c: float = 30.0,
    leak_detected: bool = False,
    sensors_valid: bool = True,
) -> SensorSnapshot:
    """
    Create a valid SensorSnapshot for testing.

    Keyword-only parameters make each test explicit about the
    condition it is testing.
    """

    return SensorSnapshot(
        timestamp_s=timestamp_s,

        server_id=server_id,

        workload_pct=workload_pct,

        heater_power_w=heater_power_w,

        temperature_c=temperature_c,

        coolant_inlet_c=30.0,

        coolant_outlet_c=35.0,

        flow_rate_lpm=flow_rate_lpm,

        pressure_kpa=pressure_kpa,

        ambient_temperature_c=ambient_temperature_c,

        leak_detected=leak_detected,

        sensors_valid=sensors_valid,
    )


# ============================================================
# THERMAL CALCULATIONS
# ============================================================


def test_mass_flow_conversion() -> None:
    """Verify L/min → kg/s conversion."""

    result = mass_flow_rate_kg_s(
        flow_lpm=1.0,
        density_kg_m3=1000.0,
    )

    expected = (
        1.0
        / 1000.0
        / 60.0
        * 1000.0
    )

    assert abs(
        result - expected
    ) < 1e-12


def test_heat_transfer_is_positive() -> None:
    """Positive coolant flow and ΔT should produce positive heat transfer."""

    heat = heat_transfer_rate_w(
        flow_lpm=1.0,
        density_kg_m3=1000.0,
        specific_heat_j_kg_k=4180.0,
        delta_t_c=5.0,
    )

    assert heat > 0.0


def test_electrical_power_calculation() -> None:
    """Verify P = V × I."""

    power = electrical_power_w(
        voltage_v=12.0,
        current_a=2.0,
    )

    assert power == 24.0


# ============================================================
# DATA MODEL
# ============================================================


def test_sensor_snapshot_delta_temperature() -> None:
    """Verify coolant ΔT property."""

    snapshot = make_snapshot()

    assert (
        snapshot.coolant_delta_t_c
        == 5.0
    )


def test_actuator_command_initializes_valves() -> None:
    """Verify actuator commands always have a valve dictionary."""

    command = ActuatorCommand()

    assert (
        command.valve_positions_pct
        == {}
    )


# ============================================================
# SAFETY
# ============================================================


def test_normal_system_is_safe() -> None:
    """Normal conditions should result in SAFE."""

    controller = (
        SafetyController()
    )

    decision = controller.evaluate(
        [
            make_snapshot(
                temperature_c=40.0,
                flow_rate_lpm=1.0,
            )
        ]
    )

    assert (
        decision.state
        == SystemState.SAFE
    )

    assert (
        decision.disable_heater
        is False
    )


def test_warning_temperature_is_detected() -> None:
    """Temperature above normal range should generate WARNING."""

    controller = (
        SafetyController()
    )

    decision = controller.evaluate(
        [
            make_snapshot(
                temperature_c=72.0,
                flow_rate_lpm=1.0,
            )
        ]
    )

    assert (
        decision.state
        == SystemState.WARNING
    )

    assert (
        decision.force_max_cooling
        is True
    )


def test_critical_temperature_disables_heater() -> None:
    """Critical temperature should force an emergency response."""

    controller = (
        SafetyController()
    )

    decision = controller.evaluate(
        [
            make_snapshot(
                temperature_c=85.0,
                flow_rate_lpm=1.0,
            )
        ]
    )

    assert (
        decision.state
        == SystemState.EMERGENCY
    )

    assert (
        decision.disable_heater
        is True
    )

    assert (
        decision.force_max_cooling
        is True
    )


def test_low_flow_triggers_safety() -> None:
    """Insufficient cooling flow with active heat should be unsafe."""

    controller = (
        SafetyController()
    )

    decision = controller.evaluate(
        [
            make_snapshot(
                temperature_c=45.0,
                flow_rate_lpm=0.05,
                heater_power_w=50.0,
            )
        ]
    )

    assert (
        decision.state
        == SystemState.EMERGENCY
    )


def test_leak_triggers_emergency() -> None:
    """A detected leak must produce emergency state."""

    controller = (
        SafetyController()
    )

    decision = controller.evaluate(
        [
            make_snapshot(
                leak_detected=True,
                temperature_c=40.0,
                flow_rate_lpm=1.0,
            )
        ]
    )

    assert (
        decision.state
        == SystemState.EMERGENCY
    )

    assert (
        decision.disable_heater
        is True
    )

    assert (
        decision.disable_pump
        is True
    )


def test_invalid_sensor_triggers_fault() -> None:
    """Invalid sensor data must fail safely."""

    controller = (
        SafetyController()
    )

    decision = controller.evaluate(
        [
            make_snapshot(
                sensors_valid=False,
            )
        ]
    )

    assert (
        decision.state
        == SystemState.FAULT
    )

    assert (
        decision.disable_heater
        is True
    )


def test_safety_override_changes_command() -> None:
    """Safety override must modify a dangerous actuator command."""

    command = ActuatorCommand(
        pump_speed_pct=40.0,
        fan_speed_pct=40.0,
        heater_enabled=True,
        cooling_mode=(
            CoolingMode.ADAPTIVE
        ),
    )

    controller = (
        SafetyController()
    )

    decision = controller.evaluate(
        [
            make_snapshot(
                temperature_c=85.0,
                flow_rate_lpm=1.0,
            )
        ]
    )

    modified = apply_safety_override(
        command,
        decision,
    )

    assert (
        modified.heater_enabled
        is False
    )

    assert (
        modified.pump_speed_pct
        == 100.0
    )

    assert (
        modified.fan_speed_pct
        == 100.0
    )


# ============================================================
# ADAPTIVE CONTROL
# ============================================================


def test_adaptive_controller_returns_command() -> None:
    """Adaptive controller should generate a valid cooling command."""

    controller = (
        AdaptiveController()
    )

    command, metrics = (
        controller.control_step(
            [
                make_snapshot(
                    temperature_c=50.0,
                    flow_rate_lpm=1.0,
                )
            ]
        )
    )

    assert (
        command.cooling_mode
        == CoolingMode.ADAPTIVE
    )

    assert (
        0.0
        <= command.pump_speed_pct
        <= 100.0
    )

    assert (
        0.0
        <= command.fan_speed_pct
        <= 100.0
    )

    assert len(
        metrics
    ) == 1


def test_adaptive_controller_supports_multiple_zones() -> None:
    """Controller should process multiple thermal modules."""

    controller = (
        AdaptiveController()
    )

    snapshots = [
        make_snapshot(
            server_id="SERVER_A",
            temperature_c=45.0,
        ),

        make_snapshot(
            server_id="SERVER_B",
            temperature_c=55.0,
        ),
    ]

    command, metrics = (
        controller.control_step(
            snapshots
        )
    )

    assert len(
        metrics
    ) == 2

    assert set(
        command.valve_positions_pct.keys()
    ) == {
        "SERVER_A",
        "SERVER_B",
    }