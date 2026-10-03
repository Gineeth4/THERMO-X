"""
THERMO-X Main Dashboard.

Launch:

    python -m streamlit run dashboard/dashboard.py

Current operating mode:
    Software simulation

Future operating modes:
    ESP32 / USB Serial / MQTT
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from app.main import (
    load_experiment_csv,
    run_interactive_scenario,
    run_live_simulation,
)

from ml.predict_ensemble import (
    load_ensemble_bundle,
    predict_dataframe,
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# THERMO-X IMPORTS
# ============================================================

from analytics.comparison import (
    compare_runs,
)

from analytics.metrics import (
    calculate_run_metrics,
)

from app.config import (
    DEFAULT_SENSOR_SEED,
    DEFAULT_SIMULATION_DURATION_S,
    DEFAULT_SIMULATION_DT_S,
    EXPERIMENT_DATA_DIR,
    FAN_RATED_POWER_W,
    MODEL_DIR,
    PUMP_RATED_POWER_W,
)

from app.main import (
    load_experiment_csv,
    run_live_simulation,
)

from components import (
    comparison_figure,
    cooling_control_figure,
    coolant_delta_figure,
    inject_dashboard_css,
    latest_rows,
    render_application_bar,
    render_download_button,
    render_footer,
    render_server_cards,
    render_status_chip,
    render_system_kpis,
    temperature_figure,
    workload_figure,
)

from experiments.experiment_config import (
    ExperimentConfig,
)

from experiments.experiment_runner import (
    run_experiment,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=(
        "THERMO-X — Thermal Control Centre"
    ),

    page_icon="🌡️",

    layout="wide",

    initial_sidebar_state="expanded",
)


inject_dashboard_css()


# ============================================================
# SESSION STATE
# ============================================================

if (
    "live_dataframe"
    not in st.session_state
):

    st.session_state[
        "live_dataframe"
    ] = pd.DataFrame()


if (
    "validation_message"
    not in st.session_state
):

    st.session_state[
        "validation_message"
    ] = ""


# ============================================================
# APPLICATION BAR
# ============================================================

render_application_bar(
    "SIMULATION ONLINE"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## THERMO-X"
    )

    st.caption(
        "Adaptive thermal-management research platform"
    )

    st.divider()

    st.markdown(
        "### Simulation Lab"
    )

    simulation_duration = st.slider(
        "Duration",
        min_value=30,
        max_value=600,
        value=int(
            DEFAULT_SIMULATION_DURATION_S
        ),
        step=10,
        help=(
            "Total simulated thermal-system time."
        ),
    )

    simulation_speed = st.slider(
        "Simulation speed",
        min_value=1,
        max_value=50,
        value=15,
        step=1,
        help=(
            "1× is approximately real time."
        ),
    )

    sensor_seed = st.number_input(
        "Sensor seed",
        min_value=0,
        max_value=1_000_000,
        value=DEFAULT_SENSOR_SEED,
        step=1,
    )

    st.divider()

    st.markdown(
        "### Operating Mode"
    )

    st.info(
        "SIMULATION\n\n"
        "Virtual thermal modules are acting as "
        "the physical plant."
    )

    st.divider()

    st.markdown(
        "### Safety Hierarchy"
    )

    st.markdown(
        """
        **01 — Safety**

        Hard limits override optimization.

        **02 — Reliability**

        Maintain required cooling capability.

        **03 — Optimization**

        Adjust cooling to thermal demand.
        """
    )

    st.divider()

    st.caption(
        "Hardware interface"
    )

    st.code(
        "ESP32  →  Future physical interface",
        language="text",
    )


# ============================================================
# NAVIGATION
# ============================================================

(
    command_tab,
    validation_tab,
    prediction_tab,
    architecture_tab,
) = st.tabs(
    [
        "COMMAND CENTRE",
        "VALIDATION LAB",
        "PREDICTIVE CONTROL",
        "SYSTEM ARCHITECTURE",
    ]
)


# ============================================================
# COMMAND CENTRE
# ============================================================

with command_tab:

    st.markdown(
        "## Command Centre"
    )

    st.caption(
        "Monitor thermal demand and observe adaptive cooling "
        "response in real time."
    )

    # --------------------------------------------------------
    # Top KPI strip
    # --------------------------------------------------------

    live_dataframe = (
        st.session_state[
            "live_dataframe"
        ]
    )

    if live_dataframe.empty:

        columns = st.columns(
            4
        )

        with columns[0]:

            st.metric(
                "Thermal zones",
                "2",
            )

        with columns[1]:

            st.metric(
                "Control",
                "Adaptive",
            )

        with columns[2]:

            st.metric(
                "Plant",
                "Virtual",
            )

        with columns[3]:

            st.metric(
                "Prediction",
                "30 s",
            )

        st.info(
            "Start a simulation from the "
            "Run Simulation button below."
        )

    else:

        latest = latest_rows(
            live_dataframe
        )

        average_temperature = float(
            latest[
                "temperature_c"
            ].mean()
        )

        peak_temperature = float(
            live_dataframe[
                "temperature_c"
            ].max()
        )

        current_workload = float(
            latest[
                "workload_pct"
            ].mean()
        )

        pump = float(
            latest[
                "pump_speed_pct"
            ].iloc[0]
        )

        system_state = str(
            latest[
                "system_state"
            ].iloc[0]
        )

        columns = st.columns(
            5
        )

        with columns[0]:

            st.metric(
                "Average temperature",
                f"{average_temperature:.2f} °C",
            )

        with columns[1]:

            st.metric(
                "Peak temperature",
                f"{peak_temperature:.2f} °C",
            )

        with columns[2]:

            st.metric(
                "Current workload",
                f"{current_workload:.0f} %",
            )

        with columns[3]:

            st.metric(
                "Pump demand",
                f"{pump:.1f} %",
            )

        with columns[4]:

            st.markdown(
                "System state"
            )

            render_status_chip(
                system_state
            )

    st.divider()

    # --------------------------------------------------------
    # Simulation controls
    # --------------------------------------------------------

    control_column, info_column = (
        st.columns(
            [
                1,
                2,
            ]
        )
    )

    with control_column:

        st.markdown(
            "### Simulation Control"
        )

        run_simulation = st.button(
            "▶  RUN ADAPTIVE SIMULATION",
            type="primary",
            width="stretch",
        )

        clear_data = st.button(
            "Clear current run",
            width="stretch",
        )

    with info_column:

        st.markdown(
            "### Current Experiment"
        )

        st.markdown(
            """
            **Workload sequence**

            LOW → MEDIUM → HIGH → LOW

            **Thermal loop**

            Heat source → cold plate → coolant →
            radiator → ambient

            **Controller**

            Safety layer → adaptive thermal-demand control
            """
        )

    if clear_data:

        st.session_state[
            "live_dataframe"
        ] = pd.DataFrame()

        st.rerun()

    # --------------------------------------------------------
    # Run simulation
    # --------------------------------------------------------

    if run_simulation:

        st.markdown(
            "### Live Simulation"
        )

        progress = st.progress(
            0,
            text="Initializing THERMO-X..."
        )

        live_chart = st.empty()

        live_status = st.empty()

        def update_live(
            dataframe: pd.DataFrame,
        ) -> None:
            """
            Refresh the live interface.

            Only the most important chart and status section
            are refreshed during the simulation to keep the
            interface responsive.
            """

            if dataframe.empty:
                return

            current_time = float(
                dataframe[
                    "timestamp_s"
                ].max()
            )

            with live_chart.container():

                st.plotly_chart(
                    temperature_figure(
                        dataframe
                    ),
                    width="stretch",
                    key=(
                        f"live-temperature-"
                        f"{current_time:.2f}"
                    ),
                )

            latest_live = latest_rows(
                dataframe
            )

            avg_temp = float(
                latest_live[
                    "temperature_c"
                ].mean()
            )

            pump_value = float(
                latest_live[
                    "pump_speed_pct"
                ].iloc[0]
            )

            state = str(
                latest_live[
                    "system_state"
                ].iloc[0]
            )

            with live_status.container():

                status_cols = st.columns(
                    3
                )

                with status_cols[0]:

                    st.metric(
                        "Current temperature",
                        f"{avg_temp:.2f} °C",
                    )

                with status_cols[1]:

                    st.metric(
                        "Pump command",
                        f"{pump_value:.1f} %",
                    )

                with status_cols[2]:

                    st.markdown(
                        "Controller state"
                    )

                    render_status_chip(
                        state
                    )

            progress_value = min(
                current_time
                / float(
                    simulation_duration
                ),
                1.0,
            )

            progress.progress(
                progress_value,
                text=(
                    "Simulated time: "
                    f"{current_time:.0f} / "
                    f"{simulation_duration:.0f} s"
                ),
            )

        dataframe = run_live_simulation(
            duration_s=float(
                simulation_duration
            ),

            dt_s=DEFAULT_SIMULATION_DT_S,

            sensor_seed=int(
                sensor_seed
            ),

            realtime_speed=float(
                simulation_speed
            ),

            update_callback=update_live,
        )

        st.session_state[
            "live_dataframe"
        ] = dataframe

        progress.progress(
            1.0,
            text="Simulation complete.",
        )

        st.success(
            "THERMO-X simulation completed successfully."
        )

    # --------------------------------------------------------
    # Current live dataset
    # --------------------------------------------------------

    live_dataframe = (
        st.session_state[
            "live_dataframe"
        ]
    )

    if not live_dataframe.empty:

        st.divider()

        render_server_cards(
            live_dataframe
        )

        st.divider()

        render_system_kpis(
            live_dataframe
        )

        st.divider()

        st.markdown(
            "### Thermal & Control Telemetry"
        )

        chart_col_1, chart_col_2 = (
            st.columns(2)
        )

        with chart_col_1:

            st.plotly_chart(
                workload_figure(
                    live_dataframe
                ),
                width="stretch",
                key="live-workload",
            )

        with chart_col_2:

            st.plotly_chart(
                cooling_control_figure(
                    live_dataframe
                ),
                width="stretch",
                key="live-control",
            )

        st.plotly_chart(
            temperature_figure(
                live_dataframe
            ),
            width="stretch",
            key="live-temperature-final",
        )

        st.plotly_chart(
            coolant_delta_figure(
                live_dataframe
            ),
            width="stretch",
            key="live-delta",
        )

        with st.expander(
            "Inspect telemetry dataset"
        ):

            st.dataframe(
                live_dataframe.tail(
                    150
                ),
                width="stretch",
                hide_index=True,
            )

        render_download_button(
            live_dataframe,
            "thermo_x_live_simulation.csv",
            "Download Telemetry CSV",
        )


# ============================================================
# VALIDATION LAB
# ============================================================

with validation_tab:

    st.markdown(
        "## Validation Lab"
    )

    st.caption(
        "Controlled comparison of fixed cooling and "
        "THERMO-X adaptive cooling."
    )

    run_validation = st.button(
        "Run Fixed vs Adaptive Validation",
        type="primary",
        width="stretch",
    )

    if run_validation:

        validation_config = ExperimentConfig(
            server_ids=(
                "SERVER_A",
                "SERVER_B",
            ),

            duration_s=float(
                simulation_duration
            ),

            dt_s=1.0,

            sensor_seed=int(
                sensor_seed
            ),
        )

        with st.spinner(
            "Running controlled validation experiment..."
        ):

            run_experiment(
                validation_config
            )

        st.success(
            "Validation experiment completed. "
            "Results are now available below."
        )

    fixed_csv = (
        EXPERIMENT_DATA_DIR
        / "fixed_cooling.csv"
    )

    adaptive_csv = (
        EXPERIMENT_DATA_DIR
        / "adaptive_cooling.csv"
    )

    if (
        fixed_csv.exists()
        and adaptive_csv.exists()
    ):

        try:

            fixed_dataframe = (
                load_experiment_csv(
                    fixed_csv
                )
            )

            adaptive_dataframe = (
                load_experiment_csv(
                    adaptive_csv
                )
            )

            comparison = compare_runs(
                fixed_dataframe,
                adaptive_dataframe,
                pump_rated_power_w=(
                    PUMP_RATED_POWER_W
                ),
                fan_rated_power_w=(
                    FAN_RATED_POWER_W
                ),
            )

            fixed_metrics = (
                calculate_run_metrics(
                    fixed_dataframe,
                    pump_rated_power_w=(
                        PUMP_RATED_POWER_W
                    ),
                    fan_rated_power_w=(
                        FAN_RATED_POWER_W
                    ),
                ).iloc[0]
            )

            adaptive_metrics = (
                calculate_run_metrics(
                    adaptive_dataframe,
                    pump_rated_power_w=(
                        PUMP_RATED_POWER_W
                    ),
                    fan_rated_power_w=(
                        FAN_RATED_POWER_W
                    ),
                ).iloc[0]
            )

            st.divider()

            st.markdown(
                "### Thermal Comparison"
            )

            st.plotly_chart(
                comparison_figure(
                    fixed_dataframe,
                    adaptive_dataframe,
                ),
                width="stretch",
                key="validation-comparison",
            )

            st.markdown(
                "### Engineering Metrics"
            )

            columns = st.columns(
                4
            )

            with columns[0]:

                st.metric(
                    "Fixed peak",
                    (
                        f"{fixed_metrics['maximum_temperature_c']:.2f}"
                        " °C"
                    ),
                )

                st.metric(
                    "Adaptive peak",
                    (
                        f"{adaptive_metrics['maximum_temperature_c']:.2f}"
                        " °C"
                    ),
                )

            with columns[1]:

                st.metric(
                    "Fixed average",
                    (
                        f"{fixed_metrics['average_temperature_c']:.2f}"
                        " °C"
                    ),
                )

                st.metric(
                    "Adaptive average",
                    (
                        f"{adaptive_metrics['average_temperature_c']:.2f}"
                        " °C"
                    ),
                )

            with columns[2]:

                st.metric(
                    "Fixed cooling energy",
                    (
                        f"{fixed_metrics['cooling_energy_wh']:.3f}"
                        " Wh"
                    ),
                )

                st.metric(
                    "Adaptive cooling energy",
                    (
                        f"{adaptive_metrics['cooling_energy_wh']:.3f}"
                        " Wh"
                    ),
                )

            with columns[3]:

                st.metric(
                    "Fixed average flow",
                    (
                        f"{fixed_metrics['average_flow_lpm']:.3f}"
                        " L/min"
                    ),
                )

                st.metric(
                    "Adaptive average flow",
                    (
                        f"{adaptive_metrics['average_flow_lpm']:.3f}"
                        " L/min"
                    ),
                )

            st.divider()

            st.markdown(
                "### Numerical Comparison"
            )

            st.dataframe(
                comparison.round(
                    4
                ),
                width="stretch",
                hide_index=True,
            )

            render_download_button(
                comparison,
                "thermo_x_validation_comparison.csv",
                "Download Validation Results",
            )

            st.caption(
                "These results are software-model outputs. "
                "They are not physical prototype measurements."
            )

        except Exception as exc:

            st.error(
                "The validation dataset could not be processed."
            )

            st.exception(
                exc

            )

    else:

        st.info(
            "No validation run is currently available. "
            "Use the button above to generate a controlled "
            "fixed-vs-adaptive experiment."
        )


# ============================================================
# PREDICTIVE CONTROL
# ============================================================

with prediction_tab:

    st.markdown(
        "## AI Predictive Control"
    )

    st.caption(
        "Four-model thermal intelligence: Random Forest + "
        "XGBoost + Deep MLP + LSTM."
    )

    ensemble_path = (
        MODEL_DIR
        / "thermo_x_ensemble.joblib"
    )

    if not ensemble_path.exists():

        st.warning(
            "The THERMO-X four-model ensemble has not been trained yet."
        )

        st.markdown(
            "Run the complete AI pipeline:"
        )

        st.code(
            (
                "python -m ml.run_ai_pipeline "
                "--scenarios 80 "
                "--duration 240 "
                "--horizon 30"
            ),
            language="powershell",
        )

    else:

        bundle = (
            load_ensemble_bundle(
                ensemble_path
            )
        )

        st.success(
            "FOUR-MODEL THERMO-X AI ENGINE ONLINE"
        )

        # ----------------------------------------------------
        # AI scenario controls
        # ----------------------------------------------------

        st.markdown(
            "### AI Scenario Lab"
        )

        st.caption(
            "Change the operating condition and let the "
            "ensemble predict the thermal response."
        )

        control_1, control_2 = (
            st.columns(2)
        )

        with control_1:

            scenario_pattern = st.selectbox(
                "Thermal scenario",
                [
                    "Mixed",
                    "Spike",
                    "Ramp",
                    "High Load",
                    "Unbalanced",
                ],
                index=0,
            )

            ambient_temperature = st.slider(
                "Ambient temperature (°C)",
                min_value=24.0,
                max_value=38.0,
                value=30.0,
                step=1.0,
            )

        with control_2:

            peak_intensity = st.slider(
                "Peak workload (%)",
                min_value=50.0,
                max_value=100.0,
                value=90.0,
                step=5.0,
            )

            zone_imbalance = st.slider(
                "Zone imbalance (%)",
                min_value=0.0,
                max_value=60.0,
                value=20.0,
                step=5.0,
            )

        scenario_duration = st.slider(
            "Scenario duration (seconds)",
            min_value=60,
            max_value=240,
            value=150,
            step=10,
        )

        run_ai = st.button(
            "▶ RUN AI FORECAST",
            type="primary",
            width="stretch",
        )

        if run_ai:

            with st.spinner(
                "Running thermal scenario and four-model prediction..."
            ):

                scenario_dataframe = (
                    run_interactive_scenario(
                        pattern=scenario_pattern,

                        duration_s=float(
                            scenario_duration
                        ),

                        ambient_temperature_c=(
                            float(
                                ambient_temperature
                            )
                        ),

                        imbalance_pct=(
                            float(
                                zone_imbalance
                            )
                        ),

                        peak_intensity_pct=(
                            float(
                                peak_intensity
                            )
                        ),

                        seed=42,
                    )
                )

                prediction_dataframe = (
                    predict_dataframe(
                        scenario_dataframe,
                        bundle,
                    )
                )

            st.session_state[
                "ai_scenario_dataframe"
            ] = scenario_dataframe

            st.session_state[
                "ai_prediction_dataframe"
            ] = prediction_dataframe

            st.success(
                "AI forecast completed."
            )

        prediction_dataframe = (
            st.session_state.get(
                "ai_prediction_dataframe",
                pd.DataFrame(),
            )
        )

        if not prediction_dataframe.empty:

            # ------------------------------------------------
            # Latest system state
            # ------------------------------------------------

            latest_time = (
                prediction_dataframe[
                    "timestamp_s"
                ].max()
            )

            latest = (
                prediction_dataframe[
                    prediction_dataframe[
                        "timestamp_s"
                    ]
                    == latest_time
                ]
                .copy()
            )

            # ------------------------------------------------
            # AI OUTCOME
            # ------------------------------------------------

            current_average = float(
                latest[
                    "temperature_c"
                ].mean()
            )

            predicted_average = float(
                latest[
                    "ensemble_future_temperature_c"
                ]
                .dropna()
                .mean()
            )

            forecast_delta = (
                predicted_average
                - current_average
            )

            consensus = float(
                latest[
                    "ensemble_spread_c"
                ]
                .dropna()
                .mean()
            )

            if forecast_delta >= 5.0:

                outlook = (
                    "RAPID THERMAL RISE"
                )

                action = (
                    "PRE-COOL NOW"
                )

            elif forecast_delta >= 2.0:

                outlook = (
                    "THERMAL RISE DETECTED"
                )

                action = (
                    "INCREASE COOLING"
                )

            elif forecast_delta <= -2.0:

                outlook = (
                    "THERMAL LOAD FALLING"
                )

                action = (
                    "COOLING CAN BE REDUCED"
                )

            else:

                outlook = (
                    "THERMAL STATE STABLE"
                )

                action = (
                    "MAINTAIN CURRENT COOLING"
                )

            if consensus < 1.0:

                consensus_text = (
                    "HIGH MODEL AGREEMENT"
                )

            elif consensus < 2.0:

                consensus_text = (
                    "MODERATE MODEL AGREEMENT"
                )

            else:

                consensus_text = (
                    "MODELS DIVERGE"
                )

            st.divider()

            st.markdown(
                "### AI Outcome"
            )

            outcome_1, outcome_2, outcome_3 = (
                st.columns(3)
            )

            with outcome_1:

                st.info(
                    f"THERMAL OUTLOOK\n\n"
                    f"**{outlook}**"
                )

            with outcome_2:

                st.success(
                    f"AI ACTION\n\n"
                    f"**{action}**"
                )

            with outcome_3:

                st.warning(
                    f"MODEL CONSENSUS\n\n"
                    f"**{consensus_text}**"
                )

            st.caption(
                "This is a simulation-level advisory outcome. "
                "It is not a physical safety limit or hardware command."
            )

            # ------------------------------------------------
            # Thermal-zone cards
            # ------------------------------------------------

            st.markdown(
                "### Zone Forecast"
            )

            zone_columns = (
                st.columns(
                    len(
                        latest
                    )
                )
            )

            for column, (
                _,
                row,
            ) in zip(
                zone_columns,
                latest.iterrows(),
            ):

                with column:

                    st.markdown(
                        f"#### {row['server_id']}"
                    )

                    current = float(
                        row[
                            "temperature_c"
                        ]
                    )

                    future = float(
                        row[
                            "ensemble_future_temperature_c"
                        ]
                    )

                    change = (
                        future
                        - current
                    )

                    st.metric(
                        "Current",
                        f"{current:.2f} °C",
                    )

                    st.metric(
                        "AI forecast",
                        f"{future:.2f} °C",
                        delta=(
                            f"{change:+.2f} °C"
                        ),
                    )

            # ------------------------------------------------
            # Prediction chart
            # ------------------------------------------------

            st.markdown(
                "### Ensemble Thermal Forecast"
            )

            figure = {
                "data": [],

                "layout": {
                    "template":
                        "plotly_dark",

                    "paper_bgcolor":
                        "rgba(0,0,0,0)",

                    "plot_bgcolor":
                        "#0b1220",

                    "title": {
                        "text":
                            "Current vs Predicted Future Temperature",
                        "x":
                            0.02,
                    },

                    "xaxis": {
                        "title":
                            "Simulation Time (s)"
                    },

                    "yaxis": {
                        "title":
                            "Temperature (°C)"
                    },

                    "hovermode":
                        "x unified",

                    "height":
                        440,
                },
            }

            for server_id, group in (
                prediction_dataframe
                .groupby(
                    "server_id"
                )
            ):

                group = (
                    group
                    .sort_values(
                        "timestamp_s"
                    )
                )

                figure[
                    "data"
                ].append(
                    {
                        "x":
                            group[
                                "timestamp_s"
                            ],

                        "y":
                            group[
                                "temperature_c"
                            ],

                        "type":
                            "scatter",

                        "mode":
                            "lines",

                        "name":
                            (
                                f"{server_id} "
                                "Current"
                            ),

                        "line": {
                            "width":
                                2.5,
                        },
                    }
                )

                figure[
                    "data"
                ].append(
                    {
                        "x":
                            group[
                                "timestamp_s"
                            ],

                        "y":
                            group[
                                "ensemble_future_temperature_c"
                            ],

                        "type":
                            "scatter",

                        "mode":
                            "lines",

                        "name":
                            (
                                f"{server_id} "
                                "AI Forecast"
                            ),

                        "line": {
                            "width":
                                2.5,
                            "dash":
                                "dash",
                        },
                    }
                )

            st.plotly_chart(
                figure,
                width="stretch",
                key=(
                    "ensemble-live-forecast"
                ),
            )

            # ------------------------------------------------
            # Model contributions
            # ------------------------------------------------

            with st.expander(
                "Four-model fusion details"
            ):

                weight_columns = (
                    st.columns(4)
                )

                for column, model_name in zip(
                    weight_columns,
                    [
                        "Random Forest",
                        "XGBoost",
                        "Deep MLP",
                        "LSTM",
                    ],
                ):

                    with column:

                        weight = (
                            bundle[
                                "weights"
                            ][
                                model_name
                            ]
                        )

                        st.metric(
                            model_name,
                            f"{weight * 100:.1f}%",
                        )

                st.markdown(
                    "The ensemble weights were learned from "
                    "a separate validation set; they are not "
                    "manually chosen for the demonstration."
                )

            # ------------------------------------------------
            # Diagnostic results
            # ------------------------------------------------

            with st.expander(
                "ML validation diagnostics"
            ):

                metrics_table = pd.DataFrame(
                    bundle[
                        "metrics"
                    ][
                        "test_table"
                    ]
                )

                st.dataframe(
                    metrics_table.round(
                        4
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "These are held-out scenario results from "
                    "the generated simulation dataset."
                )

            # ------------------------------------------------
            # Download
            # ------------------------------------------------

            csv_data = (
                prediction_dataframe
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )

            st.download_button(
                "Download AI Scenario Results",

                data=csv_data,

                file_name=(
                    "thermo_x_ai_scenario.csv"
                ),

                mime="text/csv",

                width="stretch",
            )

        else:

            st.info(
                "Configure a scenario and press "
                "'RUN AI FORECAST'."
            )

# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

with architecture_tab:

    st.markdown(
        "## System Architecture"
    )

    st.caption(
        "The software is designed so that the virtual plant "
        "can later be replaced by physical ESP32 hardware."
    )

    st.markdown(
        "### High-level control architecture"
    )

    st.code(
        """
                         THERMO-X
                            │
             ┌──────────────┴──────────────┐
             │                             │
        WORKLOAD                      ENVIRONMENT
             │                    Ambient / Weather
             │                             │
             └──────────────┬──────────────┘
                            ↓
                  ┌──────────────────┐
                  │ THERMAL SENSING  │
                  │                  │
                  │ Temperature      │
                  │ Flow             │
                  │ Pressure         │
                  │ Power            │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ SAFETY CONTROLLER│
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ ADAPTIVE CONTROL │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ PUMP / FAN /     │
                  │ VALVE COMMANDS   │
                  └────────┬─────────┘
                           ↓
                  ┌──────────────────┐
                  │ THERMAL PLANT    │
                  │                  │
                  │ Heater           │
                  │ Cold plate       │
                  │ Coolant loop     │
                  │ Radiator         │
                  └────────┬─────────┘
                           │
                           └───────→ feedback
        """,
        language="text",
    )

    st.markdown(
        "### Physical thermal path"
    )

    st.code(
        """
        HEAT SOURCE
             │
             ▼
        THERMAL BLOCK
             │
             ▼
        COLD PLATE
             │
             ▼
        CLOSED COOLANT LOOP
             │
             ▼
        HEAT EXCHANGER
             │
             ▼
        DRY RADIATOR
             │
             ▼
        AMBIENT AIR
             │
             └──────────→ RETURN
        """,
        language="text",
    )

    st.markdown(
        "### Software → Hardware Transition"
    )

    card_a, card_b, card_c = st.columns(
        3
    )

    with card_a:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### CURRENT"
            )

            st.markdown(
                "### Virtual Plant"
            )

            st.caption(
                "Python thermal model, virtual sensors, "
                "simulated pump/fan and workload generator."
            )

    with card_b:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### NEXT"
            )

            st.markdown(
                "### ESP32 Interface"
            )

            st.caption(
                "Real temperature, flow, pressure and "
                "actuator signals will use the same "
                "software interfaces."
            )

    with card_c:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### FUTURE"
            )

            st.markdown(
                "### Physical Prototype"
            )

            st.caption(
                "Heater blocks, cold plate, closed-loop "
                "coolant, pump, radiator, fan and safety hardware."
            )

    st.divider()

    st.markdown(
        "### Engineering principle"
    )

    st.info(
        "Cooling intensity should follow actual thermal demand "
        "while every thermal zone remains inside its defined "
        "safe operating envelope."
    )


# ============================================================
# FOOTER
# ============================================================

render_footer()