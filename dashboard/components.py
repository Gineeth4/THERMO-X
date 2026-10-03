"""
THERMO-X Dashboard UI Components.

This module contains presentation-only functions.

Important architecture rule:

    UI code
        ↓
    application layer
        ↓
    controller / simulator / analytics

The dashboard should not contain thermal-control algorithms.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# GLOBAL DASHBOARD STYLING
# ============================================================


def inject_dashboard_css() -> None:
    """
    Apply the visual system used by THERMO-X.

    The previous dashboard used white metric cards while the
    surrounding application used a dark theme. That produced
    the white-on-white readability problem visible in the
    screenshots.

    All dashboard cards now explicitly use a dark surface and
    explicitly define their text colors.
    """

    st.markdown(
        """
        <style>

        /* ====================================================
           APPLICATION BACKGROUND
           ==================================================== */

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(
                    circle at 80% 0%,
                    rgba(14, 116, 144, 0.10),
                    transparent 30%
                ),
                #070b14;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }


        /* ====================================================
           SIDEBAR
           ==================================================== */

        [data-testid="stSidebar"] {
            background: #0a101b;

            border-right:
                1px solid
                rgba(148, 163, 184, 0.14);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: #e2e8f0;
        }


        /* ====================================================
           HEADINGS
           ==================================================== */

        h1, h2, h3 {
            letter-spacing: -0.025em;
        }


        /* ====================================================
           METRIC CARDS
           ==================================================== */

        div[data-testid="stMetric"] {
            background: #0f1724 !important;

            border:
                1px solid
                rgba(148, 163, 184, 0.16) !important;

            border-radius: 14px !important;

            padding:
                0.85rem
                0.95rem !important;

            min-height: 98px;

            box-shadow:
                0
                8px
                24px
                rgba(0, 0, 0, 0.16);
        }

        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;

            font-size: 0.78rem !important;

            font-weight: 650 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;

            font-size: 1.45rem !important;

            font-weight: 720 !important;
        }

        div[data-testid="stMetricDelta"] {
            font-size: 0.72rem !important;
        }


        /* ====================================================
           CONTAINERS
           ==================================================== */

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
        }


        /* ====================================================
           TABS
           ==================================================== */

        button[data-baseweb="tab"] {
            font-weight: 650;

            color: #94a3b8;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #22d3ee;
        }


        /* ====================================================
           BUTTONS
           ==================================================== */

        .stButton > button {
            border-radius: 10px;

            min-height: 2.6rem;

            font-weight: 700;
        }


        /* ====================================================
           DOWNLOAD BUTTONS
           ==================================================== */

        .stDownloadButton > button {
            border-radius: 10px;

            font-weight: 650;
        }


        /* ====================================================
           DATAFRAMES
           ==================================================== */

        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }


        /* ====================================================
           SMALL STATUS LABEL
           ==================================================== */

        .status-chip {
            display: inline-block;

            padding:
                0.28rem
                0.65rem;

            border-radius: 999px;

            font-size: 0.72rem;

            font-weight: 750;

            letter-spacing: 0.03em;
        }

        .status-chip.safe {
            color: #bbf7d0;

            background:
                rgba(
                    34,
                    197,
                    94,
                    0.13
                );

            border:
                1px solid
                rgba(
                    34,
                    197,
                    94,
                    0.28
                );
        }

        .status-chip.warning {
            color: #fde68a;

            background:
                rgba(
                    245,
                    158,
                    11,
                    0.13
                );

            border:
                1px solid
                rgba(
                    245,
                    158,
                    11,
                    0.28
                );
        }

        .status-chip.emergency {
            color: #fecaca;

            background:
                rgba(
                    239,
                    68,
                    68,
                    0.13
                );

            border:
                1px solid
                rgba(
                    239,
                    68,
                    68,
                    0.28
                );
        }


        /* ====================================================
           TOP APPLICATION BAR
           ==================================================== */

        .app-bar {
            display: flex;

            align-items: center;

            justify-content: space-between;

            padding:
                0.75rem
                1rem;

            margin-bottom: 1.25rem;

            border:
                1px solid
                rgba(
                    148,
                    163,
                    184,
                    0.14
                );

            border-radius: 14px;

            background:
                rgba(
                    15,
                    23,
                    36,
                    0.78
                );

            backdrop-filter: blur(12px);

            box-shadow:
                0
                10px
                30px
                rgba(
                    0,
                    0,
                    0,
                    0.12
                );
        }

        .app-brand {
            font-size: 1.0rem;

            font-weight: 800;

            color: #f8fafc;

            letter-spacing: 0.02em;
        }

        .app-subtitle {
            margin-left: 0.65rem;

            color: #64748b;

            font-size: 0.76rem;
        }


        /* ====================================================
           KPI STRIP
           ==================================================== */

        .kpi-strip {
            padding:
                0.65rem
                0.8rem;

            border:
                1px solid
                rgba(
                    148,
                    163,
                    184,
                    0.12
                );

            border-radius: 14px;

            background: #0b1220;

            margin-bottom: 1rem;
        }


        /* ====================================================
           SECTION LABELS
           ==================================================== */

        .eyebrow {
            text-transform: uppercase;

            letter-spacing: 0.10em;

            font-size: 0.68rem;

            font-weight: 800;

            color: #22d3ee;

            margin-bottom: 0.2rem;
        }

        .muted {
            color: #64748b;

            font-size: 0.82rem;
        }


        /* ====================================================
           FEATURE CARD
           ==================================================== */

        .feature-card {
            padding: 1rem;

            border-radius: 15px;

            background:
                linear-gradient(
                    145deg,
                    #101a29,
                    #0b1220
                );

            border:
                1px solid
                rgba(
                    148,
                    163,
                    184,
                    0.13
                );

            min-height: 130px;
        }

        .feature-title {
            color: #f8fafc;

            font-weight: 720;

            margin-bottom: 0.45rem;
        }

        .feature-body {
            color: #94a3b8;

            font-size: 0.82rem;

            line-height: 1.55;
        }


        /* ====================================================
           FOOTER
           ==================================================== */

        .dashboard-footer {
            margin-top: 2rem;

            padding-top: 1rem;

            border-top:
                1px solid
                rgba(
                    148,
                    163,
                    184,
                    0.10
                );

            text-align: center;

            color: #475569;

            font-size: 0.72rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP BAR
# ============================================================


def render_application_bar(
    status: str = "SIMULATION",
) -> None:
    """
    Render the THERMO-X application bar using native
    Streamlit components.

    IMPORTANT:
    No HTML is used here.

    This prevents HTML markup from ever appearing as
    literal text in the dashboard.
    """

    with st.container(
        border=True
    ):

        left, center, right = (
            st.columns(
                [2.2, 4.5, 1.8]
            )
        )

        with left:

            st.markdown(
                "### 🌡️ THERMO-X"
            )

        with center:

            st.caption(
                "Adaptive Thermal Control Centre"
            )

        with right:

            normalized = (
                str(status)
                .strip()
                .upper()
            )

            if (
                "ONLINE"
                in normalized
            ):

                st.success(
                    "● ONLINE",
                    icon="✅",
                )

            elif (
                "WARNING"
                in normalized
            ):

                st.warning(
                    "● WARNING",
                    icon="⚠️",
                )

            else:

                st.info(
                    f"● {normalized}",
                    icon="ℹ️",
                )


# ============================================================
# STATUS
# ============================================================


def render_status_chip(
    status: str,
) -> None:
    """
    Render system status using native Streamlit UI.

    HTML is intentionally avoided.
    """

    normalized = (
        str(status)
        .strip()
        .upper()
    )

    if normalized == "SAFE":

        st.success(
            "SAFE",
            icon="✅",
        )

    elif normalized == "WARNING":

        st.warning(
            "WARNING",
            icon="⚠️",
        )

    elif normalized in {
        "EMERGENCY",
        "FAULT",
    }:

        st.error(
            normalized,
            icon="🚨",
        )

    else:

        st.info(
            normalized,
            icon="ℹ️",
        )


# ============================================================
# LATEST DATA
# ============================================================


def latest_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return the most recent sample for each thermal zone."""

    if dataframe.empty:
        return pd.DataFrame()

    latest_time = float(
        dataframe[
            "timestamp_s"
        ].max()
    )

    return (
        dataframe[
            dataframe[
                "timestamp_s"
            ]
            == latest_time
        ]
        .sort_values(
            "server_id"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SERVER CARDS
# ============================================================


def render_server_cards(
    dataframe: pd.DataFrame,
) -> None:
    """Render current thermal state for all zones."""

    latest = latest_rows(
        dataframe
    )

    if latest.empty:

        st.info(
            "No thermal telemetry available."
        )

        return

    st.markdown(
        '<div class="eyebrow">Thermal Zones</div>',
        unsafe_allow_html=True,
    )

    columns = st.columns(
        len(latest)
    )

    for column, (_, row) in zip(
        columns,
        latest.iterrows(),
    ):

        with column:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### {row['server_id']}"
                )

                st.metric(
                    "Temperature",
                    f"{row['temperature_c']:.2f} °C",
                )

                col1, col2 = (
                    st.columns(2)
                )

                with col1:

                    st.metric(
                        "Workload",
                        f"{row['workload_pct']:.0f} %",
                    )

                with col2:

                    st.metric(
                        "Heat Load",
                        f"{row['heater_power_w']:.1f} W",
                    )

                col3, col4 = (
                    st.columns(2)
                )

                with col3:

                    st.metric(
                        "Flow",
                        f"{row['flow_rate_lpm']:.3f} L/min",
                    )

                with col4:

                    st.metric(
                        "Valve",
                        f"{row['valve_position_pct']:.1f} %",
                    )


# ============================================================
# SYSTEM KPI CARDS
# ============================================================


def render_system_kpis(
    dataframe: pd.DataFrame,
) -> None:
    """Render shared cooling-loop values."""

    latest = latest_rows(
        dataframe
    )

    if latest.empty:
        return

    row = latest.iloc[0]

    st.markdown(
        '<div class="eyebrow">Cooling Infrastructure</div>',
        unsafe_allow_html=True,
    )

    columns = st.columns(
        6
    )

    values = [
        (
            columns[0],
            "Coolant In",
            f"{row['coolant_inlet_c']:.2f} °C",
        ),

        (
            columns[1],
            "Coolant Out",
            f"{row['coolant_outlet_c']:.2f} °C",
        ),

        (
            columns[2],
            "Coolant ΔT",
            f"{row['delta_t_c']:.2f} °C",
        ),

        (
            columns[3],
            "Pump",
            f"{row['pump_speed_pct']:.1f} %",
        ),

        (
            columns[4],
            "Fan",
            f"{row['fan_speed_pct']:.1f} %",
        ),

        (
            columns[5],
            "Pressure",
            f"{row['pressure_kpa']:.1f} kPa",
        ),
    ]

    for column, label, value in values:

        with column:

            st.metric(
                label,
                value,
            )


# ============================================================
# TEMPERATURE CHART
# ============================================================


def temperature_figure(
    dataframe: pd.DataFrame,
) -> go.Figure:
    """Create the primary thermal-response figure."""

    figure = go.Figure()

    palette = [
        "#38bdf8",
        "#22d3ee",
        "#818cf8",
        "#a78bfa",
        "#34d399",
    ]

    for index, (
        server_id,
        group,
    ) in enumerate(
        dataframe.groupby(
            "server_id"
        )
    ):

        group = (
            group
            .sort_values(
                "timestamp_s"
            )
        )

        figure.add_trace(
            go.Scatter(
                x=group[
                    "timestamp_s"
                ],

                y=group[
                    "temperature_c"
                ],

                mode="lines",

                name=server_id,

                line={
                    "width": 2.6,
                    "color":
                        palette[
                            index
                            % len(
                                palette
                            )
                        ],
                },
            )
        )

    figure.update_layout(
        template="plotly_dark",

        title={
            "text":
                "Thermal Response",
            "x": 0.02,
        },

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="#0b1220",

        font={
            "color":
                "#cbd5e1",
        },

        xaxis={
            "title":
                "Simulation Time (s)",
            "gridcolor":
                "rgba(148,163,184,0.12)",
            "zeroline":
                False,
        },

        yaxis={
            "title":
                "Temperature (°C)",
            "gridcolor":
                "rgba(148,163,184,0.12)",
            "zeroline":
                False,
        },

        hovermode="x unified",

        height=430,

        margin={
            "l": 55,
            "r": 20,
            "t": 70,
            "b": 50,
        },

        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 0,
        },
    )

    return figure


# ============================================================
# WORKLOAD CHART
# ============================================================


def workload_figure(
    dataframe: pd.DataFrame,
) -> go.Figure:
    """Create workload versus time figure."""

    figure = go.Figure()

    palette = [
        "#38bdf8",
        "#f472b6",
        "#a78bfa",
    ]

    for index, (
        server_id,
        group,
    ) in enumerate(
        dataframe.groupby(
            "server_id"
        )
    ):

        group = (
            group
            .sort_values(
                "timestamp_s"
            )
        )

        figure.add_trace(
            go.Scatter(
                x=group[
                    "timestamp_s"
                ],

                y=group[
                    "workload_pct"
                ],

                mode="lines",

                name=server_id,

                line={
                    "width": 2.4,
                    "color":
                        palette[
                            index
                            % len(
                                palette
                            )
                        ],
                },
            )
        )

    figure.update_layout(
        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="#0b1220",

        font={
            "color":
                "#cbd5e1",
        },

        title={
            "text":
                "Computational Workload",
            "x": 0.02,
        },

        xaxis={
            "title":
                "Time (s)",
            "gridcolor":
                "rgba(148,163,184,0.12)",
        },

        yaxis={
            "title":
                "Workload (%)",
            "range": [
                0,
                100,
            ],
            "gridcolor":
                "rgba(148,163,184,0.12)",
        },

        hovermode="x unified",

        height=360,

        margin={
            "l": 55,
            "r": 20,
            "t": 70,
            "b": 50,
        },
    )

    return figure


# ============================================================
# CONTROL CHART
# ============================================================


def cooling_control_figure(
    dataframe: pd.DataFrame,
) -> go.Figure:
    """Create pump/fan response figure."""

    control = (
        dataframe[
            [
                "timestamp_s",
                "pump_speed_pct",
                "fan_speed_pct",
            ]
        ]
        .drop_duplicates(
            subset=[
                "timestamp_s"
            ]
        )
        .sort_values(
            "timestamp_s"
        )
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=control[
                "timestamp_s"
            ],

            y=control[
                "pump_speed_pct"
            ],

            mode="lines",

            name="Pump",

            line={
                "width": 2.7,
                "color": "#22d3ee",
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=control[
                "timestamp_s"
            ],

            y=control[
                "fan_speed_pct"
            ],

            mode="lines",

            name="Fan",

            line={
                "width": 2.5,
                "dash": "dash",
                "color": "#818cf8",
            },
        )
    )

    figure.update_layout(
        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="#0b1220",

        font={
            "color":
                "#cbd5e1",
        },

        title={
            "text":
                "Cooling Command Response",
            "x": 0.02,
        },

        xaxis={
            "title":
                "Time (s)",
            "gridcolor":
                "rgba(148,163,184,0.12)",
        },

        yaxis={
            "title":
                "Command (%)",
            "range": [
                0,
                100,
            ],
            "gridcolor":
                "rgba(148,163,184,0.12)",
        },

        hovermode="x unified",

        height=360,

        margin={
            "l": 55,
            "r": 20,
            "t": 70,
            "b": 50,
        },
    )

    return figure


# ============================================================
# COOLANT DELTA-T CHART
# ============================================================


def coolant_delta_figure(
    dataframe: pd.DataFrame,
) -> go.Figure:
    """Create coolant heat-transfer indicator figure."""

    figure = go.Figure()

    palette = [
        "#34d399",
        "#f59e0b",
        "#a78bfa",
    ]

    for index, (
        server_id,
        group,
    ) in enumerate(
        dataframe.groupby(
            "server_id"
        )
    ):

        group = (
            group
            .sort_values(
                "timestamp_s"
            )
        )

        figure.add_trace(
            go.Scatter(
                x=group[
                    "timestamp_s"
                ],

                y=group[
                    "delta_t_c"
                ],

                mode="lines",

                name=server_id,

                line={
                    "width": 2.4,
                    "color":
                        palette[
                            index
                            % len(
                                palette
                            )
                        ],
                },
            )
        )

    figure.update_layout(
        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="#0b1220",

        font={
            "color":
                "#cbd5e1",
        },

        title={
            "text":
                "Coolant Temperature Rise",
            "x": 0.02,
        },

        xaxis={
            "title":
                "Time (s)",
            "gridcolor":
                "rgba(148,163,184,0.12)",
        },

        yaxis={
            "title":
                "ΔT (°C)",
            "gridcolor":
                "rgba(148,163,184,0.12)",
        },

        hovermode="x unified",

        height=360,

        margin={
            "l": 55,
            "r": 20,
            "t": 70,
            "b": 50,
        },
    )

    return figure


# ============================================================
# FIXED VS ADAPTIVE CHART
# ============================================================


def comparison_figure(
    fixed_dataframe: pd.DataFrame,
    adaptive_dataframe: pd.DataFrame,
) -> go.Figure:
    """Compare average thermal response."""

    fixed = (
        fixed_dataframe
        .groupby(
            "timestamp_s"
        )["temperature_c"]
        .mean()
        .reset_index()
    )

    adaptive = (
        adaptive_dataframe
        .groupby(
            "timestamp_s"
        )["temperature_c"]
        .mean()
        .reset_index()
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=fixed[
                "timestamp_s"
            ],

            y=fixed[
                "temperature_c"
            ],

            mode="lines",

            name="Fixed Cooling",

            line={
                "width": 2.7,
                "dash": "dash",
                "color": "#94a3b8",
            },
        )
    )

    figure.add_trace(
        go.Scatter(
            x=adaptive[
                "timestamp_s"
            ],

            y=adaptive[
                "temperature_c"
            ],

            mode="lines",

            name="THERMO-X Adaptive",

            line={
                "width": 3.0,
                "color": "#22d3ee",
            },
        )
    )

    figure.update_layout(
        template="plotly_dark",

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="#0b1220",

        font={
            "color":
                "#cbd5e1",
        },

        title={
            "text":
                "Fixed vs Adaptive Thermal Behaviour",
            "x": 0.02,
        },

        xaxis={
            "title":
                "Time (s)",
            "gridcolor":
                "rgba(148,163,184,0.12)",
        },

        yaxis={
            "title":
                "Average Temperature (°C)",
            "gridcolor":
                "rgba(148,163,184,0.12)",
        },

        hovermode="x unified",

        height=440,

        margin={
            "l": 55,
            "r": 20,
            "t": 70,
            "b": 50,
        },

        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 0,
        },
    )

    return figure


# ============================================================
# DOWNLOAD
# ============================================================


def render_download_button(
    dataframe: pd.DataFrame,
    filename: str,
    label: str,
) -> None:
    """Render CSV download button."""

    if dataframe.empty:
        return

    data = (
        dataframe
        .to_csv(
            index=False
        )
        .encode(
            "utf-8"
        )
    )

    st.download_button(
        label=label,

        data=data,

        file_name=filename,

        mime="text/csv",

        width="stretch",
    )


# ============================================================
# FOOTER
# ============================================================


def render_footer() -> None:
    """Render the dashboard footer."""

    st.markdown(
        """
        <div class="dashboard-footer">
            THERMO-X • Cryo-inspired adaptive thermal-management
            research platform • Simulation / validation environment
        </div>
        """,
        unsafe_allow_html=True,
    )