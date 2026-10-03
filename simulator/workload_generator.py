"""
Deterministic workload generator for THERMO-X experiments.

The generated workload represents an experimental electrical
heat source. It must NOT be described as actual GPU power.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class WorkloadSegment:
    """One time interval in a workload schedule."""

    start_s: float
    end_s: float

    # Human-facing workload indicator.
    workload_pct: float

    # Actual simulated heat injected into the thermal model.
    heater_power_w: float

    def contains(
        self,
        timestamp_s: float,
    ) -> bool:
        """Check whether a timestamp belongs to this segment."""

        return (
            self.start_s
            <= timestamp_s
            < self.end_s
        )


class WorkloadProfile:
    """
    Workload schedule for one or more simulated servers.
    """

    def __init__(
        self,
        schedules: Mapping[
            str,
            list[WorkloadSegment],
        ],
    ) -> None:

        self.schedules = dict(
            schedules
        )

    def get_workload(
        self,
        timestamp_s: float,
        server_id: str,
    ) -> tuple[float, float]:
        """
        Return the workload and simulated heater power for a server.

        Behaviour:

            1. If the timestamp is inside a configured segment,
            return that segment's values.

            2. If the timestamp occurs after the final segment,
            hold the final segment's operating state.

        Holding the final state prevents a longer simulation from
        unexpectedly turning the workload and heat source to zero
        merely because the original schedule ended.

        Returns:

            (workload_percent, heater_power_w)
        """

        segments = self.schedules.get(
            server_id
        )

        if not segments:
            return (
                0.0,
                0.0,
            )

        # --------------------------------------------------------
        # Normal case:
        # find the segment containing the current timestamp.
        # --------------------------------------------------------

        for segment in segments:

            if segment.contains(
                timestamp_s
            ):

                return (
                    segment.workload_pct,
                    segment.heater_power_w,
                )

        # --------------------------------------------------------
        # Important fallback:
        #
        # If the simulation continues beyond the final configured
        # segment, HOLD the last operating state instead of
        # silently changing to 0% / 0 W.
        #
        # This is particularly important for demonstrations where
        # the user selects a duration longer than the base workload
        # schedule.
        # --------------------------------------------------------

        final_segment = (
            segments[-1]
        )

        if (
            timestamp_s
            >= final_segment.end_s
        ):

            return (
                final_segment.workload_pct,
                final_segment.heater_power_w,
            )

    # --------------------------------------------------------
    # If a timestamp somehow occurs before the first segment,
    # return a safe zero state.
    # --------------------------------------------------------

        return (
            0.0,
            0.0,
        )


def create_demo_profile(
    server_ids: list[str],
) -> WorkloadProfile:
    """
    Create the initial two-zone THERMO-X workload.

    Sequence:

        LOW
        MEDIUM
        HIGH
        LOW

    The final LOW segment is intentionally extended beyond the
    normal 240-second demonstration so longer simulations do
    not fall into an artificial 0% / 0 W state.

    Heater powers are software test conditions only.
    """

    schedules: dict[
        str,
        list[WorkloadSegment],
    ] = {}

    for index, server_id in enumerate(
        server_ids
    ):

        # SERVER_B has a 15-second phase offset so the two
        # thermal zones do not transition at exactly the same
        # time.
        offset = (
            0.0
            if index == 0
            else 15.0
        )

        schedules[
            server_id
        ] = [

            # ------------------------------------------------
            # LOW LOAD
            # ------------------------------------------------
            WorkloadSegment(
                start_s=0.0,

                end_s=60.0 + offset,

                workload_pct=20.0,

                heater_power_w=20.0,
            ),

            # ------------------------------------------------
            # MEDIUM LOAD
            # ------------------------------------------------
            WorkloadSegment(
                start_s=60.0 + offset,

                end_s=120.0 + offset,

                workload_pct=50.0,

                heater_power_w=50.0,
            ),

            # ------------------------------------------------
            # HIGH LOAD
            # ------------------------------------------------
            WorkloadSegment(
                start_s=120.0 + offset,

                end_s=180.0 + offset,

                workload_pct=90.0,

                heater_power_w=90.0,
            ),

            # ------------------------------------------------
            # LOW LOAD
            #
            # Continue indefinitely.
            # ------------------------------------------------
            WorkloadSegment(
                start_s=180.0 + offset,

                end_s=float("inf"),

                workload_pct=20.0,

                heater_power_w=20.0,
            ),
        ]

    return WorkloadProfile(
        schedules
    )