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
        Return:

            (workload_percent, heater_power_w)
        """

        segments = self.schedules.get(
            server_id
        )

        if not segments:
            return 0.0, 0.0

        for segment in segments:

            if segment.contains(
                timestamp_s
            ):
                return (
                    segment.workload_pct,
                    segment.heater_power_w,
                )

        # Return zero after the experiment ends.
        return 0.0, 0.0


def create_demo_profile(
    server_ids: list[str],
) -> WorkloadProfile:
    """
    Create the initial two-zone THERMO-X workload.

    The workload stages are intentionally simple:

        LOW
        MEDIUM
        HIGH
        LOW

    Heater powers are software test conditions only.
    """

    schedules: dict[
        str,
        list[WorkloadSegment],
    ] = {}

    for index, server_id in enumerate(
        server_ids
    ):

        # Offset the second server's workload to create
        # a slightly different thermal demand profile.
        offset = (
            0.0
            if index == 0
            else 15.0
        )

        schedules[
            server_id
        ] = [
            WorkloadSegment(
                start_s=0.0,
                end_s=60.0 + offset,
                workload_pct=20.0,
                heater_power_w=20.0,
            ),

            WorkloadSegment(
                start_s=60.0 + offset,
                end_s=120.0 + offset,
                workload_pct=50.0,
                heater_power_w=50.0,
            ),

            WorkloadSegment(
                start_s=120.0 + offset,
                end_s=180.0 + offset,
                workload_pct=90.0,
                heater_power_w=90.0,
            ),

            WorkloadSegment(
                start_s=180.0 + offset,
                end_s=240.0 + offset,
                workload_pct=20.0,
                heater_power_w=20.0,
            ),
        ]

    return WorkloadProfile(
        schedules
    )