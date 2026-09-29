"""Observation model and the source interface shared by UDP and replay inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Observation:
    """One box pose, already converted to the base_link planning frame.

    t_s is on the runtime host monotonic clock (time.perf_counter time base).
    valid is False for packets that arrived with a bad tracking state; they are
    still surfaced so the runtime can count dropouts, but never used to plan.
    """

    t_s: float
    position_m: np.ndarray
    quat_xyzw: np.ndarray
    valid: bool
    device_t_s: float


class BoxObservationSource(Protocol):
    """Yields observations in order; returns None on timeout instead of raising."""

    def next(self, timeout_s: float) -> Observation | None: ...
