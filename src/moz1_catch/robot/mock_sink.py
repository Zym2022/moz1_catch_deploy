"""In-memory command sink for replay/dry runs and tests."""

from __future__ import annotations

import numpy as np

from moz1_catch.robot.sink import HandTargets


class MockCartesianSink:
    """Records every published command instead of driving hardware."""

    def __init__(self):
        self.sent_times: list[float] = []
        self.sent_positions: list[np.ndarray] = []
        self.sent_quaternions: list[np.ndarray] = []

    def send(self, targets: HandTargets, t_host: float) -> None:
        self.sent_times.append(t_host)
        self.sent_positions.append(targets.positions_m.copy())
        self.sent_quaternions.append(np.array([rotation.as_quat() for rotation in targets.rotations]))

    @property
    def count(self) -> int:
        return len(self.sent_times)
