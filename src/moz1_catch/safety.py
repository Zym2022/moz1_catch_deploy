"""Last-resort clamps between the planner and the controller.

The planner already respects its own speed and acceleration budgets; this layer
only contains configuration or calibration mistakes (wrong extrinsics, a stale
plan, a mistyped wait pose) before they reach the controller.  Every clamp is
recorded and reported - a clean catch attempt should produce zero violations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from moz1_catch.config import SafetyConfig


@dataclass
class SafetyEnvelope:
    config: SafetyConfig

    def clamp(self, positions: np.ndarray, previous: np.ndarray | None, dt: float
              ) -> tuple[np.ndarray, list[str]]:
        """Project targets into the workspace box and cap the per-command step."""
        clamped = np.clip(np.asarray(positions, dtype=float),
                          self.config.workspace[:, 0], self.config.workspace[:, 1])
        violations = []
        if np.any(clamped != positions):
            violations.append("workspace")
        if previous is not None and dt > 0:
            step = np.linalg.norm(clamped - previous, axis=1)
            limit = self.config.max_cartesian_speed_mps * dt
            for side in np.flatnonzero(step > limit):
                clamped[side] = previous[side] + (clamped[side] - previous[side]) * (limit / step[side])
                violations.append(f"speed:{'left' if side == 0 else 'right'}")
        return clamped, violations
