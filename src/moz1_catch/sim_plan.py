"""Replay a frozen MozBoxer simulation plan through the deployment Executor.

A catch trace saved by the simulation (``np.savez_compressed`` with the keys
below) already contains the planner's output as a dense time series: the palm
targets commanded every 1 ms physics tick.  :class:`SimTracePlan` wraps that
series in the small interface :class:`moz1_catch.executor.Executor` expects
from a plan - ``contact_time``, ``stop_time`` and ``target(t)`` - so a recorded
plan can be streamed to the robot without re-running the planner.

Values at the command instants (120 Hz on hardware) are linear
(position/velocity) and SLERP (rotation) interpolations of the 1 ms samples,
so the replayed trajectory is the simulated one up to interpolation noise.

Required npz keys (base_link frame, hands ordered [left, right]):

- ``time_s`` (N,), ``target_palm_position`` (N, 2, 3),
  ``target_palm_rotation_xyzw`` (N, 2, 4), ``target_palm_velocity`` (N, 2, 3)
- ``observation_time_s``, ``execution_latency_s``, ``contact_time_s``,
  ``catch_decision``

Optional keys that sharpen the plan-segment detection:
``contact_positions_m`` (2, 3) and ``start_palm_positions_m`` (2, 3).

Traces are written with one of three time conventions: observation-relative
(index 0 is the observation, the plan starts after ``execution_latency_s``;
``simulation.py`` accept traces), episode-absolute (index 0 is episode start;
live episode streams) or plan-relative (index 0 is the plan start).
``load_sim_trace_plan`` tries the offset implied by each convention and accepts
the first whose start sits at ``start_palm_positions_m`` and whose contact time
lands near ``contact_positions_m``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

_REQUIRED_KEYS = ("time_s", "target_palm_position", "target_palm_rotation_xyzw",
                  "target_palm_velocity", "observation_time_s",
                  "execution_latency_s", "contact_time_s", "catch_decision")

_START_MATCH_TOLERANCE_M = 0.005
_CONTACT_MATCH_TOLERANCE_M = 0.030


class SimTracePlan:
    """Plan-like view of the execution segment of a simulation trace.

    ``speed_scale`` stretches time (0.5 = replay takes twice as long, all
    velocities halved); the path shape is unchanged.
    """

    def __init__(self, time_s: np.ndarray, positions: np.ndarray,
                 quaternions_xyzw: np.ndarray, velocities: np.ndarray,
                 contact_time_s: float, speed_scale: float = 1.0):
        if speed_scale <= 0.:
            raise ValueError(f"speed_scale must be positive, got {speed_scale}")
        self._time_s = time_s
        self._positions = positions
        self._velocities = velocities
        self._slerps = tuple(
            Slerp(time_s, Rotation.from_quat(quaternions_xyzw[:, hand]))
            for hand in range(2))
        self._scale = speed_scale
        self._plan_duration = float(time_s[-1] - time_s[0])
        self.contact_time = contact_time_s / speed_scale
        self.stop_time = max(0., self._plan_duration - contact_time_s) / speed_scale

    def target(self, t: float):
        """Positions, rotations, velocities at replay time t (seconds)."""
        plan_t = float(np.clip(t * self._scale, 0., self._plan_duration))
        query = self._time_s[0] + plan_t
        positions = np.empty((2, 3))
        velocities = np.empty((2, 3))
        for hand in range(2):
            for axis in range(3):
                positions[hand, axis] = np.interp(
                    query, self._time_s, self._positions[:, hand, axis])
                velocities[hand, axis] = np.interp(
                    query, self._time_s, self._velocities[:, hand, axis])
        rotations = tuple(slerp(query) for slerp in self._slerps)
        return positions, rotations, velocities * self._scale, None


def _locate_plan_start(path, time_s, positions, dt, start_s, contact_s, data):
    """Index of the plan's t=0 sample, robust to both trace time conventions.

    A candidate index is accepted when the target stream sits at the recorded
    plan-start pose there (within ``_START_MATCH_TOLERANCE_M``) and is near the
    contact positions one ``contact_time_s`` later (within
    ``_CONTACT_MATCH_TOLERANCE_M`` - the streamed target keeps a small normal
    gap at contact, hence the looser bound).  Episode-absolute traces hold the
    wait pose first, so the observation+latency candidate is tried before 0.
    """
    start_positions = contact_positions = None
    if "start_palm_positions_m" in data.files:
        start_positions = np.asarray(data["start_palm_positions_m"], dtype=float)
    if "contact_positions_m" in data.files:
        contact_positions = np.asarray(data["contact_positions_m"], dtype=float)
    contact_samples = int(round(contact_s / dt))
    latency_s = float(data["execution_latency_s"])
    candidates = [int(round((latency_s - time_s[0]) / dt)),       # observation-relative trace
                  int(round((start_s - time_s[0]) / dt)),         # episode-absolute trace
                  0]                                              # plan-relative trace
    for index in candidates:
        if not 0 <= index <= len(time_s) - 2:
            continue
        if start_positions is None:
            return index, None
        start_offset = float(np.abs(positions[index] - start_positions).max())
        if start_offset > _START_MATCH_TOLERANCE_M:
            continue
        if contact_positions is not None:
            probe = min(index + contact_samples, len(positions) - 1)
            contact_offset = float(np.abs(positions[probe] - contact_positions).max())
            if contact_offset > _CONTACT_MATCH_TOLERANCE_M:
                continue
        return index, start_offset
    raise ValueError(
        f"{path}: cannot locate the plan segment start (tried indices "
        f"{candidates}; the recorded target stream never sits at the plan "
        f"start pose within {_START_MATCH_TOLERANCE_M * 1000:.0f} mm)")


def load_sim_trace_plan(path: Path | str, speed_scale: float = 1.0):
    """Load and validate a trace npz; returns (plan, info dict)."""
    path = Path(path)
    with np.load(path) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}")
        decision = str(data["catch_decision"])
        if decision != "accept":
            raise ValueError(f"{path}: only accept decisions can be replayed, "
                             f"got {decision!r}")
        time_s = np.asarray(data["time_s"], dtype=float)
        positions = np.asarray(data["target_palm_position"], dtype=float)
        quaternions = np.asarray(data["target_palm_rotation_xyzw"], dtype=float)
        velocities = np.asarray(data["target_palm_velocity"], dtype=float)
        start_s = float(data["observation_time_s"]) + float(data["execution_latency_s"])
        contact_s = float(data["contact_time_s"])

        if time_s.ndim != 1 or len(time_s) < 2 or np.any(np.diff(time_s) <= 0.):
            raise ValueError(f"{path}: time_s must be increasing with >= 2 samples")
        if positions.shape != (len(time_s), 2, 3) \
                or quaternions.shape != (len(time_s), 2, 4) \
                or velocities.shape != (len(time_s), 2, 3):
            raise ValueError(f"{path}: target arrays must be (N, 2, 3)/(N, 2, 4) "
                             f"matching time_s, got {positions.shape}, "
                             f"{quaternions.shape}, {velocities.shape}")
        dt = float(np.median(np.diff(time_s)))
        start_index, start_offset = _locate_plan_start(
            path, time_s, positions, dt, start_s, contact_s, data)

    plan = SimTracePlan(time_s[start_index:], positions[start_index:],
                        quaternions[start_index:], velocities[start_index:],
                        contact_s, speed_scale=speed_scale)
    info = {
        "source": str(path),
        "decision": decision,
        "trace_dt_s": dt,
        "trace_samples": int(len(time_s) - start_index),
        "plan_start_index": start_index,
        "plan_start_offset_m": start_offset,
        "execution_start_s": start_s,
        "max_palm_speed_mps": float(np.linalg.norm(velocities[start_index:],
                                                   axis=2).max()) * speed_scale,
        "speed_scale": speed_scale,
        "contact_time_s": plan.contact_time,
        "stop_time_s": plan.stop_time,
    }
    return plan, info
