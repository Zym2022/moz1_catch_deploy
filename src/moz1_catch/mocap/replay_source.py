"""CSV replay source for the five recorded throws.

Parses the mocap export CSVs (header rows, Timecode HH:MM:SS, xyzw quaternions,
TrackingState), applies the same per-frame box-centre transform as the offline
study, optionally downsamples causally to a target rate, optionally recenters
x/y so the release lands at (0, -1.5, z_release), and paces playback either in
real time or as fast as possible.

Timestamps are mapped onto the runtime host clock by anchoring the first sample
at "now", so the runtime's timing logic runs exactly as it will live.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from moz1_catch.calib import FrameChain
from moz1_catch.config import ReplayConfig
from moz1_catch.mocap.source import Observation

R_GD = np.array(((0., 1., 0.), (0., 0., 1.), (1., 0., 0.)))
C_GD = np.array((-.0003, 0., .0485375))
# Analysis frame basis: +X = mocap Z, +Y = mocap X, +Z = mocap Y.  The recorded
# throws fly along analysis +Y (toward the robot) with +Z roughly up, which is
# the frame the replay profile's mission planes and acceleration prior assume.
ANALYSIS_BASIS_M = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))


def load_recording(path: Path) -> dict:
    """Parse one CSV into timecode seconds, box-centre poses and validity."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        next(stream)
        metadata = next(csv.reader([next(stream)]))
        rows = list(csv.DictReader(stream))
    frame = np.array([int(row["Frame"]) for row in rows])
    time = np.array([sum(float(x) * factor for x, factor in
                         zip(row["Timecode"].split(":"), (3600, 60, 1))) for row in rows])
    time += np.cumsum(np.r_[0, np.diff(time) < -43200]) * 86400  # clock wraps
    time -= time[0]
    position = np.array([[float(row[key]) for key in
                          ("PositionX", "PositionY", "PositionZ")] for row in rows])
    quaternion = np.array([[float(row[key]) for key in
                            ("quatX", "quatY", "quatZ", "quatW")] for row in rows])
    tracking = np.array([int(row["TrackingState"]) for row in rows])
    if (len(time) < 10 or np.any(np.diff(frame) <= 0) or np.any(np.diff(time) <= 0)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or np.any(np.abs(np.linalg.norm(quaternion, axis=1) - 1) > .01)):
        raise ValueError(f"{path}: invalid frames, timestamps, positions or quaternions")
    # Same per-frame conversions as the offline study: rigid-body pose composed
    # with the fixed T_DG (in the mocap frame), then the analysis basis change.
    rotation = Rotation.from_quat(quaternion).as_matrix() @ R_GD.T
    center = position - np.einsum("nij,j->ni", rotation, C_GD)
    center = center @ ANALYSIS_BASIS_M.T
    rotation = ANALYSIS_BASIS_M @ rotation
    return dict(time=time, center=center, rotation=rotation,
                valid=tracking == 8, fps=float(metadata[2]),
                file=path.name, rows=len(rows))


def _release_index(recording: dict) -> int:
    """Peak smoothed upward velocity + 10 ms, the offline candidate-start rule."""
    ids = np.flatnonzero(recording["valid"])
    smooth = savgol_filter(recording["center"][ids], 15, 3, axis=0)
    vertical_speed = np.gradient(smooth[:, 2], recording["time"][ids])
    apex = int(np.argmax(smooth[:, 2]))
    peak = int(np.argmax(vertical_speed[:apex + 1]))
    return int(np.searchsorted(recording["time"], recording["time"][ids[peak]] + .010))


class ReplayMocapSource:
    """Feeds one recorded throw to the runtime, in order, on the host clock."""

    def __init__(self, replay: ReplayConfig, chain: FrameChain | None = None):
        self._recording = load_recording(replay.csv)
        self._replay = replay
        ids = np.flatnonzero(self._recording["valid"])
        if replay.downsample_hz > 0:
            rate = replay.downsample_hz
            ticks = np.arange(self._recording["time"][ids[0]],
                              self._recording["time"][ids[-1]] + 1e-9, 1. / rate)
            # Previous captured pose only: never interpolate with future frames.
            ids = np.unique(ids[np.maximum(0, np.searchsorted(
                self._recording["time"][ids], ticks, side="right") - 1)])
        if replay.recenter_to_release:
            start = _release_index(self._recording)
            self._shift = np.array((self._recording["center"][start, 0],
                                    self._recording["center"][start, 1] + 1.5, 0.))
        else:
            self._shift = np.zeros(3)
        self._samples = [index for index in ids]
        self._cursor = 0
        self._chain = chain
        self._anchor_host = None
        self._anchor_device = self._recording["time"][self._samples[0]]
        self.info = dict(file=self._recording["file"], rows=self._recording["rows"],
                         samples=len(self._samples),
                         fps=self._recording["fps"], shift=self._shift.tolist())

    def next(self, timeout_s: float) -> Observation | None:
        if self._cursor >= len(self._samples):
            return None
        index = self._samples[self._cursor]
        device_t = float(self._recording["time"][index])
        if self._anchor_host is None:
            self._anchor_host = time.perf_counter()
        deadline = time.perf_counter() + max(0., timeout_s)
        # Always paced in real time: the runtime's latency accounting and the
        # planner's retiming assume the observation clock advances at wall rate.
        due = self._anchor_host + (device_t - self._anchor_device)
        now = time.perf_counter()
        if now < due:
            wait_until = min(due, deadline)
            if now < wait_until:
                time.sleep(wait_until - now)
            if time.perf_counter() < due:
                return None  # caller timeout elapsed before this sample is due
        position = self._recording["center"][index] - self._shift
        quaternion = Rotation.from_matrix(self._recording["rotation"][index]).as_quat()
        valid = bool(self._recording["valid"][index])
        if self._chain is not None:
            position, rotation = self._chain.box_geometry_pose(position, quaternion)
            quaternion = rotation.as_quat()
        self._cursor += 1
        return Observation(
            t_s=self._anchor_host + (device_t - self._anchor_device),
            position_m=np.asarray(position, dtype=float),
            quat_xyzw=np.asarray(quaternion, dtype=float),
            valid=valid,
            device_t_s=device_t,
        )

    def exhausted(self) -> bool:
        return self._cursor >= len(self._samples)
