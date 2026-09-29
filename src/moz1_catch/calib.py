"""Frame chain and clock mapping applied to raw mocap samples.

Frame discipline (deliberately conservative - the planner never changes frames):

  * The PLANNING frame is base_link, unchanged from the simulation.  Mission
    geometry, wait poses, workspace boxes and the replay data all keep their
    existing base_link numbers.
  * The hardware touches two other frames, each converted at exactly one place
    with its own explicit constant:
      - input:  the mocap extrinsic is calibrated to torso_flange (T_FM,
        hand-eye).  Samples are converted mocap -> torso_flange -> base_link,
        i.e. through the fixed legwaist constant T_base_torso.
      - output: palm targets are converted base_link -> torso_flange (the
        inverse of the same constant) -> controller flange, inside
        palm_targets_to_tcp.

Per-sample input chain (column vectors, metres):

    T_BG = T_base_torso @ T_FM @ T_MD @ T_DG

  T_MD         : box rigid body pose in the mocap global frame, as delivered
  T_FM         : mocap global -> torso_flange, from the Park hand-eye calibration
  T_base_torso : ^base T_torso, fixed while legwaist is locked (URDF FK value)
  T_DG         : box geometry frame from the box rigid body (fixed installation)

The effective mocap->base rotation for the acceleration prior is
R_base_torso @ R_FM (see rotate_prior): the analysis-frame prior and the
deployed prior are the same physical vector expressed in different frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

# Analysis frame used by the mocap studies: +X=MZ, +Y=MX, +Z=MY.
ANALYSIS_BASIS_M = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
# Calibrated on all 34 evaluable throws of the 36-record study (2026-09-29,
# MozBoxer commit f603888); see that repo's calibration.json.
ANALYSIS_PRIOR_MPS2 = np.array((-0.011349310646618586, -0.3868940737031952, -8.791483013796956))
# Empirical vertical forecast gain k in a_forecast_z = a_fit_z - k*v_z*|v|
# (same calibration).  Carries over to any planning frame whose +Z is the
# same vertical direction; 0 disables the correction.
ANALYSIS_VERTICAL_GAIN_PER_M = 0.06827448239423729


def as_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix


def transform_inverse(matrix: np.ndarray) -> np.ndarray:
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    return as_transform(rotation.T, -rotation.T @ translation)


def check_se3(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} is not a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0., 0., 0., 1.), atol=1e-9):
        raise ValueError(f"{name} has a non-standard bottom row")
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError(f"{name} rotation is not in SO(3)")
    return matrix


def effective_extrinsic(T_FM: np.ndarray, T_base_torso: np.ndarray) -> np.ndarray:
    """Mocap global -> base_link as one matrix: T_base_torso @ T_FM."""
    return check_se3(T_base_torso, "T_base_torso") @ check_se3(T_FM, "T_FM")


def rotate_prior(prior_analysis_mps2: np.ndarray, T_MB: np.ndarray) -> np.ndarray:
    """Express an analysis-frame acceleration prior in base_link.

    a_base = R_MB @ a_mocap with a_mocap = ANALYSIS_BASIS_M.T @ a_analysis,
    and R_MB the effective mocap->base rotation, i.e. R_base_torso @ R_FM
    (pass effective_extrinsic(T_FM, T_base_torso)).
    """
    rotation = check_se3(T_MB, "T_MB")[:3, :3]
    return rotation @ (ANALYSIS_BASIS_M.T @ np.asarray(prior_analysis_mps2, dtype=float))


@dataclass(frozen=True)
class FrameChain:
    """Converts one raw mocap rigid-body pose into the base_link planning frame."""

    T_FM: np.ndarray                 # mocap global -> torso_flange (hand-eye)
    T_DG: np.ndarray                 # box geometry <- box rigid body (fixed)
    T_base_torso: np.ndarray = field(default_factory=lambda: np.eye(4))
    position_scale: float = 1.0      # device units -> metres

    def __post_init__(self):
        object.__setattr__(self, "T_FM", check_se3(self.T_FM, "T_FM"))
        object.__setattr__(self, "T_DG", check_se3(self.T_DG, "T_DG"))
        object.__setattr__(self, "T_base_torso", check_se3(self.T_base_torso, "T_base_torso"))
        if not (np.isfinite(self.position_scale) and self.position_scale > 0):
            raise ValueError("invalid position scale")

    @property
    def effective(self) -> np.ndarray:
        """Mocap global -> base_link as one matrix (for reporting and priors)."""
        return effective_extrinsic(self.T_FM, self.T_base_torso)

    def box_geometry_pose(self, position_units: np.ndarray, quaternion_xyzw: np.ndarray
                          ) -> tuple[np.ndarray, Rotation]:
        """Map a raw box rigid-body pose (T_MD parts) to a box-centre pose in base_link."""
        quaternion = np.asarray(quaternion_xyzw, dtype=float)
        if quaternion.shape != (4,):
            raise ValueError("expected one xyzw quaternion")
        R_MD = Rotation.from_quat(quaternion / np.linalg.norm(quaternion))
        T_MD = as_transform(R_MD.as_matrix(), np.asarray(position_units, dtype=float) * self.position_scale)
        T_BG = self.effective @ T_MD @ self.T_DG
        return T_BG[:3, 3].copy(), Rotation.from_matrix(T_BG[:3, :3])


@dataclass(frozen=True)
class MocapClock:
    """Maps device timestamps onto the runtime host monotonic clock."""

    offset_s: float = 0.0  # t_host = t_device + offset_s

    def to_host(self, t_device: float) -> float:
        return float(t_device) + self.offset_s


class ArrivalClockAnchor:
    """Device->host clock offset measured from datagram arrival times.

    The runtime mixes two clocks: mocap samples carry device timestamps
    (opti_fTimestamp seconds-of-day) while planning, execution and watchdog
    timing run on time.perf_counter().  The offset is the difference between
    the two readings of the SAME instant - a pure bookkeeping constant, not a
    latency.  Measuring it per run makes it immune to host reboots
    (perf_counter resets), mocap clock jumps and the midnight seconds-of-day
    wrap:

      * the first packet received after the socket backlog is drained provides
        a provisional offset - its arrival time stands in for its capture
        instant, absorbing the sub-millisecond transport latency;
      * the median over the first refine_packets arrivals (or max_refine_s of
        host time, whichever closes the window first) replaces it exactly
        once; the shift is well under a millisecond and lands long before the
        operator can arm a throw.

    The anchor is then frozen; every trace records device_t_s next to t_s, so
    the applied offset stays auditable offline.
    """

    def __init__(self, fallback_offset_s: float = 0.0, refine_packets: int = 120,
                 max_refine_s: float = 2.0, log=None):
        if refine_packets < 1 or max_refine_s <= 0:
            raise ValueError("invalid clock anchoring window")
        self._offset_s = float(fallback_offset_s)
        self._refine_packets = int(refine_packets)
        self._max_refine_s = float(max_refine_s)
        self._log = log
        self._diffs: list[float] | None = None
        self._started_s = 0.0
        self._installed = False

    @property
    def offset_s(self) -> float:
        """Current best device->host offset (the fallback until first arrival)."""
        return self._offset_s

    @property
    def anchored(self) -> bool:
        """True once any arrival has replaced the fallback offset."""
        return self._diffs is not None

    def update(self, t_device: float, t_arrival_host: float) -> bool:
        """Feed the device timestamp and host arrival time of one packet.

        Returns True when offset_s changed (the provisional anchor, or the
        single refinement install); False while collecting or after freezing.
        """
        diff = t_arrival_host - t_device
        if self._diffs is None:
            self._offset_s = diff
            self._diffs = [diff]
            self._started_s = t_arrival_host
            self._announce(f"mocap_clock_anchor=provisional offset_s={diff:+.6f}")
            return True
        if not self._installed and not self._window_open(t_arrival_host):
            return self._install()
        if self._installed:
            return False
        self._diffs.append(diff)
        if not self._window_open(t_arrival_host):
            return self._install()
        return False

    def _window_open(self, t_arrival_host: float) -> bool:
        return (len(self._diffs) < self._refine_packets
                and t_arrival_host - self._started_s < self._max_refine_s)

    def _install(self) -> bool:
        self._installed = True
        refined = float(np.median(self._diffs))
        shift = refined - self._offset_s
        self._offset_s = refined
        self._announce(f"mocap_clock_anchor=refined offset_s={refined:+.6f} "
                       f"samples={len(self._diffs)} shift_ms={1000. * shift:+.3f}")
        return True

    def _announce(self, message: str) -> None:
        if self._log is not None:
            self._log(message)
