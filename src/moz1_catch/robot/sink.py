"""Command sink interface and the palm-target-to-controller-flange conversion.

The planner emits, per hand, the pose of the PALM TARGET frame: origin at the
12-sphere array's front tangent-plane centre, axes along the hand link (see
core/geometry.py - one consistent rigid frame, the hand link translated by
PALM_CENTER_OFFSETS_BODY_M).  The hardware controller accepts poses of its
flange frame relative to torso_flange, so the output conversion composes two
fixed transforms in one place:

    T_torso_tcp = T_torso_base @ T_base_palm @ T_palm_tcp

  T_torso_base  : ^torso T_base, inverse of the fixed legwaist constant
                  T_base_torso from config/robot.toml (URDF FK value)
  T_base_palm   : the planner's palm-target pose (what plan.target() returns)
  T_palm_tcp    : T_tcp_palm from robot.toml, used as stored: ^palm T_tcp, the
                  flange expressed in the palm-target frame (NOT the hand-link
                  frame - the palm origin is offset by the tangent-plane centre;
                  and NOT inverted - the stored direction is palm -> flange)

Input-side conversions live in calib.FrameChain; this is the only output-side
one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.calib import as_transform, check_se3, transform_inverse


@dataclass(frozen=True)
class HandTargets:
    """Palm-target-frame poses for [left, right] in base_link."""

    positions_m: np.ndarray  # (2, 3)
    rotations: tuple[Rotation, Rotation]

    def __post_init__(self):
        positions = np.asarray(self.positions_m, dtype=float)
        if positions.shape != (2, 3) or not np.isfinite(positions).all():
            raise ValueError("hand target positions must be finite (2, 3)")
        object.__setattr__(self, "positions_m", positions)


class CartesianCommandSink(Protocol):
    """Receives one command per control period; lateness is the caller's problem."""

    def send(self, targets: HandTargets, t_host: float) -> None: ...


def palm_state_snapshot(positions: np.ndarray, quaternions_xyzw
                        ) -> tuple[np.ndarray, tuple[Rotation, Rotation]]:
    """Pack measured/configured palm poses into the plan_catch argument shape."""
    quaternions = np.asarray(quaternions_xyzw, dtype=float).reshape(2, 4)
    rotations = tuple(Rotation.from_quat(quaternions[side]) for side in range(2))
    return np.asarray(positions, dtype=float).reshape(2, 3), rotations


def palm_targets_to_tcp(targets: HandTargets, T_tcp_palms: tuple[np.ndarray, ...],
                        T_torso_base: np.ndarray | None = None
                        ) -> HandTargets:
    """Convert base_link palm-target poses into controller-flange poses in torso_flange."""
    if len(T_tcp_palms) != 2:
        raise ValueError("expected two hand transforms")
    if T_torso_base is None:
        T_torso_base = np.eye(4)
    T_torso_base = check_se3(T_torso_base, "T_torso_base")
    positions = np.empty((2, 3))
    rotations = []
    for side, T_tcp_palm in enumerate(T_tcp_palms):
        T_palm_tcp = check_se3(T_tcp_palm, "T_tcp_palm")  # stored palm -> flange
        T_base_palm = as_transform(targets.rotations[side].as_matrix(), targets.positions_m[side])
        T_torso_tcp = T_torso_base @ T_base_palm @ T_palm_tcp
        positions[side] = T_torso_tcp[:3, 3]
        rotations.append(Rotation.from_matrix(T_torso_tcp[:3, :3]))
    return HandTargets(positions_m=positions, rotations=tuple(rotations))
