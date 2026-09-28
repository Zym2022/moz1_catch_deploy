#!/usr/bin/env python3
"""Verify the config-derived robot constants against the URDF and the Isaac fixture.

Since the ready-posture joint angles became the single source of truth
(config/robot.toml [robot.posture]), this script no longer generates values to
paste - load_config derives wait poses, T_base_torso and T_tcp_palm via FK at
load time.  It remains the verification tool:

  * prints everything derived from the current config posture;
  * --check-base compares the wait poses with the Isaac fixture reference
    (expected residual: 0.002 deg rotation, 10.05 mm position = the
    2026-09-28 palm-target revision);
  * --joint-deg evaluates another posture without editing the config
    (20 angles in deg: LegWaist0-5, LeftArm0-6, RightArm0-6).

    uv run python scripts/compute_palm_frames.py --check-base
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.calib import transform_inverse
from moz1_catch.config import load_config
from moz1_catch.kinematics import parse_joints, robot_geometry

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config"

# Isaac fixture reference: start palm poses of the accepted nominal catch run
# (sim world = base_link aligned), recorded BEFORE the 2026-09-28 palm-target
# revision - hence the expected 10.05 mm position difference.
FIXTURE_WAIT = {
    "left": (np.array((0.19169, -0.62308, 1.1937)),
             np.array((-0.13962998, -0.10416850, -0.67948160, 0.71271113))),
    "right": (np.array((-0.19171, -0.62064, 1.20004)),
              np.array((0.14528112, 0.12739464, 0.71503513, -0.67185472))),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--joint-deg", type=float, nargs=20,
                        help="What-if posture (overrides the config values for this run only)")
    parser.add_argument("--check-base", action="store_true",
                        help="Compare derived wait poses with the Isaac fixture reference")
    args = parser.parse_args(argv)

    config = load_config(args.config_dir)
    if args.joint_deg is None:
        posture = config.robot.posture
        geometry = dict(T_base_torso=config.robot.T_base_torso,
                        hands={hand.name: dict(
                            wait_position_m=hand.wait_position_m,
                            wait_quat_xyzw=hand.wait_quat_xyzw,
                            T_palm_flange=hand.T_tcp_palm)
                            for hand in config.robot.hands})
    else:
        posture_args = (args.joint_deg[0:6], args.joint_deg[6:13], args.joint_deg[13:20])
        geometry = robot_geometry(parse_joints(config.robot.posture.urdf), *posture_args)
        print(f"(what-if posture: {list(args.joint_deg)})")

    print(f"urdf: {config.robot.posture.urdf}")
    print(f"posture (deg): legwaist={list(config.robot.posture.legwaist_joint_deg)}")
    T_base_torso = geometry["T_base_torso"]
    delta = Rotation.from_matrix(T_base_torso[:3, :3]).as_rotvec()
    print(f"T_base_torso translation (m): {T_base_torso[:3, 3].round(6).tolist()}, "
          f"rotation (deg): {np.rad2deg(delta).round(4).tolist()}")

    for side in ("left", "right"):
        derived = geometry["hands"][side]
        print(f"\n# {side}")
        print(f"wait_position_m = {derived['wait_position_m'].round(6).tolist()}")
        print(f"wait_quat_xyzw  = {derived['wait_quat_xyzw'].round(6).tolist()}")
        print(f"T_tcp_palm (^palm T_flange):\n{np.round(derived['T_palm_flange'], 6)}")
        normal = Rotation.from_quat(derived["wait_quat_xyzw"]).apply(
            (0., -1., 0.) if side == "left" else (0., 1., 0.))
        print(f"palm closing normal in base_link: {normal.round(4).tolist()}")

        if args.check_base:
            fixture_position, fixture_quat = FIXTURE_WAIT[side]
            rotation = Rotation.from_quat(derived["wait_quat_xyzw"])
            angle = (rotation * Rotation.from_quat(fixture_quat).inv()).magnitude()
            print(f"  fixture reference: {fixture_position.tolist()}")
            print(f"  position residual (mm): "
                  f"{1000*np.linalg.norm(derived['wait_position_m'] - fixture_position):.2f}, "
                  f"rotation residual (deg): {np.rad2deg(angle):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
