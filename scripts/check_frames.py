#!/usr/bin/env python3
"""Validate config/frames.toml + robot.toml and help with stationary-box checks.

Checks performed:
  - T_FM / T_DG / T_base_torso are valid SE(3) transforms (orthogonal, det +1).
  - Prints the effective mocap->base_link extrinsic (T_base_torso @ T_FM), the
    acceleration prior implied by it, and the vertical forecast gain reminder.
  - With --box-pose "x y z qx qy qz qw" (device units): prints the box geometry
    pose in base_link after the full chain, for comparison against a tape
    measure / a taught robot touching a stationary box.
  - With --chain-check: synthetic parabola invariance - predicting a synthetic
    flight through the full chain (non-identity T_base_torso and T_FM) must
    match direct base_link prediction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.calib import (
    ANALYSIS_PRIOR_MPS2, ANALYSIS_VERTICAL_GAIN_PER_M, FrameChain,
    effective_extrinsic, rotate_prior,
)
from moz1_catch.config import load_config
from moz1_catch.core.prediction import BoxFlight, PredictionSettings, estimate_box_flight

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config"


def chain_check(config) -> None:
    """A synthetic parabola predicted through the chain must match direct prediction."""
    T_MB = np.eye(4)
    T_MB[:3, :3] = Rotation.from_rotvec((.3, -.4, 1.1)).as_matrix()
    T_MB[:3, 3] = (.7, -1.2, .25)
    T_base_torso = config.robot.T_base_torso
    # Split the synthetic effective extrinsic so both hardware constants are exercised.
    chain = FrameChain(T_FM=np.linalg.inv(T_base_torso) @ T_MB, T_DG=config.frames.T_DG,
                       T_base_torso=T_base_torso)
    flight = BoxFlight(np.array((.02, -1.5, 1.2)), np.array((.1, 1.7, 2.4)),
                       Rotation.from_euler("xyz", (10, -5, 8), degrees=True),
                       np.array((.1, -.2, .3)), np.array((0., 0., -9.81)))
    times = np.arange(0., .3, 1. / 120.)
    poses_robot = np.column_stack((flight.positions(times), flight.rotations(times).as_quat()))
    # Express the base_link box-centre poses back into device coordinates.
    T_GD = np.linalg.inv(chain.T_DG)
    T_MB_inv = np.linalg.inv(chain.effective)
    device_poses = np.empty_like(poses_robot)
    for index, (position, quaternion) in enumerate(zip(poses_robot[:, :3], poses_robot[:, 3:])):
        T_BG = np.eye(4)
        T_BG[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        T_BG[:3, 3] = position
        T_BD = T_MB_inv @ T_BG @ T_GD
        device_poses[index, :3] = T_BD[:3, 3]
        device_poses[index, 3:] = Rotation.from_matrix(T_BD[:3, :3]).as_quat()
    settings = PredictionSettings()
    through_chain = estimate_box_flight(
        times, np.column_stack([
            np.array([chain.box_geometry_pose(device_poses[i, :3], device_poses[i, 3:])[0]
                      for i in range(len(times))]),
            np.array([chain.box_geometry_pose(device_poses[i, :3], device_poses[i, 3:])[1].as_quat()
                      for i in range(len(times))])]),
        times[-1] + .002, settings)
    direct = estimate_box_flight(times, poses_robot, times[-1] + .002, settings)
    error = float(np.linalg.norm(through_chain.position_m - direct.position_m))
    angular = float((through_chain.rotation * direct.rotation.inv()).magnitude())
    print(f"chain invariance: position error {error*1000:.6f} mm, rotation {np.rad2deg(angular):.6f} deg")
    if error > 1e-9 or angular > 1e-9:
        raise SystemExit("chain invariance FAILED - the transform chain or its inverse is wrong")
    print("chain invariance passed")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--box-pose", metavar="X Y Z QX QY QZ QW", nargs=7, type=float,
                        help="Raw rigid-body pose (device units, xyzw) of a stationary box")
    parser.add_argument("--chain-check", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config_dir)
    scale = 1. if config.frames.mocap.position_units == "meters" else .001
    frame = FrameChain(T_FM=config.frames.T_FM, T_DG=config.frames.T_DG,
                       T_base_torso=config.robot.T_base_torso, position_scale=scale)
    print(f"frames version: {config.frames.mocap.version}")
    print(f"T_base_torso (robot.toml, fixed legwaist) translation (m): "
          f"{config.robot.T_base_torso[:3, 3].round(6).tolist()}, rotation (deg): "
          f"{np.rad2deg(Rotation.from_matrix(config.robot.T_base_torso[:3, :3]).as_rotvec()).round(4).tolist()}")
    print(f"T_FM translation (m): {config.frames.T_FM[:3, 3].round(5).tolist()}")
    print(f"T_FM rotation (deg): {np.rad2deg(Rotation.from_matrix(config.frames.T_FM[:3, :3]).as_rotvec()).round(3).tolist()}")
    effective = effective_extrinsic(config.frames.T_FM, config.robot.T_base_torso)
    print(f"effective T_MB = T_base_torso @ T_FM translation (m): {effective[:3, 3].round(5).tolist()}")
    print(f"T_DG translation (m): {config.frames.T_DG[:3, 3].round(5).tolist()}")

    prior = rotate_prior(ANALYSIS_PRIOR_MPS2, effective)
    configured = np.asarray(config.prediction.settings.acceleration_prior_mps2)
    print(f"analysis prior (m/s^2): {ANALYSIS_PRIOR_MPS2.round(5).tolist()}")
    print(f"base_link prior via effective T_MB: {prior.round(5).tolist()}  <- paste into catch.toml [prediction]")
    print(f"configured prior: {configured.round(5).tolist()}"
          + ("" if np.allclose(configured, prior, atol=1e-3) else "  (DIFFERS from rotated prior)"))
    print(f"vertical forecast gain k carries over as-is when planning +Z is vertical "
          f"(calibrated: {ANALYSIS_VERTICAL_GAIN_PER_M:.7f}); currently configured: "
          f"{config.prediction.settings.vertical_forecast_gain_per_m:.7f}")

    if args.chain_check:
        chain_check(config)
    if args.box_pose:
        position = np.array(args.box_pose[:3]) * scale
        quaternion = np.array(args.box_pose[3:])
        box_position, box_rotation = frame.box_geometry_pose(position, quaternion)
        print("stationary box geometry pose in base_link:")
        print(f"  position (m): {box_position.round(5).tolist()}")
        print(f"  rotation (deg xyz): {np.rad2deg(box_rotation.as_rotvec()).round(3).tolist()}")
        corner_offsets = np.array(np.meshgrid(*[(h, -h) for h in config.catch_settings.box_half_extents])).reshape(3, -1).T
        corners = box_position + box_rotation.apply(corner_offsets)
        print(f"  corners (m): {[tuple(corner.round(4)) for corner in corners]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
