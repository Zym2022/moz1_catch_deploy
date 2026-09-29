#!/usr/bin/env python3
"""One-command catch launcher: slow joint-space approach, then the live bridge.

Phase A moves both arms from wherever they are to the catch ready posture
THROUGH JOINT SPACE (scripts/move_to_ready.py logic): a quintic profile with
zero start/end velocity and acceleration, peak joint speed capped by
--max-joint-speed (default 0.25 rad/s ~ 14 deg/s) and a hard 4 s duration
floor - the plan is printed before anything moves.  Joint space avoids the
singularity and controllability problems of driving an arbitrary start
configuration through cartesian targets.

Phase B starts the catch bridge (scripts/run_catch.py) in this same process.
The handover is continuous by construction: phase A ends exactly at the
ready-posture joints, and the wait pose phase B streams first is the forward
kinematics of those same joints - no jump.  Arrival is verified (within
--arrival-tolerance) before the cartesian phase is allowed to start.

Both phases share one rclpy context and publish on the same mx_mix_command
topic; phase A commands jnt_pos (use_jnt=true), phase B commands end_pose.
Guards carried over from move_to_ready: legwaist must already be locked at
the ready posture (checked, never commanded), tracking-error and
feedback-silence aborts, Ctrl+C stops publishing immediately.

Usage (robot host, ROS_DOMAIN_ID=33, ROS + movax_interface sourced):
    PYTHONPATH="src:$PYTHONPATH" python3 scripts/start_catch.py --dry-run
    PYTHONPATH="src:$PYTHONPATH" python3 scripts/start_catch.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

# scripts/ is on sys.path when this file runs as a script; reuse the mover.
from move_to_ready import (DEFAULT_PEAK_SPEED_RAD_S, JOINT_TOPIC,
                           LEFT_PREFIX, RIGHT_PREFIX, JointStateReader,
                           deg_summary, enable_outer_control, legwaist_check,
                           move_arms_to_ready, slow_duration)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- phase A: joint-space approach (same contract as move_to_ready.py) ---
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, default=None,
                        help="phase A motion time in s; default: derived from --max-joint-speed")
    parser.add_argument("--max-joint-speed", type=float, default=DEFAULT_PEAK_SPEED_RAD_S,
                        help=f"peak joint speed cap in rad/s (default {DEFAULT_PEAK_SPEED_RAD_S})")
    parser.add_argument("--rate-hz", type=float, default=100.,
                        help="phase A command publication rate")
    parser.add_argument("--max-tracking-error", type=float, default=0.2,
                        help="abort when any arm joint deviates this far from the command, rad")
    parser.add_argument("--legwaist-tolerance", type=float, default=math.radians(1.),
                        help="abort when the locked legwaist deviates this far, rad")
    parser.add_argument("--feedback-timeout", type=float, default=0.5,
                        help="abort when /joint_states is silent this long mid-motion, s")
    parser.add_argument("--hold-s", type=float, default=1.0,
                        help="hold the final joints this long before the handover")
    parser.add_argument("--arrival-tolerance", type=float, default=math.radians(1.),
                        help="phase B starts only when both arms are this close to target, rad")
    parser.add_argument("--enable-outer-ctrl", action="store_true",
                        help="call robot_cmd_service EnableOuterCtrl before phase A")
    parser.add_argument("--dry-run", action="store_true",
                        help="read and print the plan, move nothing, start no bridge")
    # --- phase B: catch bridge (passed through to run_catch.main) ---
    parser.add_argument("--profile", default="live",
                        help="catch runtime profile for phase B (default live)")
    parser.add_argument("--max-wait-s", type=float, default=60.0,
                        help="phase B gives up if no throw starts within this window")
    args = parser.parse_args(argv)

    from moz1_catch.config import load_config
    posture = load_config(args.config_dir).robot.posture
    target = {LEFT_PREFIX: np.deg2rad(posture.left_arm_joint_deg),
              RIGHT_PREFIX: np.deg2rad(posture.right_arm_joint_deg)}
    legwaist_ready = np.deg2rad(posture.legwaist_joint_deg)

    import rclpy
    from rosidl_runtime_py.utilities import get_message
    message_cls = get_message("mc_core_interface/msg/MechUnitCmdArray")

    # One rclpy context for both phases; the cartesian sink reuses it (its
    # ownership guard skips re-init) and this script tears it down at the end.
    rclpy.init()
    node = rclpy.create_node("moz1_start_catch")
    reader = JointStateReader(node)
    print(f"waiting for {JOINT_TOPIC} ...")
    if not reader.wait_for_arms(node):
        node.destroy_node(); rclpy.shutdown()
        raise SystemExit(f"no arm joints on {JOINT_TOPIC} within 10 s; "
                         f"names seen: {sorted(reader._latest)[:6]}")

    current = reader.arms()
    for prefix, label in ((LEFT_PREFIX, "left "), (RIGHT_PREFIX, "right")):
        print(f"{label} current {deg_summary(current[prefix])}")
        print(f"{label} target  {deg_summary(target[prefix])}")
    delta = max(np.abs(target[p] - current[p]).max() for p in (LEFT_PREFIX, RIGHT_PREFIX))
    duration = slow_duration(delta, args.max_joint_speed, args.duration)
    print(f"max joint delta {math.degrees(delta):.1f} deg -> duration {duration:.1f} s "
          f"(peak speed {1.875 * delta / duration:.2f} rad/s)")
    if not legwaist_check(reader, legwaist_ready, args.legwaist_tolerance):
        node.destroy_node(); rclpy.shutdown()
        return 1
    if args.dry_run:
        print("dry run: no motion, no bridge")
        node.destroy_node(); rclpy.shutdown()
        return 0

    if args.enable_outer_ctrl:
        enable_outer_control(node)
    publisher = node.create_publisher(message_cls, "mx_mix_command", 5)
    time.sleep(.2)  # let the publisher discover before the first command
    arrived = move_arms_to_ready(node, publisher, message_cls, current, target,
                                 duration_s=duration, rate_hz=args.rate_hz,
                                 max_tracking_error_rad=args.max_tracking_error,
                                 feedback_timeout_s=args.feedback_timeout,
                                 hold_s=args.hold_s, reader=reader)
    if not arrived:
        node.destroy_node(); rclpy.shutdown()
        return 1

    # Arrival gate: never hand over to cartesian streaming unless both arms
    # sit at the ready joints (whose FK is exactly the wait pose).
    settled = reader.arms()
    arrival = max(np.abs(settled[p] - target[p]).max()
                  for p in (LEFT_PREFIX, RIGHT_PREFIX))
    print(f"arrival error {math.degrees(arrival):.2f} deg "
          f"(tolerance {math.degrees(args.arrival_tolerance):.1f} deg)")
    if arrival > args.arrival_tolerance:
        print("ABORT: arms did not settle at the ready posture - no cartesian phase")
        node.destroy_node(); rclpy.shutdown()
        return 1
    node.destroy_node()          # phase A node done; context stays for phase B

    import run_catch
    print("=== phase B: catch bridge (cartesian streaming from the wait pose) ===")
    code = run_catch.main(["--config-dir", str(args.config_dir),
                           "--profile", args.profile,
                           "--max-wait-s", str(args.max_wait_s)])
    rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
