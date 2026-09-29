#!/usr/bin/env python3
"""Move both arms from their current joints to the catch ready posture - in JOINT space.

Why this exists: cartesian streaming (what scripts/run_catch.py does from its
first control period) started from an arbitrary arm configuration lets the
controller IK drag the arms to the wait pose through whatever solution branch
it picks - potentially singular and uncontrollable.  This module first moves
both arms to the ready-posture JOINT angles (config/robot.toml
[robot.posture], the same single source of truth the wait poses, T_base_torso
and the palm->flange constants are derived from), so any cartesian phase that
follows starts from exactly the configuration its wait pose was built for.

scripts/start_catch.py composes this with the catch bridge into one command;
this file remains runnable standalone for positioning without catching.

Motion contract (deliberately slow):
  * quintic point-to-point profile, zero velocity AND acceleration at both
    ends, so the arms ease in and out;
  * the duration is chosen so the PEAK joint speed stays under
    --max-joint-speed (default 0.25 rad/s ~ 14 deg/s) with a hard 4 s floor;
    the plan (deltas, duration, peak speed) is printed before anything moves;
  * ARMS ONLY - the legwaist is never commanded; it is checked against the
    locked ready posture and a mismatch beyond --legwaist-tolerance aborts,
    because every catch-frame constant assumes it;
  * aborts (stops publishing) if tracking error against /joint_states exceeds
    --max-tracking-error, or if feedback stops arriving mid-motion;
  * --dry-run reads and prints everything and publishes nothing;
  * Ctrl+C stops publishing immediately; the controller holds the last command.

Operational note: like the teleop bridge, mc_core may ignore mix commands
until outer control is enabled; pass --enable-outer-ctrl to call the
robot_cmd_service first (EnableOuterCtrl).

Usage (robot host, ROS + movax_interface sourced, ROS_DOMAIN_ID=33; the
package is editable-installed - no PYTHONPATH prefix, it would clobber the
sourced ROS package paths and rclpy would stop resolving):
    .venv/bin/python scripts/move_to_ready.py --dry-run
    .venv/bin/python scripts/move_to_ready.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config"
JOINT_TOPIC = "/joint_states"
LEGWAIST_PREFIX, LEFT_PREFIX, RIGHT_PREFIX = "LegWaist-", "LeftArm-", "RightArm-"
DEFAULT_PEAK_SPEED_RAD_S = 0.25   # ~14 deg/s peak; the default slow contract
MIN_DURATION_S = 4.0              # hard floor however small the delta is


def quintic_position(fraction: float) -> float:
    """Zero-vel/acc start-and-end profile: s = 10t^3 - 15t^4 + 6t^5, s'(1/2)=1.875."""
    t = min(max(fraction, 0.), 1.)
    return t * t * t * (10. + t * (-15. + 6. * t))


def slow_duration(delta_rad: float, peak_speed_rad_s: float,
                  requested_s: float | None = None) -> float:
    """Duration guaranteeing the quintic's peak speed stays under the cap."""
    if requested_s is not None:
        return max(MIN_DURATION_S, requested_s)
    return max(MIN_DURATION_S, 1.875 * delta_rad / peak_speed_rad_s)


def deg_summary(values_rad) -> str:
    return "(" + ", ".join(f"{math.degrees(v):+.1f}" for v in values_rad) + ") deg"


class JointStateReader:
    """Holds one /joint_states subscription; exposes named joint snapshots."""

    def __init__(self, node):
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        self._latest: dict[str, float] = {}
        self._stamp: float | None = None

        def on_message(message: JointState) -> None:
            self._latest.update(dict(zip(message.name, message.position)))
            self._stamp = time.perf_counter()

        node.create_subscription(JointState, JOINT_TOPIC, on_message,
                                 QoSProfile(depth=10,
                                            reliability=ReliabilityPolicy.BEST_EFFORT))

    def wait_for_arms(self, node, timeout_s: float = 10.) -> bool:
        deadline = time.perf_counter() + timeout_s
        while not self.has_arms():
            import rclpy
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.perf_counter() > deadline:
                return False
        return True

    def has_arms(self) -> bool:
        return (LEFT_PREFIX + "6") in self._latest and (RIGHT_PREFIX + "6") in self._latest

    def arms(self) -> dict[str, np.ndarray]:
        if not self.has_arms():
            raise RuntimeError("no arm joints received yet")
        return {prefix: np.array([self._latest[prefix + str(i)] for i in range(7)])
                for prefix in (LEFT_PREFIX, RIGHT_PREFIX)}

    def legwaist(self) -> np.ndarray:
        return np.array([self._latest.get(LEGWAIST_PREFIX + str(i), float("nan"))
                         for i in range(6)])

    def age_s(self) -> float | None:
        return None if self._stamp is None else time.perf_counter() - self._stamp


def legwaist_check(reader: JointStateReader, ready_rad: np.ndarray,
                   tolerance_rad: float, log=print) -> bool:
    measured = reader.legwaist()
    if not np.isfinite(measured).all() or np.abs(measured - ready_rad).max() > tolerance_rad:
        log(f"ABORT: legwaist is not at the locked ready posture "
            f"{deg_summary(ready_rad)}; measured {deg_summary(measured)}. "
            f"Every catch-frame constant assumes it - move it there first.")
        return False
    return True


def move_arms_to_ready(node, publisher, message_cls, current: dict[str, np.ndarray],
                       target: dict[str, np.ndarray], *, duration_s: float,
                       rate_hz: float = 120., max_tracking_error_rad: float = 0.2,
                       feedback_timeout_s: float = 0.5, hold_s: float = 1.0,
                       reader: JointStateReader | None = None, log=print) -> bool:
    """Run the quintic joint move; returns True when both arms end at target.

    The caller owns the rclpy node/context and the publisher; this function
    only pumps callbacks, publishes the profile and enforces the guards.
    """
    from moz1_catch.robot.ros2_sink import build_joint_message
    start = {prefix: current[prefix].copy() for prefix in (LEFT_PREFIX, RIGHT_PREFIX)}
    peak = max(1.875 * np.abs(target[p] - start[p]).max() / duration_s
               for p in (LEFT_PREFIX, RIGHT_PREFIX))
    log(f"moving: duration {duration_s:.1f} s, peak joint speed {peak:.2f} rad/s "
        f"({math.degrees(peak):.0f} deg/s), {rate_hz:.0f} Hz - Ctrl+C aborts")
    anchor = time.perf_counter()
    next_command, period = 0., 1. / rate_hz
    measured = dict(current)
    try:
        while True:
            now = time.perf_counter()
            elapsed = now - anchor
            import rclpy
            rclpy.spin_once(node, timeout_sec=0.)
            if elapsed > duration_s + hold_s:
                break
            if reader is not None and elapsed > 0.05:
                age = reader.age_s()
                if age is not None and age > feedback_timeout_s:
                    log("ABORT: /joint_states went silent mid-motion")
                    return False
            if now < next_command:
                continue
            next_command = now + period
            fraction = quintic_position(elapsed / duration_s)
            commanded = {prefix: start[prefix] + (target[prefix] - start[prefix]) * fraction
                         for prefix in (LEFT_PREFIX, RIGHT_PREFIX)}
            publisher.publish(build_joint_message(message_cls, commanded[LEFT_PREFIX],
                                                  commanded[RIGHT_PREFIX]))
            if reader is not None:
                latest = reader._latest
                measured = {prefix: np.array([latest.get(prefix + str(i), commanded[prefix][i])
                                              for i in range(7)])
                            for prefix in (LEFT_PREFIX, RIGHT_PREFIX)}
                tracking = max(np.abs(measured[p] - commanded[p]).max()
                               for p in (LEFT_PREFIX, RIGHT_PREFIX))
                if elapsed > 0.2 and tracking > max_tracking_error_rad:
                    log(f"ABORT: tracking error {math.degrees(tracking):.1f} deg "
                        f"(cap {math.degrees(max_tracking_error_rad):.1f} deg)")
                    return False
        log(f"done: arms at the ready posture "
            f"[L {deg_summary(measured[LEFT_PREFIX])}, "
            f"R {deg_summary(measured[RIGHT_PREFIX])}]")
        return True
    except KeyboardInterrupt:
        log("\ninterrupted - publishing stopped, controller holds the last command")
        return False


def enable_outer_control(node, log=print) -> None:
    """Call robot_cmd_service EnableOuterCtrl (mirrors the teleop bring-up)."""
    from mc_core_interface.srv import RobotCmdService
    client = node.create_client(RobotCmdService, "robot_cmd_service")
    if not client.wait_for_service(timeout_sec=2.):
        log("warning: robot_cmd_service not available, continuing anyway")
        return
    request = RobotCmdService.Request()
    request.cmd, request.data = "EnableOuterCtrl", '{"enable": true}'
    future = client.call_async(request)
    import rclpy
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.)
    log(f"EnableOuterCtrl -> {future.result().data if future.done() else 'timeout'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, default=None,
                        help="total motion time in s; default: derived from --max-joint-speed")
    parser.add_argument("--max-joint-speed", type=float, default=DEFAULT_PEAK_SPEED_RAD_S,
                        help=f"peak joint speed cap in rad/s (default {DEFAULT_PEAK_SPEED_RAD_S})")
    parser.add_argument("--rate-hz", type=float, default=120.,
                        help="command publication rate (controller standard: 120 Hz)")
    parser.add_argument("--max-tracking-error", type=float, default=0.2,
                        help="abort when any arm joint deviates this far from the command, rad")
    parser.add_argument("--legwaist-tolerance", type=float, default=math.radians(1.),
                        help="abort when the locked legwaist deviates this far, rad")
    parser.add_argument("--feedback-timeout", type=float, default=0.5,
                        help="abort when /joint_states is silent this long mid-motion, s")
    parser.add_argument("--hold-s", type=float, default=1.0,
                        help="keep publishing the final joints this long before exiting")
    parser.add_argument("--dry-run", action="store_true",
                        help="read and print the plan, publish nothing")
    parser.add_argument("--enable-outer-ctrl", action="store_true",
                        help="call robot_cmd_service EnableOuterCtrl before moving")
    args = parser.parse_args(argv)

    from moz1_catch.config import load_config
    posture = load_config(args.config_dir).robot.posture
    target = {LEFT_PREFIX: np.deg2rad(posture.left_arm_joint_deg),
              RIGHT_PREFIX: np.deg2rad(posture.right_arm_joint_deg)}
    legwaist_ready = np.deg2rad(posture.legwaist_joint_deg)

    import rclpy
    from rosidl_runtime_py.utilities import get_message
    message_cls = get_message("mc_core_interface/msg/MechUnitCmdArray")

    rclpy.init()
    node = rclpy.create_node("moz1_move_to_ready")
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
        print("dry run: no command published")
        node.destroy_node(); rclpy.shutdown()
        return 0

    if args.enable_outer_ctrl:
        enable_outer_control(node)

    publisher = node.create_publisher(message_cls, "mx_mix_command", 5)
    time.sleep(.2)  # let the publisher discover before the first command
    ok = move_arms_to_ready(node, publisher, message_cls, current, target,
                            duration_s=duration, rate_hz=args.rate_hz,
                            max_tracking_error_rad=args.max_tracking_error,
                            feedback_timeout_s=args.feedback_timeout,
                            hold_s=args.hold_s, reader=reader)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
