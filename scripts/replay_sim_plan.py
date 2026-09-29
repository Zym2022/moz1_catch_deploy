#!/usr/bin/env python3
"""Replay a frozen simulation plan on the robot (no box) to verify the deploy chain.

Phase A is the standard joint-space approach to the ready posture
(scripts/move_to_ready.py logic, same guards as scripts/start_catch.py).  Phase
B loads a MozBoxer catch trace (see data/sim_plans/README.md), wraps the
recorded 1 ms palm-target series in a plan-like object and hands it to the
normal Executor - safety clamp, palm->TCP conversion, ROS2 message building and
tracing are exactly the live-catch code paths.  The trajectory is streamed at
the configured command rate (120 Hz), then the final pose is held until Ctrl+C.

Nothing here senses the box: prediction, commit timing and contact dynamics are
out of scope; this checks the execution/output side only.

Usage (robot host, ROS + movax_interface sourced, ROS_DOMAIN_ID=33; the
package is editable-installed - do NOT prefix PYTHONPATH=..., it clobbers
the ROS package paths setup.bash exported and rclpy stops resolving):
    .venv/bin/python scripts/replay_sim_plan.py --dry data/sim_plans/final_nominal_120hz_200ms.npz
    .venv/bin/python scripts/replay_sim_plan.py data/sim_plans/final_nominal_120hz_200ms.npz
    .venv/bin/python scripts/replay_sim_plan.py --speed-scale 0.5 --skip-approach <npz>
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.config import load_config
from moz1_catch.executor import Executor
from moz1_catch.feedback import JointFeedbackLog, fk_palm_series, tracking_summary
from moz1_catch.kinematics import parse_joints
from moz1_catch.robot.mock_sink import MockCartesianSink
from moz1_catch.safety import SafetyEnvelope
from moz1_catch.sim_plan import load_sim_trace_plan
from moz1_catch.trace import TraceRecorder

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config"

WAIT_POSE_TOLERANCE_M = 0.010
WAIT_POSE_TOLERANCE_RAD = np.deg2rad(5.)


def preflight(config, plan, info, log=print) -> bool:
    """Compare the plan's first target with the configured wait pose."""
    log(f"plan source      : {info['source']}")
    log(f"decision         : {info['decision']}  "
        f"({info['trace_samples']} samples @ {info['trace_dt_s'] * 1000:.0f} ms)")
    log(f"contact/stop     : {info['contact_time_s']:.3f} s / {info['stop_time_s']:.3f} s "
        f"(speed scale {info['speed_scale']})")
    log(f"max palm speed   : {info['max_palm_speed_mps']:.2f} m/s")
    positions, rotations, _, _ = plan.target(0.)
    position_error = max(
        float(np.linalg.norm(positions[hand] - config.robot.hands[hand].wait_position_m))
        for hand in range(2))
    angle_error = max(
        float((rotations[hand] * Rotation.from_quat(
            config.robot.hands[hand].wait_quat_xyzw).inv()).magnitude())
        for hand in range(2))
    log(f"wait pose check  : plan start vs FK wait pose -> "
        f"{position_error * 1000:.1f} mm / {np.degrees(angle_error):.2f} deg")
    if position_error > WAIT_POSE_TOLERANCE_M or angle_error > WAIT_POSE_TOLERANCE_RAD:
        log("WARNING: plan start differs from the configured wait pose - the "
            "first cartesian command after the handover would jump; check "
            "frames/URDF before running for real")
        return False
    return True


def record_feedback(config, feedback, commands, stamps, streamed, log=print) -> dict:
    """FK the recorded joints into base_link palm poses; return trace extras.

    This is what makes the replay answerable: commanded targets vs what the
    arms actually did, in the same frame, in the same trace.npz.  Tracking is
    evaluated on the execute phase only (the hold segments would dilute the
    lag estimate with static zero-error samples).
    """
    if feedback is None or not len(feedback):
        log("feedback         : none recorded (mock sink or silent /joint_states)")
        return {}
    joints = parse_joints(config.robot.posture.urdf)
    arrays = feedback.arrays()
    palm_position, palm_rotation = fk_palm_series(
        joints, config.robot.posture.legwaist_joint_deg,
        arrays["feedback_joint_left_rad"], arrays["feedback_joint_right_rad"])
    extras = {**arrays, "feedback_palm_position": palm_position,
              "feedback_palm_rotation_xyzw": palm_rotation}
    execute = np.asarray(commands["phase"]) == "execute"
    try:
        tracking = tracking_summary(stamps[execute], streamed[execute],
                                    arrays["feedback_t_s"], palm_position)
    except ValueError as error:
        log(f"feedback         : {len(feedback)} snapshots, "
            f"tracking not computable ({error})")
        return extras
    lag_left_ms = tracking["left"]["lag_s"] * 1000.
    lag_right_ms = tracking["right"]["lag_s"] * 1000.
    worst_mm = max(tracking[side]["error_max_mm"] for side in ("left", "right"))
    extras.update(tracking_lag_left_ms=lag_left_ms, tracking_lag_right_ms=lag_right_ms,
                  tracking_error_max_mm=worst_mm)
    log(f"feedback         : {len(feedback)} joint snapshots -> tracking "
        f"lag L {lag_left_ms:+.0f} / R {lag_right_ms:+.0f} ms, "
        f"max error {worst_mm:.1f} mm (see check_replay_tracking.py)")
    return extras


def run_replay(config, plan, info, hold_s: float, no_hold_final: bool, log=print) -> int:
    """Phase B: stream the plan through the Executor at the command rate."""
    feedback = None
    if config.sink_kind == "mock":
        sink = MockCartesianSink()
        log("using MOCK command sink (no hardware commands)")
    else:
        from moz1_catch.robot.ros2_sink import Ros2CartesianSink
        sink = Ros2CartesianSink(config.ros2)
        feedback = JointFeedbackLog()
        sink.attach_joint_feedback(feedback)
        log(f"ros2 cartesian sink on topic {config.ros2.cartesian_topic} "
            f"(recording /joint_states feedback)")
    trace = TraceRecorder(config)
    trace.event(f"sim plan replay: {Path(info['source']).name} "
                f"(speed scale {info['speed_scale']})")
    executor = Executor(sink, SafetyEnvelope(config.safety), config.execution,
                        config.robot, trace)
    period = config.execution.command_period_s
    start = time.perf_counter()

    try:
        next_tick = start
        log(f"holding wait pose for {hold_s:.1f} s ...")
        while time.perf_counter() - start < hold_s:
            executor.tick(time.perf_counter())
            next_tick += period
            time.sleep(max(0., next_tick - time.perf_counter()))

        log("replaying plan ...")
        arm_at = time.perf_counter()
        executor.arm_plan(plan, arm_at)
        next_tick = arm_at
        while executor.phase != "done":
            executor.tick(time.perf_counter())
            next_tick += period
            time.sleep(max(0., next_tick - time.perf_counter()))

        commands = trace.commands
        stamps = np.asarray(commands["t_host"])
        tick_periods = np.diff(stamps)
        violations = sum(1 for entry in commands["clamped"] if entry)
        streamed = np.asarray(commands["palm_position"]).reshape(-1, 2, 3)
        if len(streamed) > 1:
            speeds = np.linalg.norm(
                np.diff(streamed, axis=0)
                / np.maximum(np.diff(stamps)[:, None, None], 1e-9), axis=2)
        else:
            speeds = np.zeros(1)
        log(f"commands         : {len(stamps)} "
            f"(period p50 {np.median(tick_periods) * 1000:.1f} ms, "
            f"p95 {np.percentile(tick_periods, 95) * 1000:.1f} ms, "
            f"max {tick_periods.max() * 1000:.1f} ms)")
        log(f"clamp violations : {violations} (must be 0)")
        log(f"streamed speed   : {speeds.max():.2f} m/s peak")
        extra = record_feedback(config, feedback, commands, stamps, streamed, log=log)
        attempt_dir = trace.save(config.logging.output_dir, decision="replay",
                                 reason="sim_plan_replay", extra={**info, **extra})
        log(f"trace saved      : {attempt_dir}")

        if config.execution.hold_after_finish and not no_hold_final:
            try:
                executor.hold_final_forever(log=log)
            except KeyboardInterrupt:
                pass
        return 0 if violations == 0 else 1
    finally:
        close = getattr(sink, "close", None)
        if close is not None:
            close()


def approach(args, config, log=print) -> bool:
    """Phase A: joint-space move to the ready posture (start_catch.py logic)."""
    from move_to_ready import (JOINT_TOPIC, LEFT_PREFIX, RIGHT_PREFIX,
                               JointStateReader, deg_summary, enable_outer_control,
                               legwaist_check, move_arms_to_ready, slow_duration)
    posture = config.robot.posture
    target = {LEFT_PREFIX: np.deg2rad(posture.left_arm_joint_deg),
              RIGHT_PREFIX: np.deg2rad(posture.right_arm_joint_deg)}
    legwaist_ready = np.deg2rad(posture.legwaist_joint_deg)

    import rclpy
    from rosidl_runtime_py.utilities import get_message
    message_cls = get_message("mc_core_interface/msg/MechUnitCmdArray")

    rclpy.init()
    node = rclpy.create_node("moz1_replay_sim_plan")
    reader = JointStateReader(node)
    log(f"waiting for {JOINT_TOPIC} ...")
    if not reader.wait_for_arms(node):
        node.destroy_node(); rclpy.shutdown()
        raise SystemExit(f"no arm joints on {JOINT_TOPIC} within 10 s; "
                         f"names seen: {sorted(reader._latest)[:6]}")
    current = reader.arms()
    for prefix, label in ((LEFT_PREFIX, "left "), (RIGHT_PREFIX, "right ")):
        print(f"{label} current {deg_summary(current[prefix])}")
        print(f"{label} target  {deg_summary(target[prefix])}")
    delta = max(np.abs(target[p] - current[p]).max() for p in (LEFT_PREFIX, RIGHT_PREFIX))
    duration = slow_duration(delta, args.max_joint_speed, args.duration)
    log(f"max joint delta {np.degrees(delta):.1f} deg -> duration {duration:.1f} s "
        f"(peak speed {1.875 * delta / duration:.2f} rad/s)")
    if not legwaist_check(reader, legwaist_ready, args.legwaist_tolerance):
        node.destroy_node(); rclpy.shutdown()
        return False
    if args.enable_outer_ctrl:
        enable_outer_control(node)
    publisher = node.create_publisher(message_cls, "mx_mix_command", 5)
    time.sleep(.2)  # let the publisher discover before the first command
    arrived = move_arms_to_ready(node, publisher, message_cls, current, target,
                                 duration_s=duration, rate_hz=args.rate_hz,
                                 max_tracking_error_rad=args.max_tracking_error,
                                 feedback_timeout_s=args.feedback_timeout,
                                 hold_s=args.hold_ready_s, reader=reader)
    if not arrived:
        node.destroy_node(); rclpy.shutdown()
        return False
    settled = reader.arms()
    arrival = max(np.abs(settled[p] - target[p]).max()
                  for p in (LEFT_PREFIX, RIGHT_PREFIX))
    log(f"arrival error {np.degrees(arrival):.2f} deg "
        f"(tolerance {np.degrees(args.arrival_tolerance):.1f} deg)")
    ok = arrival <= args.arrival_tolerance
    node.destroy_node()          # phase A node done; context stays for phase B
    if not ok:
        rclpy.shutdown()
        log("ABORT: arms did not settle at the ready posture - no cartesian phase")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path,
                        help="simulation trace npz to replay (see data/sim_plans/)")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="live",
                        help="config profile for phase B (default live)")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="time scale: 0.5 replays at half speed (default 1.0)")
    parser.add_argument("--hold-s", type=float, default=1.0,
                        help="hold the wait pose this long before replaying, s")
    parser.add_argument("--dry", action="store_true",
                        help="mock sink, no joint-space approach, no final hold")
    parser.add_argument("--no-hold-final", action="store_true",
                        help="exit right after the replay instead of holding the final pose")
    parser.add_argument("--skip-approach", action="store_true",
                        help="robot is already at the ready posture; cartesian phase only")
    # --- phase A options (same contract as start_catch.py) ---
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-joint-speed", type=float, default=0.25)
    parser.add_argument("--rate-hz", type=float, default=120.)
    parser.add_argument("--max-tracking-error", type=float, default=0.2)
    parser.add_argument("--legwaist-tolerance", type=float, default=np.deg2rad(1.))
    parser.add_argument("--feedback-timeout", type=float, default=0.5)
    parser.add_argument("--hold-ready-s", type=float, default=1.0)
    parser.add_argument("--arrival-tolerance", type=float, default=np.deg2rad(1.))
    parser.add_argument("--enable-outer-ctrl", action="store_true")
    parser.add_argument("--dry-run-approach", action="store_true",
                        help="print the phase A plan and exit without moving")
    args = parser.parse_args(argv)

    config = load_config(args.config_dir, args.profile)
    if args.dry:
        config = replace(config, sink_kind="mock")
    plan, info = load_sim_trace_plan(args.plan, speed_scale=args.speed_scale)
    ok = preflight(config, plan, info)
    if not ok and not args.dry:
        return 1

    owns_rclpy = False
    if args.dry_run_approach and not args.dry:
        approach(args, config)   # prints the joint plan, moves nothing
        return 0
    if not args.dry and not args.skip_approach:
        if not approach(args, config):
            return 1
        owns_rclpy = True
    elif not args.dry and config.sink_kind != "mock":
        import rclpy
        rclpy.init()
        owns_rclpy = True

    try:
        return run_replay(config, plan, info, hold_s=args.hold_s,
                          no_hold_final=args.no_hold_final or args.dry)
    finally:
        if owns_rclpy:
            import rclpy
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
