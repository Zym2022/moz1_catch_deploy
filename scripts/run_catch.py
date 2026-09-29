#!/usr/bin/env python3
"""Run one live catch attempt: UDP mocap in, ROS2 cartesian targets out.

Usage on the robot host (see README.md for the full bring-up order):
    source /opt/ros/<distro>/setup.bash
    .venv/bin/python scripts/run_catch.py --profile live

Safety: the script publishes the configured wait pose until a throw is detected,
streams one planned trajectory after commit, then holds the final pose until
Ctrl+C.  Every placeholder in config/ must be filled in first; the script
refuses to start otherwise.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from moz1_catch.calib import FrameChain, MocapClock
from moz1_catch.config import load_config
from moz1_catch.executor import Executor
from moz1_catch.mocap.udp_source import UdpMocapSource
from moz1_catch.robot.mock_sink import MockCartesianSink
from moz1_catch.runtime import CatchRuntime
from moz1_catch.safety import SafetyEnvelope
from moz1_catch.trace import TraceRecorder

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config"


def build_sink(config, log):
    if config.sink_kind == "mock":
        log("using MOCK command sink (no hardware commands)")
        return MockCartesianSink()
    from moz1_catch.robot.ros2_sink import Ros2CartesianSink
    sink = Ros2CartesianSink(config.ros2)
    log(f"ros2 cartesian sink on topic {config.ros2.cartesian_topic}")
    return sink


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="live")
    parser.add_argument("--max-wait-s", type=float, default=60.0,
                        help="Give up if no throw starts within this window")
    parser.add_argument("--dry", action="store_true",
                        help="Force the mock sink even in the live profile")
    args = parser.parse_args(argv)

    config = load_config(args.config_dir, args.profile)
    if args.dry and config.sink_kind != "mock":
        config = replace(config, sink_kind="mock")
    log = print
    frame = FrameChain(T_FM=config.frames.T_FM, T_DG=config.frames.T_DG,
                       T_base_torso=config.robot.T_base_torso,
                       position_scale=1. if config.frames.mocap.position_units == "meters" else .001)
    clock = MocapClock(offset_s=config.frames.mocap.clock_offset_s)
    source = UdpMocapSource(config.udp, frame, clock,
                            config.frames.mocap.tracking_valid_states,
                            auto_anchor=config.frames.mocap.clock_mode == "auto", log=log)
    trace = TraceRecorder(config)
    sink = build_sink(config, log)
    executor = Executor(sink, SafetyEnvelope(config.safety), config.execution,
                        config.robot, trace)
    runtime = CatchRuntime(config, source, executor, trace, log=log)
    try:
        result = runtime.run(max_wait_s=args.max_wait_s)
        # The hold must publish through the LIVE sink, so it runs inside the
        # try and the finally closes the sink only afterwards (the ordering
        # replay_sim_plan.py uses; closing first crashed the hold's first
        # publish with rclpy InvalidHandle on the destroyed publisher).
        if config.execution.hold_after_finish and result.decision in ("accept", "reject"):
            executor.hold_final_forever(log=log)  # exits on Ctrl+C ("hold released")
    finally:
        source.close()
        close = getattr(sink, "close", None)
        if close is not None:
            close()
    log(f"attempt finished: decision={result.decision} reason={result.reason!r}")
    return 0 if result.decision == "accept" else 1


if __name__ == "__main__":
    sys.exit(main())
