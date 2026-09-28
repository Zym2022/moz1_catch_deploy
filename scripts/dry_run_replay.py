#!/usr/bin/env python3
"""Replay one recorded throw through the live runtime with a mock sink.

Stage-2 dry run: exercises arming, release detection, commit, planning and the
command stream exactly as in a live attempt, but the "controller" is an in-memory
recorder and the mocap comes from a CSV.  Playback runs in real time.  After the
run, prints where the trace was saved so target tracking can be inspected.

Examples:
    .venv/bin/python scripts/dry_run_replay.py --csv data/box_flying_csv/1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from moz1_catch.config import ReplayConfig, load_config
from moz1_catch.executor import Executor
from moz1_catch.mocap.replay_source import ReplayMocapSource
from moz1_catch.robot.mock_sink import MockCartesianSink
from moz1_catch.runtime import CatchRuntime
from moz1_catch.safety import SafetyEnvelope
from moz1_catch.trace import TraceRecorder

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--native", action="store_true", help="Do not downsample to 120 Hz")
    args = parser.parse_args(argv)

    config = load_config(args.config_dir, "replay")
    from dataclasses import replace
    config = replace(config, replay=replace(
        config.replay if config.replay else ReplayConfig(csv=args.csv, downsample_hz=120.,
                                                          recenter_to_release=True),
        csv=args.csv, downsample_hz=0. if args.native else 120.))
    print(f"replay source: {ReplayMocapSource.__name__} csv={args.csv}")
    source = ReplayMocapSource(config.replay)
    print(f"recording info: {source.info}")
    trace = TraceRecorder(config)
    sink = MockCartesianSink()
    executor = Executor(sink, SafetyEnvelope(config.safety), config.execution,
                        config.robot, trace)
    runtime = CatchRuntime(config, source, executor, trace)
    result = runtime.run(max_wait_s=120.)
    print(f"commands published: {sink.count}")
    if sink.count:
        positions = np.asarray(sink.sent_positions)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
        print(f"command step mm per publish: max={1000*steps.max():.2f} mean={1000*steps.mean():.2f}")
    print(f"result: {result}")
    return 0 if result.decision == "accept" else 1


if __name__ == "__main__":
    sys.exit(main())
