#!/usr/bin/env python3
"""Command-vs-actual tracking report for a sim-plan replay attempt.

Reads an attempt directory written by scripts/replay_sim_plan.py (the latest
output/attempt_*_replay by default) and compares the commanded palm targets
with the palm poses reconstructed from the recorded /joint_states feedback:
per-hand position error, the constant lag that best explains it (positive =
the arms trail the command) and the residual after removing that lag.  This
is the quantitative answer to the guide's step-3 question - "can the
controller track the 120 Hz pose stream, and by how much does it lag?".
Traces recorded before joint feedback existed (2026-09-29) have no
feedback_* keys and are reported as such.

Run on the robot host or the dev machine; no ROS needed - everything the
analysis reads is already inside trace.npz.

Usage:
    .venv/bin/python scripts/check_replay_tracking.py [attempt_dir]
    .venv/bin/python scripts/check_replay_tracking.py --plot          # + tracking.png in the attempt dir
    .venv/bin/python scripts/check_replay_tracking.py --plot out.png  # figure to a chosen path
    .venv/bin/python scripts/check_replay_tracking.py --show          # interactive window (needs a display)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from moz1_catch.feedback import tracking_figure, tracking_summary

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output"
FEEDBACK_KEYS = ("feedback_t_s", "feedback_palm_position")


def peak_speed(t_s, position) -> float:
    """Max per-sample implied speed of a (N, ...) series, m/s."""
    if len(t_s) < 2:
        return 0.
    step = np.linalg.norm(np.diff(position, axis=0), axis=-1) / np.diff(t_s)[:, None]
    return float(np.nanmax(step))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attempt", type=Path, nargs="?", default=None,
                        help="attempt directory to analyse (default: latest "
                             "output/attempt_*_replay)")
    parser.add_argument("--plot", type=Path, nargs="?", const="auto", default=None,
                        help="save a tracking figure: per-axis command-vs-actual, "
                             "speeds, error norm, 3-D palm paths (default path: "
                             "tracking.png inside the attempt directory)")
    parser.add_argument("--show", action="store_true",
                        help="open the figure in a window (needs a display)")
    args = parser.parse_args(argv)

    attempt = args.attempt
    if attempt is None:
        candidates = sorted(Path(DEFAULT_OUTPUT).glob("attempt_*_replay"))
        if not candidates:
            print(f"no attempt_*_replay directories under {DEFAULT_OUTPUT}")
            return 1
        attempt = candidates[-1]
    trace_path = attempt / "trace.npz"
    if not trace_path.is_file():
        print(f"no trace.npz in {attempt}")
        return 1

    with np.load(trace_path, allow_pickle=True) as trace:
        if not all(key in trace.files for key in FEEDBACK_KEYS):
            print(f"{attempt}: trace predates joint feedback recording - no "
                  f"feedback_* keys; rerun the replay to capture actual motion")
            return 1
        command_t = trace["command_t_host_s"]
        phases = trace["command_phase"]
        command_pos = trace["target_palm_position"]
        feedback_t = trace["feedback_t_s"]
        feedback_pos = trace["feedback_palm_position"]
        clamped = trace["command_clamped"]
        contact = (float(trace["contact_time_s"])
                   if "contact_time_s" in trace.files else None)

    execute = phases == "execute"
    window = "execute phase" if execute.sum() >= 2 else "whole command window"
    if execute.sum() < 2:
        execute = np.ones(len(command_t), dtype=bool)
    print(f"attempt          : {attempt}  (tracking over {window})")
    print(f"samples          : {execute.sum()} command @ "
          f"{1. / np.median(np.diff(command_t[execute])):.0f} Hz, "
          f"{len(feedback_t)} feedback @ "
          f"{1. / np.median(np.diff(feedback_t)):.0f} Hz")
    print(f"clamp violations : {sum(len(entry) for entry in clamped)} (must be 0)")
    tracking = tracking_summary(command_t[execute], command_pos[execute],
                                feedback_t, feedback_pos)
    for side in ("left", "right"):
        stats = tracking[side]
        hand = 0 if side == "left" else 1
        boundary = ("  [lag at search boundary - estimate unreliable]" 
                    if stats["lag_at_boundary"] else "")
        print(f"{side:>5} hand      : lag {stats['lag_s'] * 1000:+7.1f} ms | "
              f"error mean {stats['error_mean_mm']:6.1f} mm, "
              f"max {stats['error_max_mm']:6.1f} mm | "
              f"after lag compensation mean {stats['compensated_error_mean_mm']:5.1f} mm | "
              f"peak speed cmd {peak_speed(command_t[execute], command_pos[execute, :, hand]):.2f} / "
              f"actual {peak_speed(feedback_t, feedback_pos[:, hand]):.2f} m/s{boundary}")
    print("clock note       : lag is end-to-end on the host clock (command "
          "publish -> /joint_states receipt), transport delay included")

    if args.plot is None and not args.show:
        return 0
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")            # headless save without a display
    lags = " / ".join(f"{side} {tracking[side]['lag_s'] * 1000:+.0f} ms"
                      for side in ("left", "right"))
    worst = max(tracking[side]["error_max_mm"] for side in ("left", "right"))
    figure = tracking_figure(
        command_t, phases, command_pos, feedback_t, feedback_pos, tracking,
        contact_after_execute_s=contact,
        title=f"sim-plan replay tracking - lag {lags}, max error {worst:.1f} mm")
    if args.plot is not None:
        target = args.plot if str(args.plot) != "auto" else attempt / "tracking.png"
        figure.savefig(target, dpi=150)
        print(f"figure saved     : {target}")
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
