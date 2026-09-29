#!/usr/bin/env python3
"""Send prototype_json mocap packets over UDP from a recorded CSV.

Bench utility for the UDP path: feeds a real recording through
moz1_catch/mocap/udp_source.py with parser "prototype_json" so the socket layer
can be tested without the mocap system.

The recording is recentred like the replay source (release lands at
(0, -1.5, z) in the base_link convention), then pushed through the INVERSE of
the runtime's input chain (the calibrated T_FM hand-eye plus the real legwaist
constant T_base_torso from config, i.e. inv(T_base_torso @ T_FM)) so that what
the runtime receives converts back to exactly the base_link trajectory.  The
bench therefore exercises the real conversion chain, not a neutralized one.
Playback is paced in real time and stamped with this host's perf_counter clock.

    terminal 1: .venv/bin/python scripts/run_catch.py --profile bench
    terminal 2: .venv/bin/python scripts/fake_mocap_sender.py --csv data/box_flying_csv/2.csv
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.calib import as_transform, effective_extrinsic, transform_inverse
from moz1_catch.config import load_config
from moz1_catch.mocap.replay_source import _release_index, load_recording

R_GD = np.array(((0., 1., 0.), (0., 0., 1.), (1., 0., 0.)))
C_GD = np.array((-.0003, 0., .0485375))
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1511)
    parser.add_argument("--rigid-body-id", type=int, default=5)
    parser.add_argument("--tracking-state", type=int, default=8)
    args = parser.parse_args(argv)

    recording = load_recording(args.csv)
    start = _release_index(recording)
    shift = np.array((recording["center"][start, 0], recording["center"][start, 1] + 1.5, 0.))
    # Desired base_link box poses; the runtime's chain must invert the following.
    config = load_config(args.config_dir)
    # Full mocap->base_link input chain: calibrated T_FM + legwaist constant.
    to_device = transform_inverse(
        effective_extrinsic(config.frames.T_FM, config.robot.T_base_torso))

    sent = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        anchor = time.perf_counter()
        t0 = recording["time"][0]
        for index in range(len(recording["time"])):
            if not recording["valid"][index]:
                continue
            due = anchor + (recording["time"][index] - t0)
            now = time.perf_counter()
            if now < due:
                time.sleep(due - now)
            T_BG = as_transform(recording["rotation"][index],
                                recording["center"][index] - shift)
            # T_MD = inv(T_base_torso @ T_FM) @ T_BG @ T_GD, with T_GD = as_transform(R_GD, C_GD).
            T_MD = to_device @ T_BG @ as_transform(R_GD, C_GD)
            quaternion = Rotation.from_matrix(T_MD[:3, :3]).as_quat()
            packet = dict(t=time.perf_counter(), id=args.rigid_body_id,
                          state=args.tracking_state,
                          x=float(T_MD[0, 3]), y=float(T_MD[1, 3]), z=float(T_MD[2, 3]),
                          qx=float(quaternion[0]), qy=float(quaternion[1]),
                          qz=float(quaternion[2]), qw=float(quaternion[3]))
            sock.sendto(json.dumps(packet).encode("ascii"), (args.host, args.port))
            sent += 1
    print(f"sent {sent} packets from {args.csv.name} "
          f"(base_link-recentered, pre-transformed through inv(T_base_torso @ T_FM), host-stamped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
