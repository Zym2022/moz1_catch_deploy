"""End-to-end runtime checks: synthetic flights, recorded throws, UDP loopback.

The synthetic source feeds a clean 120 Hz parabola (quasi-static hold, release,
crossing) straight into CatchRuntime with the mock sink and the replay-profile
config (identity extrinsics).  The recorded-throw test replays the five real
CSVs in real time and pins the per-throw decisions.
"""

from dataclasses import replace
import json
from pathlib import Path
import socket
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from moz1_catch.config import ReplayConfig, load_config
from moz1_catch.core.prediction import BoxFlight
from moz1_catch.executor import Executor
from moz1_catch.mocap.replay_source import ReplayMocapSource
from moz1_catch.mocap.source import Observation
from moz1_catch.mocap.udp_source import UdpMocapSource
from moz1_catch.robot.mock_sink import MockCartesianSink
from moz1_catch.runtime import CatchRuntime
from moz1_catch.safety import SafetyEnvelope
from moz1_catch.trace import TraceRecorder

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


class SyntheticSource:
    """Deterministic in-memory observation stream, paced in real time."""

    def __init__(self, observations):
        self._observations = list(observations)
        self._cursor = 0
        self.anchor = time.perf_counter()
        self._t0 = self._observations[0][0]

    def next(self, timeout_s: float) -> Observation | None:
        if self._cursor >= len(self._observations):
            return None
        device_t, position, quaternion, valid = self._observations[self._cursor]
        due = self.anchor + (device_t - self._t0)
        now = time.perf_counter()
        if now < due:
            wait_until = min(due, now + max(0., timeout_s))
            if now < wait_until:
                time.sleep(wait_until - now)
            if time.perf_counter() < due:
                return None
        self._cursor += 1
        return Observation(t_s=self.anchor + (device_t - self._t0),
                           position_m=np.asarray(position),
                           quat_xyzw=np.asarray(quaternion), valid=valid, device_t_s=device_t)


def synthetic_throw(acceleration=(0., 0., -9.81)):
    """Hold near the release point, then a clean parabola crossing both planes."""
    flight = BoxFlight(np.array((0.0, -1.5, 1.2)), np.array((0.05, 1.7, 2.4525)),
                       Rotation.from_euler("xyz", (3, -2, 4), degrees=True),
                       np.array((0.02, -0.05, 0.1)), np.asarray(acceleration))
    samples = []
    rate = 120.
    hold = [np.array((0.0, -1.5 + 0.002 * index / rate, 1.2))
            for index in range(int(0.8 * rate))]  # quasi-static drift ~2 mm/s
    for index, position in enumerate(hold):
        samples.append((index / rate, position, (0., 0., 0., 1.), True))
    t_flight = np.arange(1e-3, 0.75, 1. / rate)
    for t in t_flight:
        samples.append((0.8 + t, flight.positions(t), flight.rotations(t).as_quat(), True))
    return flight, samples


def make_runtime(config, source):
    trace = TraceRecorder(config)
    sink = MockCartesianSink()
    executor = Executor(sink, SafetyEnvelope(config.safety), config.execution,
                        config.robot, trace)
    runtime = CatchRuntime(config, source, executor, trace, log=lambda *_: None)
    return runtime, sink, trace


def replay_config(csv, **overrides):
    config = load_config(CONFIG_DIR, "replay")
    values = dict(csv=Path(csv), downsample_hz=120., recenter_to_release=True)
    values.update(overrides)
    return replace(config, replay=ReplayConfig(**values))


def test_synthetic_throw_is_accepted_and_streamed(tmp_path):
    config = replay_config(ROOT / "data/box_flying_csv/1.csv", downsample_hz=120.)
    config = replace(config, logging=replace(config.logging, output_dir=tmp_path))
    flight, samples = synthetic_throw()
    source = SyntheticSource(samples)
    runtime, sink, _ = make_runtime(config, source)
    result = runtime.run(max_wait_s=5.)
    assert result.decision == "accept", result.reason
    assert result.planning_time_ms < 500  # generous bound for any host
    assert sink.count > 10
    # The palms must reach the contact pose at the absolute predicted crossing
    # moment: commit observation time + execution delay + retimed contact time.
    commit_device_t = next(t for t, position, _, _ in samples
                           if t >= 0.8 and position[1] >= config.mission.commit_plane_y_m)
    predicted_absolute = (source.anchor + commit_device_t
                          + result.execution_delay_s + result.contact_time_s)
    true_absolute = source.anchor + 0.8 + flight.crossing_time(config.catch_settings.plane_y)
    assert predicted_absolute == pytest.approx(true_absolute, abs=0.03)
    # Trace written with the sim-compatible key set.
    trace_dir = Path(result.trace_dir)
    assert (trace_dir / "trace.npz").is_file() and (trace_dir / "meta.json").is_file()
    with np.load(trace_dir / "trace.npz") as trace:
        assert str(trace["catch_decision"]) == "accept"
        assert trace["target_palm_position"].shape[1:] == (2, 3)
        assert trace["observation_position_m"].shape[1:] == (3,)
        assert np.isfinite(trace["command_t_plan_s"]).all()
    meta = json.loads((trace_dir / "meta.json").read_text())
    assert meta["decision"] == "accept"
    assert meta["catch_settings"]["plane_y"] == config.catch_settings.plane_y


def test_box_that_never_crosses_is_rejected_for_timeout(tmp_path):
    config = replay_config(ROOT / "data/box_flying_csv/1.csv")
    config = replace(config, logging=replace(config.logging, output_dir=tmp_path))
    flight = BoxFlight(np.array((0.0, -1.5, 1.2)), np.array((0.02, 0.20, 2.4525)),
                       Rotation.identity(), np.zeros(3), np.array((0., 0., -9.81)))
    samples = []
    rate = 120.
    for index in range(int(0.8 * rate)):  # quasi-static hold
        samples.append((index / rate, np.array((0.0, -1.5, 1.2)), (0., 0., 0., 1.), True))
    t_flight = np.arange(1e-3, 1.5, 1. / rate)
    for t in t_flight:
        samples.append((0.8 + t, flight.positions(t), (0., 0., 0., 1.), True))
    runtime, sink, _ = make_runtime(config, SyntheticSource(samples))
    result = runtime.run(max_wait_s=8.)
    assert result.decision == "reject"
    assert "commit" in result.reason


def test_carried_in_box_rearms_and_the_throw_is_caught(tmp_path):
    """A box carried into the region at walking speed must not waste the attempt.

    The pre-arming shortcut fires a candidate release when the carried box
    enters the region; when the operator stops, the runtime falls back to
    ARMED, the quasi-static hold completes, and the actual throw is caught.
    Without the fallback this scenario dies on the commit timeout instead.
    """
    config = replay_config(ROOT / "data/box_flying_csv/1.csv")
    config = replace(config, logging=replace(config.logging, output_dir=tmp_path))
    rate = 120.
    samples = []
    t = 0.
    y = -2.10                                   # outside the region (y < -1.85)
    while y < -1.50:                            # walk in at 0.9 m/s along +Y
        samples.append((t, np.array((0.0, y, 1.2)), (0., 0., 0., 1.), True))
        t += 1. / rate
        y += 0.9 / rate
    for _ in range(int(0.8 * rate)):            # stand and hold the box
        samples.append((t, np.array((0.0, -1.50, 1.2)), (0., 0., 0., 1.), True))
        t += 1. / rate
    flight = BoxFlight(np.array((0.0, -1.50, 1.2)), np.array((0.05, 1.7, 2.4525)),
                       Rotation.from_euler("xyz", (3, -2, 4), degrees=True),
                       np.array((0.02, -0.05, 0.1)), np.array((0., 0., -9.81)))
    for moment in np.arange(1e-3, 0.75, 1. / rate):
        samples.append((t, flight.positions(float(moment)),
                        flight.rotations(float(moment)).as_quat(), True))
        t += 1. / rate
    runtime, sink, _ = make_runtime(config, SyntheticSource(samples))
    result = runtime.run(max_wait_s=8.)
    assert result.decision == "accept", result.reason
    assert sink.count > 10


def test_recorded_throws_replay_with_pinned_decisions(tmp_path):
    # The five originally pinned records keep the regression fast (real-time
    # pacing); any of the other 31 can be replayed manually via
    # scripts/dry_run_replay.py.
    decisions = {}
    for csv in sorted((ROOT / "data/box_flying_csv").glob("[1-5].csv")):
        config = replay_config(csv, downsample_hz=120.)
        config = replace(config, logging=replace(config.logging, output_dir=tmp_path))
        source = ReplayMocapSource(config.replay)
        runtime, sink, _ = make_runtime(config, source)
        result = runtime.run(max_wait_s=15.)
        decisions[csv.name] = (result.decision, result.reason)
        assert result.decision in ("accept", "reject")
        assert (Path(result.trace_dir) / "trace.npz").is_file()
    print(decisions)
    # Per-throw outcomes aligned with the 2026-09-28 frozen-prediction study
    # (mocap_prediction results doc): 1/3/4/5 fall outside the fixed-face
    # corridor or reachable region; throw 2 is the accepted one.
    for name in ("1.csv", "2.csv", "3.csv", "4.csv", "5.csv"):
        expected = "accept" if name == "2.csv" else "reject"
        assert decisions[name][0] == expected, (name, decisions[name])


def test_udp_source_loopback_with_prototype_parser():
    config = load_config(CONFIG_DIR)
    from moz1_catch.calib import FrameChain, MocapClock
    from moz1_catch.config import UdpConfig
    udp = replace(config.udp, bind_host="127.0.0.1", bind_port=0,
                  parser="prototype_json", rigid_body_id=5)
    frame = FrameChain(T_FM=np.eye(4), T_DG=np.eye(4))
    source = UdpMocapSource(udp, frame, MocapClock(0.), (8,))
    port = source._socket.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        packet = dict(t=1.0, id=5, state=8, x=0.1, y=-1.5, z=1.2,
                      qx=0., qy=0., qz=0., qw=1.)
        sender.sendto(json.dumps(packet).encode("ascii"), ("127.0.0.1", port))
        other = dict(packet, id=7)
        sender.sendto(json.dumps(other).encode("ascii"), ("127.0.0.1", port))
    deadline = time.perf_counter() + 2.
    received = []
    while time.perf_counter() < deadline and len(received) < 1:
        observation = source.next(0.05)
        if observation is not None:
            received.append(observation)
    source.close()
    assert len(received) == 1  # the other rigid body was filtered out
    np.testing.assert_allclose(received[0].position_m, (0.1, -1.5, 1.2), atol=1e-12)
    assert received[0].valid and received[0].t_s == pytest.approx(1.0)
