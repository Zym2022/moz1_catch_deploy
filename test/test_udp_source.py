"""Parser unit tests and a loopback test for the mocap UDP source."""

import json
import socket
import time
from pathlib import Path

import numpy as np
import pytest

from moz1_catch.calib import ArrivalClockAnchor, FrameChain, MocapClock
from moz1_catch.config import UdpConfig, load_config
from moz1_catch.mocap.udp_source import PARSERS, UdpMocapSource

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# A rigid-body datagram captured from the real stream (2026-09-29,
# 192.168.12.3 -> robot host :65501); markerTotalCnt/markerVisibleCnt were
# truncated in the capture printout and are restored to plausible values -
# the parser does not read them.
REAL_RB_PACKET = {
    "opti_frmIdx": 527016, "opti_fTimestamp": "07:55:54.024",
    "opti_rb_0_trackid": 490978, "opti_rb_0_id": 0, "opti_rb_0_tag": 0,
    "opti_rb_0_rigidtype": 104,
    "opti_rb_0_px": -1.37283, "opti_rb_0_py": 0.987164, "opti_rb_0_pz": -0.233824,
    "opti_rb_0_qw": -0.874895, "opti_rb_0_qx": 0.0129181,
    "opti_rb_0_qy": 0.484139, "opti_rb_0_qz": -0.00106734,
    "opti_rb_0_meanError": 0.000731982, "opti_rb_0_params": 1,
    "opti_rb_0_btracked": 1, "opti_rb_0_markerTotalCnt": 3,
    "opti_rb_0_markerVisibleCnt": 3,
}
MARKER_PACKET = {"marker_frmIdx": 527016, "marker_fTimestamp": "07:55:54.024"}


def _datagram(packet: dict) -> bytes:
    return json.dumps(packet).encode("ascii")


def test_opti_json_parses_captured_rigid_body_datagram():
    device_t, position, quaternion, state, rigid_body_id = \
        PARSERS["opti_json"](_datagram(REAL_RB_PACKET))
    assert device_t == pytest.approx(7 * 3600 + 55 * 60 + 54.024)
    np.testing.assert_allclose(position, (-1.37283, 0.987164, -0.233824))
    # The stream is wxyz; the repo contract is xyzw.
    np.testing.assert_allclose(quaternion, (0.0129181, 0.484139, -0.00106734, -0.874895))
    assert state == 1 and rigid_body_id == 0


def test_opti_json_ignores_marker_only_datagram():
    assert PARSERS["opti_json"](_datagram(MARKER_PACKET)) is None


def test_opti_json_without_rigid_bodies_is_none():
    assert PARSERS["opti_json"](_datagram({"opti_frmIdx": 1,
                                           "opti_fTimestamp": "00:00:00.000"})) is None


def test_opti_json_timestamp_is_seconds_of_day():
    packet = dict(REAL_RB_PACKET, opti_fTimestamp="23:59:59.999")
    assert PARSERS["opti_json"](_datagram(packet))[0] == pytest.approx(86399.999)


def test_udp_source_loopback_through_real_chain():
    config = load_config(CONFIG_DIR)
    # Bind an ephemeral loopback port, never the real 65501 stream.
    udp = UdpConfig(bind_host="127.0.0.1", bind_port=0, parser="opti_json",
                    rigid_body_id=0)
    chain = FrameChain(T_FM=config.frames.T_FM, T_DG=config.frames.T_DG,
                       T_base_torso=config.robot.T_base_torso)
    source = UdpMocapSource(udp, chain, MocapClock(),
                            config.frames.mocap.tracking_valid_states)
    port = source._socket.getsockname()[1]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(_datagram(MARKER_PACKET), ("127.0.0.1", port))
        sender.sendto(_datagram(REAL_RB_PACKET), ("127.0.0.1", port))
    first = source.next(timeout_s=1.)
    assert first is None            # marker datagram: dropped, nothing delivered
    observation = source.next(timeout_s=1.)
    assert observation is not None and observation.valid
    assert observation.device_t_s == pytest.approx(7 * 3600 + 55 * 60 + 54.024)
    assert observation.t_s == observation.device_t_s     # zero clock offset
    np.testing.assert_allclose(
        observation.position_m,
        (-1.248276771068513, -1.652010148392673, 0.8417878801639517))
    assert source.packets_dropped == 1
    source.close()


def _stamp(device_t: float) -> str:
    """Seconds -> the opti_json 'HH:MM:SS.mmm' device timestamp (hours >23 ok)."""
    ms = int(round(device_t * 1000.))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def test_arrival_anchor_provisional_median_then_frozen():
    anchor = ArrivalClockAnchor(fallback_offset_s=99.0, refine_packets=4)
    assert anchor.offset_s == 99.0 and not anchor.anchored
    assert anchor.update(900.000, 100.000) is True        # provisional anchor
    assert anchor.anchored and anchor.offset_s == pytest.approx(-800.0)
    assert anchor.update(900.005, 100.005) is False       # still collecting
    assert anchor.update(900.010, 100.010) is False
    assert anchor.update(900.015, 100.015) is True        # 4th sample closes: median
    assert anchor.offset_s == pytest.approx(-800.0)
    assert anchor.update(800.000, 100.000) is False       # frozen afterwards
    assert anchor.offset_s == pytest.approx(-800.0)


def test_arrival_anchor_median_drops_a_stale_provisional_packet():
    anchor = ArrivalClockAnchor(refine_packets=5)
    assert anchor.update(1000.000, 60.010) is True        # 10 ms stale outlier
    assert anchor.offset_s == pytest.approx(-939.990)
    for device, host in ((1000.005, 60.005), (1000.010, 60.010), (1000.015, 60.015)):
        assert anchor.update(device, host) is False
    assert anchor.update(1000.020, 60.020) is True        # 5th sample closes
    assert anchor.offset_s == pytest.approx(-940.0)       # stale sample outvoted


def test_arrival_anchor_closes_by_elapsed_time_and_installs_median():
    anchor = ArrivalClockAnchor(refine_packets=1000, max_refine_s=1.0)
    anchor.update(0.0, 10.0)
    assert anchor.update(0.1, 10.1) is False
    assert anchor.update(0.2, 10.2) is False
    assert anchor.update(0.0, 11.5) is True               # window shut by time
    assert anchor.offset_s == pytest.approx(10.0)         # median of the three diffs
    assert anchor.update(0.3, 11.6) is False              # frozen


def test_arrival_anchor_rejects_invalid_window():
    with pytest.raises(ValueError):
        ArrivalClockAnchor(refine_packets=0)
    with pytest.raises(ValueError):
        ArrivalClockAnchor(max_refine_s=0.)


def test_udp_source_auto_anchor_drains_backlog_and_anchors_on_arrival():
    config = load_config(CONFIG_DIR)
    # Bind an ephemeral loopback port, never the real 65501 stream.
    udp = UdpConfig(bind_host="127.0.0.1", bind_port=0, parser="opti_json",
                    rigid_body_id=0)
    chain = FrameChain(T_FM=config.frames.T_FM, T_DG=config.frames.T_DG,
                       T_base_torso=config.robot.T_base_torso)
    logs: list[str] = []
    # An absurd manual offset must not survive the auto anchor.
    source = UdpMocapSource(udp, chain, MocapClock(offset_s=1234.0),
                            config.frames.mocap.tracking_valid_states,
                            auto_anchor=True, log=logs.append)
    port = source._socket.getsockname()[1]
    true_offset = -100.0                    # pretend the device clock lags the host
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        # Backlog queued before the first next() call, its device time half a
        # second stale - the anchor must drain it, not anchor on it.
        stale = dict(REAL_RB_PACKET,
                     opti_fTimestamp=_stamp(time.perf_counter() + true_offset - .5))
        sender.sendto(_datagram(stale), ("127.0.0.1", port))
    assert source.next(timeout_s=.05) is None            # backlog drained, nothing fresh
    assert source.clock_offset_s == 1234.0               # not anchored yet
    sent_at = time.perf_counter()
    fresh = dict(REAL_RB_PACKET, opti_fTimestamp=_stamp(sent_at + true_offset))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.sendto(_datagram(fresh), ("127.0.0.1", port))
    observation = source.next(timeout_s=.05)
    assert observation is not None and observation.valid
    # Anchored on the FRESH arrival: t_s is the host send time within loopback
    # + queueing slop, not device time + the absurd manual offset.
    assert observation.t_s == pytest.approx(sent_at, abs=.1)
    assert source.clock_offset_s == pytest.approx(observation.t_s - observation.device_t_s)
    assert source.clock_offset_s != pytest.approx(1234.0)
    assert any("mocap_clock_drained_backlog_datagrams=1" in line for line in logs)
    assert any("mocap_clock_anchor=provisional" in line for line in logs)
    source.close()
