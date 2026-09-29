"""Parser unit tests and a loopback test for the mocap UDP source."""

import json
import socket
from pathlib import Path

import numpy as np
import pytest

from moz1_catch.calib import FrameChain, MocapClock
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
