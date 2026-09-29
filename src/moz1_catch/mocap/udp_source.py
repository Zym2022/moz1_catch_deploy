"""UDP listener for the mocap rigid-body stream.

The wire format is intentionally left as a swappable parser: register a callable
in PARSERS under a short name and select it via config/interfaces.toml
([mocap_udp] parser = "...").  The parser receives the raw datagram and returns
(device_t_s, position_in_device_units, quaternion_xyzw, tracking_state,
rigid_body_id) or None for datagrams that are not the box rigid body.

Registered parsers:

  "opti_json"      the real vendor stream (measured 2026-09-29, see the parser
                   docstring for the captured packet shape)
  "prototype_json" the documented test format used by scripts/fake_mocap_sender.py:

    {"t": 12.345, "id": 5, "x": 0.1, "y": -1.5, "z": 1.2,
     "qx": 0., "qy": 0., "qz": 0., "qw": 1., "state": 8}
"""

from __future__ import annotations

import json
import socket
import time

import numpy as np

from moz1_catch.calib import ArrivalClockAnchor, FrameChain, MocapClock
from moz1_catch.config import UdpConfig
from moz1_catch.mocap.source import Observation

ParsedPacket = tuple[float, np.ndarray, np.ndarray, int, int] | None


def _parse_prototype_json(datagram: bytes) -> ParsedPacket:
    packet = json.loads(datagram.decode("ascii"))
    return (float(packet["t"]),
            np.array((packet["x"], packet["y"], packet["z"]), dtype=float),
            np.array((packet["qx"], packet["qy"], packet["qz"], packet["qw"]), dtype=float),
            int(packet["state"]),
            int(packet["id"]))


def _parse_opti_json(datagram: bytes) -> ParsedPacket:
    """Real mocap UDP stream, measured on the robot host 2026-09-29.

    The device (192.168.12.3) sends two datagrams per mocap tick at 200 Hz:
    a ~455-byte rigid-body frame and a 59-byte marker frame:

      {"opti_frmIdx":527016,"opti_fTimestamp":"07:55:54.024",
       "opti_rb_0_trackid":490978,"opti_rb_0_id":0,"opti_rb_0_tag":0,
       "opti_rb_0_rigidtype":104,"opti_rb_0_px":-1.37283,"opti_rb_0_py":...,
       "opti_rb_0_pz":...,"opti_rb_0_qw":-0.874895,"opti_rb_0_qx":...,
       "opti_rb_0_qy":...,"opti_rb_0_qz":...,"opti_rb_0_meanError":0.00073,
       "opti_rb_0_params":1,"opti_rb_0_btracked":1,
       "opti_rb_0_markerTotalCnt":3,"opti_rb_0_markerVisibleCnt":3}
      {"marker_frmIdx":527016,"marker_fTimestamp":"07:55:54.024"}

    Conversions applied here:
      * quaternion arrives wxyz, this repo uses xyzw;
      * "HH:MM:SS.mmm" becomes seconds-of-day (a catch attempt is seconds long,
        so the once-per-day midnight wrap cannot strike mid-run);
      * the tracking state is the btracked flag (1 = tracked), so
        frames.toml tracking_valid_states must contain 1;
      * marker-only datagrams parse to None (counted as dropped, which is fine
        - they pair 1:1 with rigid-body frames).
    Returns the first rigid-body group of the frame; rigid-body-free frames
    also parse to None.  When the device later streams several bodies in one
    datagram, extend the slot loop to select by id.
    """
    packet = json.loads(datagram.decode("ascii"))
    if "opti_frmIdx" not in packet:
        return None
    hours, minutes, seconds = packet["opti_fTimestamp"].split(":")
    device_t = 3600. * int(hours) + 60. * int(minutes) + float(seconds)
    slot = next((index for index in range(32) if f"opti_rb_{index}_id" in packet), None)
    if slot is None:
        return None
    prefix = f"opti_rb_{slot}_"
    return (device_t,
            np.array((packet[prefix + "px"], packet[prefix + "py"], packet[prefix + "pz"]),
                     dtype=float),
            np.array((packet[prefix + "qx"], packet[prefix + "qy"],
                      packet[prefix + "qz"], packet[prefix + "qw"]), dtype=float),
            int(packet[prefix + "btracked"]),
            int(packet[prefix + "id"]))


def _parse_unimplemented(datagram: bytes) -> ParsedPacket:
    raise NotImplementedError(
        "The requested mocap UDP packet parser is not implemented. Register it "
        "in moz1_catch/mocap/udp_source.py PARSERS and set [mocap_udp] parser in "
        "config/interfaces.toml. For bench tests use 'prototype_json' with "
        "scripts/fake_mocap_sender.py.")


PARSERS = {
    "TODO_REPLACE_ME": _parse_unimplemented,
    "prototype_json": _parse_prototype_json,
    "opti_json": _parse_opti_json,
}


class UdpMocapSource:
    """Blocking-with-timeout reader for one box rigid body.

    With auto_anchor=True the device->host clock offset is measured per run
    (calib.ArrivalClockAnchor): the first next() drains everything queued
    since bind, then the first fresh arrival anchors the offset and the
    median of the following arrivals refines it once.  The clock handed in
    only supplies the pre-anchor fallback offset.
    """

    def __init__(self, udp: UdpConfig, chain: FrameChain, clock: MocapClock,
                 valid_states: tuple[int, ...], *, auto_anchor: bool = False, log=None):
        if udp.parser not in PARSERS:
            raise ValueError(f"unknown mocap parser {udp.parser!r}; known: {sorted(PARSERS)}")
        self._parse = PARSERS[udp.parser]
        self._udp = udp
        self._chain = chain
        self._clock = clock
        self._valid_states = set(valid_states)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((udp.bind_host, udp.bind_port))
        self._socket.setblocking(False)
        self._buffer = bytearray(65536)
        self.packets_dropped = 0
        self._log = log
        self._anchor = ArrivalClockAnchor(clock.offset_s, log=log) if auto_anchor else None
        self._pending_drain = auto_anchor

    @property
    def clock_offset_s(self) -> float:
        """The device->host offset currently applied to observations."""
        return self._anchor.offset_s if self._anchor is not None else self._clock.offset_s

    def _drain(self) -> int:
        """Discard datagrams queued since bind so anchoring sees a fresh arrival."""
        drained = 0
        while True:
            try:
                self._socket.recvfrom_into(self._buffer)
            except BlockingIOError:
                return drained
            except ConnectionResetError:
                continue
            drained += 1

    def next(self, timeout_s: float) -> Observation | None:
        if self._pending_drain:
            self._pending_drain = False
            drained = self._drain()
            if drained and self._log is not None:
                self._log(f"mocap_clock_drained_backlog_datagrams={drained}")
        try:
            datagram, _ = self._socket.recvfrom_into(self._buffer)
        except BlockingIOError:
            return None
        except ConnectionResetError:  # ICMP port unreachable on some peers; ignore
            return None
        parsed = self._parse(bytes(self._buffer[:datagram]))
        if parsed is None:
            self.packets_dropped += 1
            return None
        device_t, position, quaternion, state, rigid_body_id = parsed
        if self._anchor is not None and self._anchor.update(device_t, time.perf_counter()):
            self._clock = MocapClock(offset_s=self._anchor.offset_s)
        if rigid_body_id != self._udp.rigid_body_id:
            self.packets_dropped += 1
            return None
        position_m, rotation = self._chain.box_geometry_pose(position, quaternion)
        return Observation(
            t_s=self._clock.to_host(device_t),
            position_m=position_m,
            quat_xyzw=rotation.as_quat(),
            valid=state in self._valid_states,
            device_t_s=device_t,
        )

    def close(self) -> None:
        self._socket.close()
