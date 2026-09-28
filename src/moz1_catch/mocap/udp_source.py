"""UDP listener for the mocap rigid-body stream.

The wire format is intentionally left as a swappable parser: register a callable
in PARSERS under a short name and select it via config/interfaces.toml
([mocap_udp] parser = "...").  The parser receives the raw datagram and returns
(device_t_s, position_in_device_units, quaternion_xyzw, tracking_state,
rigid_body_id) or None for datagrams that are not the box rigid body.

Until the real vendor format is implemented, parser "TODO_REPLACE_ME" (the
config default) refuses to run; "prototype_json" decodes the documented test
format used by scripts/fake_mocap_sender.py:

    {"t": 12.345, "id": 5, "x": 0.1, "y": -1.5, "z": 1.2,
     "qx": 0., "qy": 0., "qz": 0., "qw": 1., "state": 8}
"""

from __future__ import annotations

import json
import socket

import numpy as np

from moz1_catch.calib import FrameChain, MocapClock
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


def _parse_unimplemented(datagram: bytes) -> ParsedPacket:
    raise NotImplementedError(
        "The real mocap UDP packet format is not implemented yet. Fill in a parser "
        "in moz1_catch/mocap/udp_source.py, register it in PARSERS, and set "
        "[mocap_udp] parser in config/interfaces.toml. For bench tests use "
        "'prototype_json' with scripts/fake_mocap_sender.py.")


PARSERS = {
    "TODO_REPLACE_ME": _parse_unimplemented,
    "prototype_json": _parse_prototype_json,
}


class UdpMocapSource:
    """Blocking-with-timeout reader for one box rigid body."""

    def __init__(self, udp: UdpConfig, chain: FrameChain, clock: MocapClock,
                 valid_states: tuple[int, ...]):
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

    def next(self, timeout_s: float) -> Observation | None:
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
