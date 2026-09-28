"""Per-attempt recording: command log, observation log, events, metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

import moz1_catch
from moz1_catch.mocap.source import Observation
from moz1_catch.robot.sink import HandTargets

_OBSERVATION_KEYS = ("t_s", "position_m", "quat_xyzw", "valid", "device_t_s")
_COMMAND_KEYS = ("t_host", "phase", "t_plan", "palm_position", "palm_velocity",
                 "palm_rotation_xyzw", "tcp_position", "tcp_rotation_xyzw", "clamped")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TraceRecorder:
    """Buffers one attempt; save() writes trace.npz and meta.json side by side."""

    def __init__(self, config):
        self.observations: dict[str, list] = {key: [] for key in _OBSERVATION_KEYS}
        self.commands: dict[str, list] = {key: [] for key in _COMMAND_KEYS}
        self.preview_time_s: list[float] = []
        self.preview_contact_position_m: list[np.ndarray] = []
        self.events: list[tuple[float, str]] = []
        self._config = config
        self.started_host = time.perf_counter()

    def event(self, message: str) -> None:
        self.events.append((time.perf_counter() - self.started_host, message))

    def record_observation(self, observation: Observation) -> None:
        record = self.observations
        record["t_s"].append(observation.t_s)
        record["position_m"].append(observation.position_m)
        record["quat_xyzw"].append(observation.quat_xyzw)
        record["valid"].append(observation.valid)
        record["device_t_s"].append(observation.device_t_s)

    def record_preview(self, t_s: float, contact_position: np.ndarray) -> None:
        self.preview_time_s.append(t_s)
        self.preview_contact_position_m.append(np.asarray(contact_position, dtype=float))

    def record_command(self, t_host: float, phase: str, t_plan: float,
                       palm_targets: HandTargets, tcp_targets: HandTargets,
                       palm_velocity: np.ndarray, clamped: list[str]) -> None:
        record = self.commands
        record["t_host"].append(t_host)
        record["phase"].append(phase)
        record["t_plan"].append(t_plan)
        record["palm_position"].append(palm_targets.positions_m)
        record["palm_velocity"].append(np.asarray(palm_velocity, dtype=float))
        record["palm_rotation_xyzw"].append(
            np.array([rotation.as_quat() for rotation in palm_targets.rotations]))
        record["tcp_position"].append(tcp_targets.positions_m)
        record["tcp_rotation_xyzw"].append(
            np.array([rotation.as_quat() for rotation in tcp_targets.rotations]))
        record["clamped"].append(clamped)

    def save(self, output_dir: Path, decision: str, reason: str, extra: dict) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        attempt_dir = output_dir / f"attempt_{stamp}_{decision}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        observation_arrays = {
            "observation_time_s": np.asarray(self.observations["t_s"]),
            "observation_position_m": np.asarray(self.observations["position_m"]).reshape(-1, 3),
            "observation_quat_xyzw": np.asarray(self.observations["quat_xyzw"]).reshape(-1, 4),
            "observation_valid": np.asarray(self.observations["valid"], dtype=bool),
            "observation_device_t_s": np.asarray(self.observations["device_t_s"]),
            "preview_time_s": np.asarray(self.preview_time_s),
            "preview_contact_position_m": np.asarray(self.preview_contact_position_m).reshape(-1, 3),
        }
        command_arrays = {
            "command_t_host_s": np.asarray(self.commands["t_host"]),
            "command_phase": np.asarray(self.commands["phase"]),
            "command_t_plan_s": np.asarray(self.commands["t_plan"]),
            "target_palm_position": np.asarray(self.commands["palm_position"]).reshape(-1, 2, 3),
            "target_palm_velocity": np.asarray(self.commands["palm_velocity"]).reshape(-1, 2, 3),
            "target_palm_rotation_xyzw": np.asarray(self.commands["palm_rotation_xyzw"]).reshape(-1, 2, 4),
            "command_tcp_position": np.asarray(self.commands["tcp_position"]).reshape(-1, 2, 3),
            "command_tcp_rotation_xyzw": np.asarray(self.commands["tcp_rotation_xyzw"]).reshape(-1, 2, 4),
            "command_clamped": np.asarray([list(item) for item in self.commands["clamped"]], dtype=object),
        }
        np.savez_compressed(attempt_dir / "trace.npz",
                            catch_decision=decision, rejection_reason=reason,
                            event_time_s=np.asarray([item[0] for item in self.events]),
                            event_message=np.asarray([item[1] for item in self.events]),
                            **observation_arrays, **command_arrays, **{
                                key: value for key, value in extra.items()
                                if isinstance(value, (np.ndarray, float, int, str, bool))})
        meta = dict(
            decision=decision, reason=reason, version=moz1_catch.__version__,
            saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            config_sources={str(source): _file_sha256(source) for source in self._config.sources},
            frames_version=self._config.frames.mocap.version,
            posture=dict(urdf=str(self._config.robot.posture.urdf),
                         urdf_sha256=_file_sha256(self._config.robot.posture.urdf),
                         legwaist_joint_deg=list(self._config.robot.posture.legwaist_joint_deg),
                         left_arm_joint_deg=list(self._config.robot.posture.left_arm_joint_deg),
                         right_arm_joint_deg=list(self._config.robot.posture.right_arm_joint_deg)),
            wait_poses_m=[hand.wait_position_m.tolist() for hand in self._config.robot.hands],
            catch_settings=asdict(self._config.catch_settings),
            prediction_settings=asdict(self._config.prediction.settings),
            mission=asdict(self._config.mission),
            execution=asdict(self._config.execution),
            events=[{"t_s": item[0], "message": item[1]} for item in self.events],
            scalar_extra={key: value for key, value in extra.items()
                          if isinstance(value, (float, int, str, bool))},
        )
        (attempt_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        return attempt_dir
