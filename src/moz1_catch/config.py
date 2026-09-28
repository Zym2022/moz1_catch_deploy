"""Typed loading and validation of the TOML configuration set.

Files: catch.toml + frames.toml + robot.toml + interfaces.toml, optionally
overlaid by one profile from config/profiles/. Profiles deep-merge over the
base files, so a profile only contains what differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from moz1_catch.core.one_shot import CatchSettings, LATE_COMMIT_SETTINGS
from moz1_catch.core.prediction import PredictionSettings


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


@dataclass(frozen=True)
class MissionConfig:
    release_y_m: float
    commit_plane_y_m: float
    min_commit_delay_s: float
    commit_timeout_s: float
    release_settle_s: float


@dataclass(frozen=True)
class PredictionConfig:
    settings: PredictionSettings
    max_observation_age_s: float


@dataclass(frozen=True)
class ArmingConfig:
    hold_s: float
    max_speed_mps: float
    release_speed_threshold_mps: float
    release_region: np.ndarray  # (3, 2) min/max per axis


@dataclass(frozen=True)
class SafetyConfig:
    workspace: np.ndarray  # (3, 2) min/max per axis, both hands
    max_cartesian_speed_mps: float
    watchdog_s: float


@dataclass(frozen=True)
class ExecutionConfig:
    command_period_s: float
    command_latency_s: float
    settle_margin_s: float
    reject_duration_s: float
    hold_after_finish: bool


@dataclass(frozen=True)
class LoggingConfig:
    output_dir: Path


@dataclass(frozen=True)
class MocapConfig:
    position_units: str
    quaternion_order: str
    tracking_valid_states: tuple[int, ...]
    clock_offset_s: float
    version: str


@dataclass(frozen=True)
class FramesConfig:
    T_FM: np.ndarray
    T_DG: np.ndarray
    mocap: MocapConfig


@dataclass(frozen=True)
class HandConfig:
    name: str
    T_tcp_palm: np.ndarray  # controller TCP frame in the planner palm-target frame
    wait_position_m: np.ndarray
    wait_quat_xyzw: np.ndarray


@dataclass(frozen=True)
class PostureConfig:
    urdf: Path
    legwaist_joint_deg: tuple[float, ...]
    left_arm_joint_deg: tuple[float, ...]
    right_arm_joint_deg: tuple[float, ...]


@dataclass(frozen=True)
class RobotConfig:
    hands: tuple[HandConfig, ...]  # order [left, right]; wait poses derived by FK
    T_base_torso: np.ndarray  # ^base T_torso, derived from the legwaist posture
    posture: PostureConfig
    palm_state_topic: str


@dataclass(frozen=True)
class UdpConfig:
    bind_host: str
    bind_port: int
    parser: str
    rigid_body_id: int


@dataclass(frozen=True)
class Ros2Config:
    node_name: str
    cartesian_topic: str
    message_type: str
    message_layout: str
    queue_size: int


@dataclass(frozen=True)
class ReplayConfig:
    csv: Path
    downsample_hz: float
    recenter_to_release: bool


@dataclass(frozen=True)
class Config:
    mission: MissionConfig
    prediction: PredictionConfig
    catch_settings: CatchSettings
    arming: ArmingConfig
    safety: SafetyConfig
    execution: ExecutionConfig
    logging: LoggingConfig
    frames: FramesConfig
    robot: RobotConfig
    source_kind: str
    sink_kind: str
    udp: UdpConfig
    ros2: Ros2Config
    replay: ReplayConfig | None
    sources: tuple[Path, ...]


def _pair(table: dict, key: str) -> tuple[float, float]:
    value = table[key]
    if (not isinstance(value, list) or len(value) != 2
            or not all(isinstance(item, (int, float)) for item in value)):
        raise ValueError(f"{key} must be a [min, max] pair")
    low, high = float(value[0]), float(value[1])
    if not low < high:
        raise ValueError(f"{key} pair is not increasing")
    return low, high


def _matrix(table: dict, key: str) -> np.ndarray:
    value = np.asarray(table[key], dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{key} must be a finite 4x4 matrix")
    return value


def _vector(table: dict, key: str, length: int) -> np.ndarray:
    value = np.asarray(table[key], dtype=float)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must be {length} finite numbers")
    return value


def load_config(config_dir: Path, profile: str | None = None) -> Config:
    config_dir = Path(config_dir)
    data = {}
    names = ("catch.toml", "frames.toml", "robot.toml", "interfaces.toml")
    for name in names:
        path = config_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing configuration file: {path}")
        data = _deep_merge(data, _read_toml(path))
    sources = [config_dir / name for name in names]
    if profile is not None:
        path = config_dir / "profiles" / f"{profile}.toml"
        if not path.is_file():
            raise FileNotFoundError(f"missing profile: {path}")
        data = _deep_merge(data, _read_toml(path))
        sources.append(path)

    mission_table = data["mission"]
    mission = MissionConfig(
        release_y_m=float(mission_table["release_y_m"]),
        commit_plane_y_m=float(mission_table["commit_plane_y_m"]),
        min_commit_delay_s=float(mission_table["min_commit_delay_s"]),
        commit_timeout_s=float(mission_table["commit_timeout_s"]),
        release_settle_s=float(mission_table["release_settle_s"]),
    )
    if not (mission.release_y_m < mission.commit_plane_y_m
            and 0 <= mission.min_commit_delay_s < mission.commit_timeout_s
            and 0 <= mission.release_settle_s):
        raise ValueError("invalid mission ordering: release < commit plane, delays inside timeout")

    prediction_table = data["prediction"]
    prior = _vector(prediction_table, "acceleration_prior_mps2", 3)
    prediction = PredictionConfig(
        settings=PredictionSettings(
            float(prediction_table["window_s"]),
            float(prediction_table["angular_window_s"]),
            tuple(prior),
            float(prediction_table["regularization_s2"]),
        ),
        max_observation_age_s=float(prediction_table["max_observation_age_s"]),
    )
    if prediction.max_observation_age_s <= 0:
        raise ValueError("max_observation_age_s must be positive")

    overrides = dict(data.get("planning", {}).get("overrides", {}))
    valid_fields = set(CatchSettings.__dataclass_fields__)
    unknown = set(overrides) - valid_fields
    if unknown:
        raise ValueError(f"unknown CatchSettings overrides: {sorted(unknown)}")
    catch_settings = LATE_COMMIT_SETTINGS
    if overrides:
        catch_settings = CatchSettings(
            **{**{f.name: getattr(LATE_COMMIT_SETTINGS, f.name)
                  for f in CatchSettings.__dataclass_fields__.values()},
               **{key: (tuple(value) if isinstance(value, list) else value)
                  for key, value in overrides.items()}})
    if not mission.commit_plane_y_m < catch_settings.plane_y:
        raise ValueError("commit plane must be reached before the contact plane")

    arming_table = data["arming"]
    arming = ArmingConfig(
        hold_s=float(arming_table["hold_s"]),
        max_speed_mps=float(arming_table["max_speed_mps"]),
        release_speed_threshold_mps=float(arming_table["release_speed_threshold_mps"]),
        release_region=np.array([_pair(arming_table, "release_region_x_m"),
                                 _pair(arming_table, "release_region_y_m"),
                                 _pair(arming_table, "release_region_z_m")]),
    )
    if not (0 < arming.max_speed_mps < arming.release_speed_threshold_mps and arming.hold_s > 0):
        raise ValueError("arming thresholds must satisfy 0 < quasi-static < release speed")

    safety_table = data["safety"]
    safety = SafetyConfig(
        workspace=np.array([_pair(safety_table, "workspace_x_m"),
                            _pair(safety_table, "workspace_y_m"),
                            _pair(safety_table, "workspace_z_m")]),
        max_cartesian_speed_mps=float(safety_table["max_cartesian_speed_mps"]),
        watchdog_s=float(safety_table["watchdog_s"]),
    )
    if safety.max_cartesian_speed_mps <= 0 or safety.watchdog_s <= 0:
        raise ValueError("safety caps must be positive")

    execution_table = data["execution"]
    execution = ExecutionConfig(
        command_period_s=float(execution_table["command_period_s"]),
        command_latency_s=float(execution_table["command_latency_s"]),
        settle_margin_s=float(execution_table["settle_margin_s"]),
        reject_duration_s=float(execution_table["reject_duration_s"]),
        hold_after_finish=bool(execution_table["hold_after_finish"]),
    )
    if min(execution.command_period_s, execution.settle_margin_s,
           execution.reject_duration_s) <= 0 or execution.command_latency_s < 0:
        raise ValueError("invalid execution timing configuration")

    logging_table = data["logging"]
    logging_config = LoggingConfig(output_dir=Path(logging_table["output_dir"]))

    frames_table = data["frames"]
    mocap_table = frames_table["mocap"]
    if mocap_table["position_units"] not in ("meters", "millimeters"):
        raise ValueError("position_units must be meters or millimeters")
    if mocap_table["quaternion_order"] != "xyzw":
        raise ValueError("only the xyzw quaternion order is supported")
    frames = FramesConfig(
        T_FM=_matrix(frames_table["extrinsics"]["T_FM"], "matrix"),
        T_DG=_matrix(frames_table["extrinsics"]["T_DG"], "matrix"),
        mocap=MocapConfig(
            position_units=mocap_table["position_units"],
            quaternion_order=mocap_table["quaternion_order"],
            tracking_valid_states=tuple(int(state) for state in mocap_table["tracking_valid_states"]),
            clock_offset_s=float(mocap_table["clock_offset_s"]),
            version=str(frames_table.get("version", "unversioned")),
        ),
    )

    robot_table = data["robot"]
    posture_table = robot_table["posture"]
    urdf = Path(posture_table["urdf"])
    if not urdf.is_absolute():
        urdf = config_dir.parent / urdf
    if not urdf.is_file():
        raise FileNotFoundError(f"posture urdf not found: {urdf}")
    posture = PostureConfig(
        urdf=urdf,
        legwaist_joint_deg=tuple(float(value) for value in posture_table["legwaist_joint_deg"]),
        left_arm_joint_deg=tuple(float(value) for value in posture_table["left_arm_joint_deg"]),
        right_arm_joint_deg=tuple(float(value) for value in posture_table["right_arm_joint_deg"]),
    )
    from moz1_catch.kinematics import parse_joints, robot_geometry
    geometry = robot_geometry(parse_joints(urdf), posture.legwaist_joint_deg,
                              posture.left_arm_joint_deg, posture.right_arm_joint_deg)
    overrides = robot_table.get("mounting_overrides", {})
    hands = []
    for name in ("left", "right"):
        derived = geometry["hands"][name]
        wait_quat = derived["wait_quat_xyzw"]
        if abs(np.linalg.norm(wait_quat) - 1.) > 1e-9 or not np.isfinite(wait_quat).all():
            raise ValueError(f"derived {name} wait quaternion is invalid")
        T_tcp_palm = (_matrix(overrides, f"T_tcp_palm_{name}") if f"T_tcp_palm_{name}" in overrides
                      else derived["T_palm_flange"])
        hands.append(HandConfig(
            name=name,
            T_tcp_palm=T_tcp_palm,
            wait_position_m=derived["wait_position_m"],
            wait_quat_xyzw=wait_quat,
        ))
    robot = RobotConfig(hands=tuple(hands),
                        T_base_torso=(_matrix(overrides, "T_base_torso")
                                      if "T_base_torso" in overrides else geometry["T_base_torso"]),
                        posture=posture,
                        palm_state_topic=str(robot_table.get("feedback", {}).get("palm_state_topic", "")))

    runtime_table = data["runtime"]
    source_kind = str(runtime_table.get("source", "udp"))
    sink_kind = str(runtime_table.get("sink", "ros2"))
    if source_kind not in ("udp", "replay") or sink_kind not in ("ros2", "mock"):
        raise ValueError(f"unknown runtime source/sink: {source_kind}/{sink_kind}")

    udp_table = data["mocap_udp"]
    udp = UdpConfig(
        bind_host=str(udp_table["bind_host"]),
        bind_port=int(udp_table["bind_port"]),
        parser=str(udp_table["parser"]),
        rigid_body_id=int(udp_table["rigid_body_id"]),
    )
    ros2_table = data["ros2"]
    ros2 = Ros2Config(
        node_name=str(ros2_table["node_name"]),
        cartesian_topic=str(ros2_table["cartesian_topic"]),
        message_type=str(ros2_table["message_type"]),
        message_layout=str(ros2_table["message_layout"]),
        queue_size=int(ros2_table["queue_size"]),
    )

    replay = None
    if source_kind == "replay":
        replay_table = data["replay"]
        replay = ReplayConfig(
            csv=Path(replay_table["csv"]),
            downsample_hz=float(replay_table["downsample_hz"]),
            recenter_to_release=bool(replay_table["recenter_to_release"]),
        )
        if replay.downsample_hz < 0:
            raise ValueError("downsample_hz must be >= 0 (0 = native)")
        if replay.csv != Path(""):
            if not replay.csv.is_file():
                raise FileNotFoundError(f"replay csv not found: {replay.csv}")
        else:
            replay = None  # per-run value; dry_run_replay.py always passes --csv

    return Config(
        mission=mission, prediction=prediction, catch_settings=catch_settings,
        arming=arming, safety=safety, execution=execution, logging=logging_config,
        frames=frames, robot=robot, source_kind=source_kind, sink_kind=sink_kind,
        udp=udp, ros2=ros2, replay=replay, sources=tuple(sources),
    )
