"""Configuration loading, profile overlay and validation."""

from pathlib import Path

import numpy as np
import pytest

from moz1_catch.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def write_copy(tmp_path: Path, edits: dict[str, list[tuple[str, str]]]) -> Path:
    """Copy the shipped config with literal text edits; replay profile included."""
    directory = tmp_path / "config"
    (directory / "profiles").mkdir(parents=True)
    for name in ("catch.toml", "frames.toml", "robot.toml", "interfaces.toml"):
        text = (CONFIG_DIR / name).read_text()
        for old, new in edits.get(name, []):
            assert old in text, f"{name}: edit pattern not found"
            text = text.replace(old, new)
        (directory / name).write_text(text)
    (directory / "profiles" / "replay.toml").write_text(
        (CONFIG_DIR / "profiles" / "replay.toml").read_text())
    data = tmp_path / "data"
    data.mkdir()
    (data / "moz1_boxer.urdf").write_bytes(
        (CONFIG_DIR.parent / "data" / "moz1_boxer.urdf").read_bytes())
    return directory


def test_replay_profile_loads_with_analysis_prior():
    config = load_config(CONFIG_DIR, "replay")
    assert config.source_kind == "replay" and config.sink_kind == "mock"
    np.testing.assert_allclose(config.prediction.settings.acceleration_prior_mps2,
                               (-0.011349310646618586, -0.3868940737031952, -8.791483013796956),
                               atol=1e-12)
    assert config.prediction.settings.vertical_forecast_gain_per_m == pytest.approx(0.06827448239423729)
    assert config.catch_settings.plane_y == -0.58  # LATE_COMMIT baseline, base_link
    assert config.frames.T_DG[0, 3] == pytest.approx(-0.0485375)


def test_base_config_is_live_udp_ros2_base_link_planning():
    config = load_config(CONFIG_DIR)
    assert config.source_kind == "udp" and config.sink_kind == "ros2"
    # Planning frame is base_link: the sim mission geometry is used unchanged.
    assert config.mission.commit_plane_y_m == -1.05
    assert config.catch_settings.plane_y == -0.58
    assert config.catch_settings.center_z == 1.20
    assert config.udp.parser == "TODO_REPLACE_ME"
    assert np.allclose(config.frames.T_FM, np.eye(4))       # hand-eye placeholder
    assert np.allclose(config.robot.T_base_torso[:3, :3], np.eye(3))  # FK constant
    assert config.robot.T_base_torso[2, 3] == pytest.approx(1.202396)


def test_unknown_planning_override_is_rejected(tmp_path):
    bad = write_copy(tmp_path, {"catch.toml": [("# max_palm_speed = 3.0", "typo_field = 1")]})
    with pytest.raises(ValueError, match="unknown CatchSettings overrides"):
        load_config(bad)


def test_planning_override_is_applied(tmp_path):
    edited = write_copy(tmp_path, {"catch.toml": [
        ("# max_palm_speed = 3.0", "max_palm_speed = 2.5")]})
    config = load_config(edited)
    assert config.catch_settings.max_palm_speed == 2.5
    assert config.catch_settings.close_duration == 0.08  # untouched baseline
    assert config.catch_settings.plane_y == -0.58


def test_mission_ordering_is_cross_checked(tmp_path):
    bad = write_copy(tmp_path, {"catch.toml": [("commit_plane_y_m = -1.05",
                                                "commit_plane_y_m = -0.30")]})
    with pytest.raises(ValueError, match="commit plane"):
        load_config(bad)


def test_missing_profile_and_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path, "replay")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="catch.toml"):
        load_config(empty)
