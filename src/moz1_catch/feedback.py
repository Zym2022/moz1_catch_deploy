"""Joint feedback recording and command-vs-actual tracking analysis.

Until 2026-09-29 the cartesian replay logged every commanded palm target but
nothing of what the robot actually did: the /joint_states subscription used by
the joint-space approach was torn down before the cartesian phase started, so
the guide's step-3 question ("can the controller track the 120 Hz pose stream,
and by how much does it lag?") could not be answered from the trace.  This
module closes that gap, ROS-free on purpose (rclpy only appears in the sink's
subscription, scripts side; everything here is numpy so it stays testable):

  * JointFeedbackLog - accumulates both-arm joint snapshots; host-clock stamps
    share the time.perf_counter domain of the command log,
  * fk_palm_series - measured arm joints + locked legwaist -> palm target
    poses in base_link (kinematics.palms_from_angles FK),
  * tracking_summary - per-hand position error against the commanded series
    and the constant lag that best explains it.

Clock caveat: feedback stamps are callback receipt times on the host clock, so
the estimated lag includes the /joint_states transport delay - it is an
end-to-end figure (command publish -> pose reported), which is what the replay
wants to observe, not a controller-internal latency.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from moz1_catch.kinematics import named_angles, palms_from_angles, parse_joints

LEFT_JOINTS = tuple(f"LeftArm-{index}" for index in range(7))
RIGHT_JOINTS = tuple(f"RightArm-{index}" for index in range(7))


class JointFeedbackLog:
    """Both-arm /joint_states snapshots; ROS-free (duck-typed payloads in).

    Messages may carry partial name sets (the phase-A JointStateReader merges
    the same way): the latest known joints are merged per message and a full
    snapshot is appended whenever both arms are present.  One snapshot per
    message, so the effective rate equals the /joint_states rate (or aliases
    to the spin rate when messages arrive faster than callbacks drain).
    """

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._latest: dict[str, float] = {}
        self.times_s: list[float] = []
        self.left_rad: list[np.ndarray] = []
        self.right_rad: list[np.ndarray] = []

    def on_message(self, names, positions) -> None:
        self._latest.update(zip(names, (float(value) for value in positions)))
        if not all(name in self._latest for name in LEFT_JOINTS + RIGHT_JOINTS):
            return
        self.times_s.append(self._clock())
        self.left_rad.append(np.array([self._latest[name] for name in LEFT_JOINTS]))
        self.right_rad.append(np.array([self._latest[name] for name in RIGHT_JOINTS]))

    def __len__(self) -> int:
        return len(self.times_s)

    def arrays(self) -> dict[str, np.ndarray]:
        """Trace-ready arrays (keys prefixed feedback_, matching command_*)."""
        return {
            "feedback_t_s": np.asarray(self.times_s),
            "feedback_joint_left_rad": np.asarray(self.left_rad).reshape(-1, 7),
            "feedback_joint_right_rad": np.asarray(self.right_rad).reshape(-1, 7),
        }


def fk_palm_series(joints: dict, legwaist_deg, left_rad, right_rad) -> tuple[np.ndarray, np.ndarray]:
    """Measured arm joints (rad) with the locked legwaist -> palm poses.

    Returns (N, 2, 3) positions and (N, 2, 4) xyzw quaternions in base_link -
    the frame the commanded palm targets are logged in.
    """
    left_rad = np.atleast_2d(np.asarray(left_rad, dtype=float))
    right_rad = np.atleast_2d(np.asarray(right_rad, dtype=float))
    if len(left_rad) != len(right_rad):
        raise ValueError(f"left/right feedback length mismatch: "
                         f"{len(left_rad)} vs {len(right_rad)}")
    positions = np.empty((len(left_rad), 2, 3))
    quats = np.empty((len(left_rad), 2, 4))
    for index in range(len(left_rad)):
        angles = named_angles(legwaist_deg,
                              np.degrees(left_rad[index]), np.degrees(right_rad[index]))
        positions[index], quats[index] = palms_from_angles(joints, angles)
    return positions, quats


def trace_feedback_extras(config, feedback, command_t_s, command_phase,
                          command_palm_position, log=print) -> dict:
    """Live-attempt trace extras: FK'd measured palms + command-vs-actual tracking.

    run_catch passes this as the runtime's extra_provider, called once at trace
    save time (never in the real-time path): the recorded /joint_states
    snapshots are FK'd into base_link palm poses - the commanded series' frame,
    so they compare without any further transform - and the execute-phase
    tracking summary is appended.  Every live attempt then answers, from one
    trace, what was commanded, where the arms actually went, and what the box
    did.  Empty/silent feedback yields {} (mock sink or dead /joint_states).
    """
    if feedback is None or not len(feedback):
        log("feedback         : none recorded (mock sink or silent /joint_states)")
        return {}
    joints = parse_joints(config.robot.posture.urdf)
    arrays = feedback.arrays()
    palm_position, palm_rotation = fk_palm_series(
        joints, config.robot.posture.legwaist_joint_deg,
        arrays["feedback_joint_left_rad"], arrays["feedback_joint_right_rad"])
    extras = {**arrays, "feedback_palm_position": palm_position,
              "feedback_palm_rotation_xyzw": palm_rotation}
    execute = np.asarray(command_phase) == "execute"
    try:
        tracking = tracking_summary(np.asarray(command_t_s)[execute],
                                    np.asarray(command_palm_position)[execute],
                                    arrays["feedback_t_s"], palm_position)
    except ValueError as error:
        log(f"feedback         : {len(feedback)} snapshots, "
            f"tracking not computable ({error})")
        return extras
    extras.update(tracking_lag_left_ms=tracking["left"]["lag_s"] * 1000.,
                  tracking_lag_right_ms=tracking["right"]["lag_s"] * 1000.,
                  tracking_error_max_mm=max(tracking[side]["error_max_mm"]
                                             for side in ("left", "right")))
    log(f"feedback         : {len(feedback)} joint snapshots -> tracking lag "
        f"L {tracking['left']['lag_s'] * 1000:+.0f} / R {tracking['right']['lag_s'] * 1000:+.0f} ms, "
        f"max error {extras['tracking_error_max_mm']:.1f} mm")
    return extras


def interp_position(t_s, series: np.ndarray, query_s) -> np.ndarray:
    """Linear interpolation of a (N, 3) position series; edges clamp."""
    out = np.empty((len(query_s), series.shape[1]))
    for axis in range(series.shape[1]):
        out[:, axis] = np.interp(query_s, t_s, series[:, axis])
    return out


def tracking_summary(command_t_s, command_position, feedback_t_s, feedback_position,
                     max_lag_s: float = 0.15, lag_step_s: float = 0.005) -> dict:
    """Position tracking of the FK feedback against the commanded palm series.

    For each hand: the error at zero lag, the constant lag in
    [-max_lag_s, +max_lag_s] that minimises the mean error (positive = the
    feedback trails the command), and the residual after removing it.  The
    commanded series is interpolated at feedback stamps shifted by the lag;
    feedback samples outside the command window are dropped.
    """
    command_t_s = np.asarray(command_t_s, dtype=float)
    command_position = np.asarray(command_position, dtype=float)
    feedback_t_s = np.asarray(feedback_t_s, dtype=float)
    feedback_position = np.asarray(feedback_position, dtype=float)
    if len(command_t_s) < 2 or len(feedback_t_s) < 2:
        raise ValueError("need at least two command and feedback samples")
    window = (feedback_t_s >= command_t_s[0]) & (feedback_t_s <= command_t_s[-1])
    t_s = feedback_t_s[window]
    actual = feedback_position[window]
    if len(t_s) < 2:
        raise ValueError("no feedback samples inside the command window")
    lags_s = np.arange(-max_lag_s, max_lag_s + lag_step_s / 2., lag_step_s)
    summary = {}
    for hand, side in enumerate(("left", "right")):
        series = actual[:, hand]
        zero_error = series - interp_position(command_t_s, command_position[:, hand], t_s)
        mean_errors = np.array([
            np.abs(series - interp_position(command_t_s, command_position[:, hand],
                                            t_s - lag)).mean()
            for lag in lags_s])
        best = int(np.argmin(mean_errors))
        summary[side] = dict(
            samples=len(t_s),
            lag_s=float(lags_s[best]),
            lag_at_boundary=bool(best in (0, len(lags_s) - 1)),
            error_mean_mm=float(np.abs(zero_error).mean() * 1000.),
            error_max_mm=float(np.abs(zero_error).max() * 1000.),
            compensated_error_mean_mm=float(mean_errors[best] * 1000.),
        )
    return summary


def _speeds(t_s: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Per-sample implied speed of an (N, ..., 3) series on its own stamps."""
    if len(t_s) < 2:
        return np.zeros(len(t_s))
    return np.linalg.norm(np.diff(position, axis=0) / np.diff(t_s)[:, None], axis=-1)


def _pyplot_with_3d():
    """Import pyplot with the "3d" projection usable despite an apt/pip mix.

    matplotlib.projections runs `from mpl_toolkits.mplot3d import Axes3D` at
    module load and silently disables the 3d projection when that fails.  On
    the robot host the apt python3-matplotlib's regular mpl_toolkits package
    shadows the venv wheel's namespace copy (a regular package anywhere on
    sys.path beats namespace portions) and its Axes3D cannot import against
    the venv matplotlib.  Point mpl_toolkits at the copy next to the active
    matplotlib BEFORE pyplot pulls projections in, then re-register
    defensively in case projections was already loaded broken.  No-op on
    machines without the apt package.
    """
    import matplotlib
    import mpl_toolkits
    ours = Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
    if not any(Path(entry).resolve() == ours for entry in mpl_toolkits.__path__):
        mpl_toolkits.__path__.insert(0, str(ours))
    import matplotlib.pyplot as plt
    from matplotlib import projections
    if "3d" not in projections.get_projection_names():
        from mpl_toolkits.mplot3d import Axes3D
        projections.register_projection(Axes3D)
    return plt


def tracking_figure(command_t_s, command_phase, command_position,
                    feedback_t_s, feedback_position, tracking: dict,
                    contact_after_execute_s: float | None = None, title: str = ""):
    """Commanded-vs-actual tracking figure for one replay attempt.

    Columns are hands; rows: per-axis position vs time (command solid, actual
    dashed - same colour per axis), speed vs time, and the tracking error norm
    at zero lag vs lag-compensated; a bottom panel spans both hands with the
    3-D palm paths in base_link (start/end marked).  The execute span is
    shaded and, when given, the (scaled) contact time marked.

    matplotlib is imported lazily (a dev dependency - the runtime never needs
    it).  Labels are ASCII on purpose: the robot host's matplotlib ships
    DejaVu only, no CJK glyphs.  Headless saving works via the Agg backend
    (set matplotlib.use("Agg") before calling when there is no display).
    """
    plt = _pyplot_with_3d()

    command_t_s = np.asarray(command_t_s, dtype=float)
    feedback_t_s = np.asarray(feedback_t_s, dtype=float)
    command_position = np.asarray(command_position, dtype=float)
    feedback_position = np.asarray(feedback_position, dtype=float)
    t_cmd = command_t_s - command_t_s[0]
    t_fb = feedback_t_s - command_t_s[0]           # common zero: first command
    execute = np.asarray(command_phase) == "execute"
    ex_start, ex_end = (t_cmd[execute][[0, -1]] if execute.any() else (None, None))

    fig = plt.figure(figsize=(15., 12.5), layout="constrained")
    grid = fig.add_gridspec(6, 2, height_ratios=[1., 1., 1., 1.2, 1.2, 2.2])
    hands = ("left hand", "right hand")
    axis_names, axis_colors = ("x", "y", "z"), ("tab:blue", "tab:orange", "tab:green")
    in_window = (t_fb >= t_cmd[0]) & (t_fb <= t_cmd[-1])

    def decorate(ax, legend=False):
        if ex_start is not None:
            ax.axvspan(ex_start, ex_end, color="0.92", zorder=0,
                       label="execute span" if legend else None)
            if contact_after_execute_s is not None:
                ax.axvline(ex_start + contact_after_execute_s, color="red",
                           ls=":", lw=1., zorder=1,
                           label="contact t" if legend else None)
        if legend:
            ax.legend(fontsize=8, loc="best")

    for axis in range(3):
        for hand in range(2):
            ax = fig.add_subplot(grid[axis, hand])
            decorate(ax, legend=(axis == 0 and hand == 0))
            ax.plot(t_cmd, command_position[:, hand, axis], color=axis_colors[axis],
                    lw=2., label="command")
            ax.plot(t_fb, feedback_position[:, hand, axis], color=axis_colors[axis],
                    lw=1., ls="--", alpha=.85, label="actual (FK)")
            ax.set_ylabel(f"{axis_names[axis]} [m]")
            if axis == 0:
                ax.set_title(hands[hand])
            if axis == 2:
                ax.set_xlabel("t since first command [s]")

    for hand in range(2):
        ax = fig.add_subplot(grid[3, hand])
        decorate(ax)
        ax.plot(t_cmd[1:], _speeds(command_t_s, command_position[:, hand]),
                color="0.3", lw=2., label="command")
        ax.plot(t_fb[1:], _speeds(feedback_t_s, feedback_position[:, hand]),
                color="tab:red", lw=1., ls="--", alpha=.85, label="actual (FK)")
        ax.set_ylabel("speed [m/s]")
        ax.set_title(f"{hands[hand]} speed")
        if hand == 0:
            ax.legend(fontsize=8, loc="best")

    for hand, side in enumerate(("left", "right")):
        ax = fig.add_subplot(grid[4, hand])
        decorate(ax)
        lag = tracking[side]["lag_s"]
        series = feedback_position[in_window][:, hand]
        zero_error = np.linalg.norm(
            series - interp_position(command_t_s, command_position[:, hand],
                                     feedback_t_s[in_window]), axis=-1)
        compensated = np.linalg.norm(
            series - interp_position(command_t_s, command_position[:, hand],
                                     feedback_t_s[in_window] - lag), axis=-1)
        ax.plot(t_fb[in_window], zero_error * 1000., color="tab:red", lw=1.5,
                label="error (zero lag)")
        ax.plot(t_fb[in_window], compensated * 1000., color="tab:green", lw=1.2,
                ls="--", label=f"after {lag * 1000:+.0f} ms lag compensation")
        ax.set_ylabel("position error [mm]")
        ax.set_title(f"{hands[hand]} tracking error "
                     f"(max {tracking[side]['error_max_mm']:.1f} mm)")
        if hand == 0:
            ax.legend(fontsize=8, loc="best")

    ax3d = fig.add_subplot(grid[5, :], projection="3d")
    for hand in range(2):
        color = "tab:blue" if hand == 0 else "tab:orange"
        ax3d.plot(command_position[:, hand, 0], command_position[:, hand, 1],
                  command_position[:, hand, 2], color=color, lw=2.,
                  label=f"{hands[hand]} command")
        ax3d.plot(feedback_position[:, hand, 0], feedback_position[:, hand, 1],
                  feedback_position[:, hand, 2], color=color, lw=1., ls="--",
                  alpha=.85, label=f"{hands[hand]} actual")
        ax3d.scatter(*command_position[0, hand], color=color, marker="o", s=25)
        ax3d.scatter(*command_position[-1, hand], color=color, marker="X", s=40)
    ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
    for setter in (ax3d.xaxis.set_major_locator, ax3d.yaxis.set_major_locator,
                   ax3d.zaxis.set_major_locator):
        setter(plt.MaxNLocator(4))            # fewer ticks: no clipped label pile-ups
    ax3d.tick_params(labelsize=7)
    handles, labels = ax3d.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.3", marker="o", ls="", label="start"),
                plt.Line2D([], [], color="0.3", marker="X", ls="", label="end")]
    ax3d.legend(handles=handles, fontsize=8, loc="upper left")
    ranges = np.array([np.ptp(command_position[:, :, i]) for i in range(3)])
    span = max(ranges.max(), 1e-6)               # degenerate (planar) paths stay plottable
    ax3d.set_box_aspect(np.maximum(ranges, span * 0.05) / span)
    fig.suptitle(title, fontsize=13)
    return fig
