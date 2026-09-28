"""One attempt of the one-shot catch, as a single-threaded state machine.

    WAIT    no usable box tracking
    ARMED   box quasi-static inside the release region; operator may throw
    FLIGHT  release detected; rolling observations and previews, committing on
            the first fresh observation that crosses the commit plane
    EXECUTING  plan frozen; streaming plan.target(t) until stop + margin
    REJECTED   hold/blend back to the wait pose
    DONE       hold the final pose (post-catch)

Everything runs on one host monotonic clock (time.perf_counter base).  The
observation timestamps are mapped onto that clock by the source; planning
latency compensation flows through plan_catch's retiming (execution_delay_s =
observation age + configured command latency + finalize ticks).

After commit the source is still drained and logged (but ignored by planning),
so every attempt keeps the full box trajectory for offline residual analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time

import numpy as np

from moz1_catch.config import Config
from moz1_catch.core.geometry import PALM_NORMAL_AXES_BODY, PALM_SPHERE_OFFSETS_BODY_M
from moz1_catch.core.one_shot import plan_catch
from moz1_catch.core.prediction import estimate_box_flight
from moz1_catch.executor import Executor
from moz1_catch.mocap.source import BoxObservationSource
from moz1_catch.robot.sink import palm_state_snapshot
from moz1_catch.trace import TraceRecorder

WAIT, ARMED, FLIGHT, EXECUTING, REJECTED, DONE, FAULT = (
    "WAIT", "ARMED", "FLIGHT", "EXECUTING", "REJECTED", "DONE", "FAULT")


@dataclass
class AttemptResult:
    decision: str          # accept | reject | abort | fault
    reason: str
    state: str
    trace_dir: str | None = None
    planning_time_ms: float = float("nan")
    observation_age_s: float = float("nan")
    execution_delay_s: float = float("nan")
    contact_time_s: float = float("nan")
    geometry_score: float = float("nan")


class CatchRuntime:
    def __init__(self, config: Config, source: BoxObservationSource, executor: Executor,
                 trace: TraceRecorder, palm_state_provider=None, log=print):
        self._config = config
        self._source = source
        self._executor = executor
        self._trace = trace
        self._log = log
        self._palm_state = palm_state_provider or self._configured_palm_state

    def _configured_palm_state(self):
        robot = self._config.robot
        return palm_state_snapshot(
            np.array([hand.wait_position_m for hand in robot.hands]),
            np.array([hand.wait_quat_xyzw for hand in robot.hands]))

    # -- observation handling -------------------------------------------------

    def _speed(self, observation) -> float:
        if self._last_position is None or observation.t_s <= self._last_t:
            return float("inf")
        return float(np.linalg.norm(observation.position_m - self._last_position)
                     / (observation.t_s - self._last_t))

    def _inside_release_region(self, position) -> bool:
        region = self._config.arming.release_region
        return bool(np.all(position >= region[:, 0]) and np.all(position <= region[:, 1]))

    def _on_observation(self, observation, state) -> tuple[str, bool]:
        """Advance the state machine by one observation; True when terminal."""
        arming = self._config.arming
        mission = self._config.mission
        if observation.valid:
            speed = self._speed(observation)
        else:
            speed = float("nan")
        self._last_position = observation.position_m if observation.valid else None
        self._last_t = observation.t_s

        if state == WAIT:
            if not observation.valid or not np.isfinite(speed):
                self._arm_since = None
            elif self._inside_release_region(observation.position_m):
                if speed <= arming.max_speed_mps:
                    if self._arm_since is None:
                        self._arm_since = observation.t_s
                    elif observation.t_s - self._arm_since >= arming.hold_s:
                        self._log(f"catch_state=ARMED box_y={observation.position_m[1]:+.3f}"
                                  " - operator may throw")
                        return ARMED, False
                elif speed >= arming.release_speed_threshold_mps:
                    # The box is already flying inside the release region (short or
                    # absent quasi-static hold): treat this observation as the release.
                    self._release_t = observation.t_s
                    self._flight_t = []
                    self._flight_poses = []
                    self._previews = 0
                    self._log(f"catch_state=FLIGHT release_t={observation.t_s:.4f}"
                              " (detected before arming)")
                    return FLIGHT, False
        elif state == ARMED:
            if not observation.valid:
                self._arm_since = None
                self._trace.event("tracking lost while armed; back to WAIT")
                return WAIT, False
            if speed >= arming.release_speed_threshold_mps:
                self._release_t = observation.t_s
                self._flight_t = []
                self._flight_poses = []
                self._previews = 0
                self._log(f"catch_state=FLIGHT release_t={observation.t_s:.4f}")
                return FLIGHT, False
            if not self._inside_release_region(observation.position_m):
                self._arm_since = None
                self._trace.event("armed box left the release region; back to WAIT")
                return WAIT, False
        elif state == FLIGHT:
            if observation.t_s - self._release_t > mission.commit_timeout_s:
                return REJECTED, self._reject("commit timeout: box did not reach the commit plane")
            if not observation.valid:
                return FLIGHT, False
            self._flight_t.append(observation.t_s)
            self._flight_poses.append(np.r_[observation.position_m, observation.quat_xyzw])
            if observation.t_s - self._release_t >= mission.release_settle_s:
                self._preview(observation)
            if (observation.position_m[1] >= mission.commit_plane_y_m
                    and observation.t_s - self._release_t >= mission.min_commit_delay_s):
                return self._commit(observation)
        return state, False

    def _preview(self, observation) -> None:
        times = np.asarray(self._flight_t)
        poses = np.asarray(self._flight_poses)
        use = times >= observation.t_s - self._config.prediction.settings.window_s
        if use.sum() < 3:
            return
        try:
            flight = estimate_box_flight(times[use], poses[use], observation.t_s,
                                          self._config.prediction.settings,
                                          self._config.prediction.max_observation_age_s)
            crossing = flight.crossing_time(self._config.catch_settings.plane_y)
            self._trace.record_preview(observation.t_s, flight.positions(crossing))
            self._previews += 1
        except ValueError:
            pass

    def _reject(self, reason: str) -> bool:
        self._log(f"catch_decision=reject reason={reason}")
        self._reason = reason
        self._executor.enter_reject(time.perf_counter())
        return True

    def _commit(self, observation) -> tuple[str, bool]:
        mission = self._config.mission
        prediction = self._config.prediction
        times = np.asarray(self._flight_t)
        poses = np.asarray(self._flight_poses)
        settled = times >= self._release_t + mission.release_settle_s
        print_fields = (f"samples={settled.sum()} window_s={prediction.settings.window_s:.3f}"
                        f" previews={self._previews}")
        try:
            flight = estimate_box_flight(times[settled], poses[settled], observation.t_s,
                                          prediction.settings, prediction.max_observation_age_s)
        except ValueError as error:
            return REJECTED, self._reject(f"estimation: {error}")
        self._log(f"catch_observation_time_s={observation.t_s:.4f} {print_fields}"
                  f" acceleration_mps2={flight.acceleration_mps2.round(5).tolist()}")
        palm_positions, palm_rotations = self._palm_state()
        planning_start = time.perf_counter()
        observation_age = planning_start - observation.t_s
        try:
            plan = plan_catch(
                flight.position_m, flight.velocity_mps, flight.rotation,
                flight.angular_velocity_radps, palm_positions, palm_rotations,
                np.asarray(PALM_NORMAL_AXES_BODY), self._config.catch_settings,
                PALM_SPHERE_OFFSETS_BODY_M,
                execution_delay_s=observation_age + self._config.execution.command_latency_s,
                control_dt_s=self._config.execution.command_period_s,
                planning_started_s=planning_start,
                box_acceleration=flight.acceleration_mps2)
        except ValueError as error:
            return REJECTED, self._reject(str(error))
        planning_ms = 1000 * (time.perf_counter() - planning_start)
        exec_start = observation.t_s + plan.execution_delay_s
        self._executor.arm_plan(plan, exec_start)
        self._log(f"catch_planning_time_ms={planning_ms:.3f}"
                  f" observation_age_s={observation_age:.4f}"
                  f" geometry_score={plan.geometry_score:.5f}"
                  f" plane_y={plan.settings.plane_y:.3f}"
                  f" geometry_mode={plan.geometry_mode}")
        self._log(f"catch_decision=accept contact_time_s={plan.contact_time:.4f}"
                  f" stop_time_s={plan.stop_time:.4f}"
                  f" execution_delay_s={plan.execution_delay_s:.4f}"
                  f" relative_tangent_mps={plan.relative_tangent_speed:.4f}")
        self._plan = plan
        self._planning_ms = planning_ms
        self._observation_age = observation_age
        self._commit_observation = observation
        self._commit_flight = flight
        return EXECUTING, False

    # -- main loop ------------------------------------------------------------

    def run(self, max_wait_s: float = 60.0) -> AttemptResult:
        period = self._config.execution.command_period_s
        watchdog = self._config.safety.watchdog_s
        state = WAIT
        self._arm_since = None
        self._last_position = None
        self._last_t = None
        self._release_t = None
        self._flight_t: list[float] = []
        self._flight_poses: list[np.ndarray] = []
        self._previews = 0
        self._reason = ""
        self._plan = None
        self._planning_ms = float("nan")
        self._observation_age = float("nan")
        self._commit_observation = None
        self._commit_flight = None
        started = time.perf_counter()
        next_publish = started
        last_publish = started
        decision = "abort"
        self._trace.event(f"runtime start (state {WAIT})")

        while True:
            now = time.perf_counter()
            timeout = max(0., min(next_publish - now, 0.05))
            observation = self._source.next(timeout)
            if observation is not None:
                self._trace.record_observation(observation)
                try:
                    state, terminal = self._on_observation(observation, state)
                except Exception as error:  # noqa: BLE001 - any bridge bug must not swing the arms
                    state = FAULT
                    decision = "fault"
                    self._reason = f"runtime exception: {error!r}"
                    self._trace.event(self._reason)
                    break
                if terminal:
                    assert state == REJECTED
                    decision = "reject"
                    break
            now = time.perf_counter()
            if now >= next_publish:
                try:
                    self._executor.tick(now)
                except Exception as error:  # noqa: BLE001 - as above
                    state = FAULT
                    decision = "fault"
                    self._reason = f"executor exception: {error!r}"
                    self._trace.event(self._reason)
                    break
                last_publish = now
                next_publish = now + period
                if state == EXECUTING and self._executor.phase == "done":
                    state = DONE
                    decision = "accept"
                    break
            if now - last_publish > watchdog:
                state = FAULT
                decision = "fault"
                self._reason = "publication watchdog exceeded"
                self._trace.event(self._reason)
                break
            if state in (WAIT, ARMED) and now - started > max_wait_s:
                self._reason = "no throw detected within max wait"
                decision = "abort"
                break
            if state == FLIGHT and self._release_t is not None \
                    and now - self._release_t > self._config.mission.commit_timeout_s + 0.5:
                state = REJECTED
                decision = "reject"
                self._reason = "commit plane not reached (wall-clock guard)"
                self._log(f"catch_decision=reject reason={self._reason}")
                self._executor.enter_reject(now)
                break

        self._trace.event(f"runtime end (state {state}, reason {self._reason!r})")
        extra = self._trace_extra()
        trace_dir = self._trace.save(self._config.logging.output_dir, decision,
                                     self._reason, extra)
        self._log(f"catch_trace_dir={trace_dir}")
        return AttemptResult(
            decision=decision, reason=self._reason, state=state, trace_dir=str(trace_dir),
            planning_time_ms=self._planning_ms, observation_age_s=self._observation_age,
            execution_delay_s=(self._plan.execution_delay_s if self._plan else float("nan")),
            contact_time_s=(self._plan.contact_time if self._plan else float("nan")),
            geometry_score=(self._plan.geometry_score if self._plan else float("nan")))

    def _trace_extra(self) -> dict:
        extra = dict(commit_plane_y_m=self._config.mission.commit_plane_y_m,
                     planning_time_ms=self._planning_ms,
                     observation_age_s=self._observation_age,
                     catch_settings_json=json.dumps(asdict(self._config.catch_settings)))
        if self._plan is not None:
            plan, flight = self._plan, self._commit_flight
            extra.update(
                contact_time_s=plan.contact_time,
                execution_latency_s=plan.execution_delay_s,
                contact_time_from_observation_s=plan.execution_delay_s + plan.contact_time,
                geometry_mode=plan.geometry_mode, geometry_score=plan.geometry_score,
                contact_positions_m=plan.contact_positions, stop_positions_m=plan.stop_positions,
                contact_normals=plan.contact_normals,
                retreat_velocity_mps=plan.retreat_velocity, retreat_durations_s=plan.retreat_durations,
                predicted_touch_times_s=np.asarray(plan.predicted_touch_times_s),
                predicted_touch_normal_cosines=np.asarray(plan.predicted_touch_normal_cosines),
                relative_tangent_mps=np.asarray(plan.relative_tangent_speed))
        if self._commit_flight is not None:
            flight = self._commit_flight
            extra.update(
                estimated_box_position_m=flight.position_m,
                estimated_box_velocity_mps=flight.velocity_mps,
                estimated_box_rotation_xyzw=flight.rotation.as_quat(),
                estimated_box_angular_velocity_radps=flight.angular_velocity_radps,
                estimated_box_acceleration_mps2=flight.acceleration_mps2)
        if self._commit_observation is not None:
            extra.update(
                observation_box_pose=np.r_[self._commit_observation.position_m,
                                           self._commit_observation.quat_xyzw])
        return extra
