"""Streams pose targets to the controller for one attempt.

Phases per attempt:

  hold     - publish the configured wait pose (states WAIT/ARMED/FLIGHT)
  execute  - sample plan.target(t) at the command period between the execution
             start and contact_time + stop_time + settle margin
  reject   - quintic blend back to the wait pose, then hold
  done     - hold the final pose (after a catch the palms stay where they stopped)

Targets are produced in the palm frames, clamped by the safety envelope, then
converted to controller TCP frames right before publishing.  The planner's
velocity output is not sent anywhere: the controller only accepts poses; it is
logged so tracking can be analysed offline.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from moz1_catch.calib import transform_inverse
from moz1_catch.config import ExecutionConfig, RobotConfig
from moz1_catch.robot.sink import CartesianCommandSink, HandTargets, palm_targets_to_tcp
from moz1_catch.safety import SafetyEnvelope
from moz1_catch.trace import TraceRecorder


class Executor:
    def __init__(self, sink: CartesianCommandSink, safety: SafetyEnvelope,
                 execution: ExecutionConfig, robot: RobotConfig, trace: TraceRecorder):
        self._sink = sink
        self._safety = safety
        self._execution = execution
        self._trace = trace
        self._T_tcp_palms = tuple(hand.T_tcp_palm for hand in robot.hands)
        self._T_torso_base = transform_inverse(robot.T_base_torso)
        self._wait_targets = HandTargets(
            positions_m=np.array([hand.wait_position_m for hand in robot.hands]),
            rotations=tuple(Rotation.from_quat(hand.wait_quat_xyzw) for hand in robot.hands),
        )
        self._phase = "hold"
        self._plan = None
        self._exec_start = None
        self._end_t = None
        self._reject_start = None
        self._reject_from = None
        self._reject_rotations = None
        self._previous_palm = self._wait_targets.positions_m.copy()
        self.last_sent: HandTargets | None = None

    @property
    def phase(self) -> str:
        return self._phase

    def arm_plan(self, plan, exec_start_t: float) -> None:
        self._plan = plan
        self._exec_start = exec_start_t
        self._end_t = exec_start_t + plan.contact_time + plan.stop_time + self._execution.settle_margin_s
        self._phase = "execute"

    def enter_reject(self, t_host: float) -> None:
        if self.last_sent is not None:
            self._reject_from = self.last_sent.positions_m
            self._reject_rotations = self.last_sent.rotations
        else:
            self._reject_from = self._wait_targets.positions_m
            self._reject_rotations = self._wait_targets.rotations
        self._reject_start = t_host
        self._phase = "reject"

    def _quintic(self, start: np.ndarray, end: np.ndarray, s: float) -> np.ndarray:
        blend = 10*s**3 - 15*s**4 + 6*s**5
        return start + blend * (end - start)

    def _blend_rotation(self, start: Rotation, end: Rotation, s: float) -> Rotation:
        vector = (end * start.inv()).as_rotvec()
        return Rotation.from_rotvec((10*s**3 - 15*s**4 + 6*s**5) * vector) * start

    def _publish(self, t_host: float, t_plan: float, targets: HandTargets,
                 velocities: np.ndarray) -> None:
        clamped_positions, violations = self._safety.clamp(
            targets.positions_m, self._previous_palm, self._execution.command_period_s)
        clamped = HandTargets(positions_m=clamped_positions, rotations=targets.rotations)
        tcp_targets = palm_targets_to_tcp(clamped, self._T_tcp_palms, self._T_torso_base)
        self._sink.send(tcp_targets, t_host)
        self._previous_palm = clamped_positions
        self.last_sent = clamped
        self._trace.record_command(t_host, self._phase, t_plan, clamped, tcp_targets,
                                   velocities, violations)

    def tick(self, t_host: float) -> str:
        """Publish one command; returns the phase after publishing."""
        period = self._execution.command_period_s
        if self._phase == "hold":
            self._publish(t_host, 0.0, self._wait_targets, np.zeros((2, 3)))
            return "hold"
        if self._phase == "execute":
            plan = self._plan
            t_plan = float(np.clip(t_host - self._exec_start, 0.,
                                   plan.contact_time + plan.stop_time))
            positions, rotations, velocities, _ = plan.target(t_plan)
            self._publish(t_host, t_plan, HandTargets(positions, tuple(rotations)), velocities)
            if t_host >= self._end_t:
                self._phase = "done"
            return self._phase
        if self._phase == "reject":
            s = float(np.clip((t_host - self._reject_start) / self._execution.reject_duration_s, 0., 1.))
            positions = self._quintic(self._reject_from, self._wait_targets.positions_m, s)
            rotations = tuple(self._blend_rotation(start, end, s) for start, end
                              in zip(self._reject_rotations, self._wait_targets.rotations))
            self._publish(t_host, 0.0, HandTargets(positions, rotations), np.zeros((2, 3)))
            if s >= 1.:
                self._phase = "done"
            return self._phase
        # done: keep publishing whatever the palms last held
        self._publish(t_host, 0.0,
                      HandTargets(self._previous_palm,
                                  self.last_sent.rotations if self.last_sent
                                  else self._wait_targets.rotations),
                      np.zeros((2, 3)))
        return "done"

    def hold_final_forever(self, log=print) -> None:
        """After a finished attempt, keep the last pose commanded until interrupted."""
        period = self._execution.command_period_s
        log("attempt finished; holding final pose - Ctrl+C to exit")
        try:
            while True:
                self.tick(time.perf_counter())
                time.sleep(period)
        except KeyboardInterrupt:
            log("hold released")
