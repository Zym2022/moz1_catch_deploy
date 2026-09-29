# Simulation plan traces for hardware replay

Frozen planning results exported from the MozBoxer simulation, replayed on the
robot by `scripts/replay_sim_plan.py` (no box) to verify the deployment
execution chain.

| file | provenance |
| --- | --- |
| `final_nominal_120hz_200ms.npz` | `MozBoxer/output/mocap_prediction_20260929/final_nominal_120hz_200ms.npz`, branch `feat/moz1-swept-geometry-catch` @ `f603888`, accept decision, dt = 1 ms, contact at 0.233 s after execution start |

Schema (subset used by the replay, base_link frame):

- `time_s` (N,), dt = 1 ms.  This artifact is plan-relative (index 0 = plan
  start); live-run traces are episode-absolute instead.  The loader
  (`moz1_catch.sim_plan`) auto-detects which segment convention a trace uses.
- `target_palm_position` (N, 2, 3), `target_palm_rotation_xyzw` (N, 2, 4),
  `target_palm_velocity` (N, 2, 3) - commanded palm targets, hands [left, right]
- `observation_time_s`, `execution_latency_s`, `contact_time_s`,
  `catch_decision`, `contact_positions_m`, `start_palm_positions_m` - used to
  locate and validate the plan segment
