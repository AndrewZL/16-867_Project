# Franka block-to-bin task

The Menagerie Panda picks up a 4 cm, 60 g block and places it in an open bin.
A hollow glass mug with a handle stands near the bin. Both block and mug are
free bodies: the fingers grasp through contact/friction, with no weld or object
teleportation during a rollout. Glass is a transparent rigid material; it does not
shatter. The robot uses Menagerie's actuators and collision meshes, with bias-force
compensation for the seven arm joints. Bin, block and mug dimensions are fixed;
their positions vary between scenarios.

## How this fits the existing environments

* `envs/toy/env.py` generates batched analytic trajectories from separate scenario
  and planner samples. Its package defines preferences and generates data on demand.
* `envs/carla/agent.py` varies driving style and adds timed perturbations;
  `collect.py` records simulated rollouts offline. Its package loads shards and
  defines raw observations, feature masks and user groups.
* Franka follows the offline CARLA pattern. `planner.py` owns scenario sampling,
  style parameters and Cartesian targets; `env.py` owns the scene, IK, physics and
  state logging; `collect.py` writes compressed shards; `__init__.py` implements
  the existing environment interface and does not import MuJoCo.
* `prefrm/users.py` converts feature masks into noisy binary ratings.
  `model.py` learns a temporal encoder and pooled/per-user reward heads from raw
  observations. `run.py` splits by scenario and normalizes using training data;
  `eval.py` measures ratings, temporal credit assignment and candidate reranking.
  None of these shared components needed changes.

## Planner controls

`Style` is a frozen dataclass, directly usable with `FrankaEnv.rollout(scenario, style)`.
The sampler draws the four controls independently. They control intended motion;
tracking, contact, and actual measured attributes can interact.

| Parameter | Range | Meaning |
| --- | --- | --- |
| `speed` | 0.07–0.32 m/s | Average target speed on lift/transport segments; minimum segment duration 0.4 s |
| `jerkiness` | 0–1 | Blend from minimum-jerk timing to monotone stop/start modulation, preserving path and duration |
| `clearance` | 0.10–0.30 m | Intended block-bottom height above the tabletop during transport |
| `cup_margin` | 0–0.20 m | Horizontal gap between the planned block edge and cup shell at passage waypoints |

Approach, close, lift, pass the mug, move above bin, lower, release, and retract
form the motion sequence. The two passage waypoints run along the robot side of
the mug. Damped least-squares pose IK maps targets to joint commands at 50 Hz;
physics runs at 500 Hz. Approach/grasp/place use fixed careful timings so style
variation mainly affects carrying the object. This is a scripted waypoint planner,
not a global collision-free search or a guarantee of feasible/successful motion.

```python
from dataclasses import replace
import numpy as np
from envs.franka.env import FrankaEnv
from envs.franka.planner import Style, sample_scenario

env = FrankaEnv()
scenario = sample_scenario(np.random.default_rng(0))
slow = env.rollout(scenario, replace(Style(), speed=0.08))
close = env.rollout(scenario, replace(Style(), cup_margin=0.01))
```

## Preferences and real failures

| Group | Dislikes |
| --- | --- |
| `smooth` (4) | Measured high Cartesian jerk |
| `eager` (3) | Low average transport speed |
| `cautious` (3) | High speed or low carrying clearance |
| `compact` (3) | Unnecessarily high carrying or a large detour from the cup |
| `rare` (1) | Passing too close to the glass mug |

Six additional users have randomly combined preferences, excluding the rare
cup-close concern. Thresholds vary within `THR_RANGES`. Jerky trajectories can be
accepted by users who do not care about smoothness; close, fast and low trajectories
are likewise preferences, not automatically failures. Thresholds are initial
simulation settings, not claims about real robot safety or a calibrated population.

Every user also rejects `collision`, `drop`, and `incomplete`:

* Collision is unexpected physical contact (including robot self-contact,
  robot/world, cup/object and block/bin-wall contacts). Normal finger/block,
  initial block/table, block/bin-floor, and cup/table contacts are allowed.
  Contacts are checked every physics step, with a 10 micrometre penetration
  tolerance for robot/cup contacts.
* A drop is loss of the lifted block before release, detected from loss of finger
  support plus block/gripper separation or falling back to the table. Setting
  the block onto the bin floor during placement is allowed. Falling below the
  table also counts as a drop.
* `incomplete` marks the final step if the block was never lifted or is not
  contained and settled in the bin at the deadline. It prevents a missed grasp
  or missed bin from receiving approval without inventing a collision/drop.

`success` requires placement with no collision or drop anywhere in the episode.
Style violations do not change physical success. Failure flags remain local in
time for credit assignment; trajectory labels reject any occurrence.

## Observations and collection

Every episode has 260 samples at 0.1 s intervals, including real simulated settling
after completion. Fixed wall-clock sampling preserves speed information; there is
no temporal rescaling, padded state, or variable-length mask needed by the model.

`observations(tr)` concatenates these raw fields, in order, to `[N, 260, 65]`:

| Fields | Dimensions |
| --- | --- |
| Arm/finger `qpos`, `qvel` | 9 + 9 |
| Actuator `ctrl` | 8 |
| World-frame TCP position, linear velocity | 3 + 3 |
| Block position, quaternion (wxyz), spatial velocity | 3 + 4 + 6 |
| Cup position, quaternion (wxyz), spatial velocity | 3 + 4 + 6 |
| Bin position | 3 |
| Raw contact force magnitudes, averaged over each logging interval | 4 |

Spatial velocities are angular then linear, in world axes. Contact channels sum
finger/block, other robot contacts, block/non-robot, and cup/non-table forces.
These are privileged simulator measurements, not pixels or a claim of a deployed
sensor interface. Phase, measured cup distance, planner parameters, failure flags,
success, scenario IDs and user thresholds never enter the classifier. Ground-truth
feature masks use separate logged state/phase and geometry distance queries.

Candidate zero always uses `Style()` as a reproducible reference; additional
candidates sample styles. Use at least two candidates to collect diverse training
data. Scenario layouts repeat exactly across candidates, with IDs `seed:index`.
Use distinct seeds across shards and train/rerank splits. The collector checkpoints
after every complete scenario. `--reference-only` is useful for baseline checks;
`--model /path/to/panda.xml` accepts an existing compatible Menagerie checkout.

```bash
uv run --extra franka python -m unittest discover -s tests -p 'test_franka.py' -v
```

Simulation tests require the assets from `python -m envs.franka.setup_assets`.
For a headless renderer use `MUJOCO_GL=egl`; ordinary collection needs no renderer.
