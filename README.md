# Per-user reward models from sparse, conflicting ratings

Shared temporal encoder + per-user linear heads, trained on binary trajectory ratings from
simulated users with known preferences (noisy-AND likelihood, `P(y=1) = prod_t sigmoid(r_k(t))`).
Compared against a single pooled head on rating accuracy, credit assignment, and
best-of-N reranking. See `proposal.txt`.

```
prefrm/        env-agnostic: model.py, users.py, eval.py, run.py
envs/toy/      2D driving toy (fast, exact ground truth)
envs/carla/    CARLA 0.9.16: perturbed BehaviorAgent (agent.py), collection (collect.py)
envs/franka/   Manipulator: pick and place with a cup near the target
scripts/       carla_server.sh, collect.sh
```

An env is a package exposing `FEATURES`, `THR_RANGES`, `GROUPS`, `N_RANDOM`, `add_args`,
`load`, `observations`, `feature_mask` (and optionally `plot_topdown`); see `prefrm/run.py`.

## Setup

```bash
uv sync                                    # toy only
uv sync --extra carla                      # + CARLA client
uv sync --extra franka                     # + MuJoCo
uv run python -m envs.franka.setup_assets  # + Assets
```

CARLA server: `~/carla/CARLA_0.9.16` (override with `CARLA_ROOT`).

## Run

```bash
uv run python -m prefrm.run --env toy --eps 0.05
```

CARLA: collect shards (one fresh server per town), then train/eval:

```bash
scripts/collect.sh train  150 2  0   2000 0   # split, scenarios/town, candidates/scenario, seed, port, gpu
scripts/collect.sh rerank  25 16 100 3000 1
uv run python -m prefrm.run --env carla
```

Results land in `out/<env>/`.

## Franka Manipulation task

```bash
uv run python -m envs.franka.collect --scenarios 150 --cands 2 --seed 0 --out data/franka/train.npz
uv run python -m envs.franka.collect --scenarios 25 --cands 16 --seed 100 --out data/franka/rerank.npz
uv run python -m prefrm.run --env franka --eps 0.05
```


## CARLA notes

- Only the `_Opt` maps load; `load_world` on e.g. `Town03` segfaults the 0.9.16 server.
- A server binds `port`, `port+1`, `port+2`; space parallel servers apart (2000, 3000, ...).
- `hard_brake` uses the brake command, not measured decel: CARLA snaps to a stop at low speed
  (~25 m/s^2 spikes at the end of every stop), and BehaviorAgent brakes at 0.5 at every red light.
- `wobble` ignores junctions, where normal turns dominate yaw-rate high-pass.
