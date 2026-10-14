"""2D toy driving env (see env.py). Fast, exact ground truth; used to debug the method."""
import numpy as np

from .env import (DT, PED_Y, T, ZONE_LEN, FEATURES, feature_mask, observations,  # noqa: F401
                  repeat_scenarios, rollout, sample_params, sample_scenarios)

THR_RANGES = dict(
    wobble=(1.8, 2.6),       # |lateral accel| m/s^2
    hard_brake=(3.5, 5.0),   # decel m/s^2
    tailgate=(0.8, 1.3),     # time headway s
    ped_close=(1.2, 1.8),    # lateral clearance m
    zone_speed=(0.5, 2.0),   # m/s over zone limit
    fast=(14.0, 16.0),       # m/s
    too_slow=(10.0, 11.5),   # mean speed m/s (global)
)

# "rare" is the long-tail user: the only one who cares about ped_close.
# "eager" vs "calm" conflict on speed but have a compatible band.
GROUPS = [
    ("smooth", ["wobble", "hard_brake"], 4),
    ("safety", ["tailgate", "zone_speed"], 4),
    ("eager", ["too_slow"], 3),
    ("calm", ["fast", "hard_brake"], 2),
    ("rare", ["ped_close"], 1),
]
N_RANDOM = 6


def add_args(ap):
    ap.add_argument("--n-traj", type=int, default=4000)
    ap.add_argument("--n-scen", type=int, default=400, help="rerank scenarios")
    ap.add_argument("--n-cand", type=int, default=32, help="rerank candidates per scenario")


def load(split, args, rng):
    n_scen, n_cand = (args.n_traj, 1) if split == "train" else (args.n_scen, args.n_cand)
    sc = repeat_scenarios(sample_scenarios(n_scen, rng), n_cand)
    tr = rollout(sc, sample_params(n_scen * n_cand, rng))
    tr["scenario"] = np.repeat(np.arange(n_scen), n_cand)
    return tr


def plot_topdown(ax, tr, i):
    ax.plot(tr["x"][i], tr["y"][i], "k-", label="ego")
    ax.plot(tr["lead_x"][i], np.zeros(T), "b--", alpha=.5, label="lead")
    ax.plot(tr["ped_x"][i], PED_Y, "r^", ms=10, label="pedestrian")
    ax.axvspan(tr["zone_x"][i], tr["zone_x"][i] + ZONE_LEN, color="y", alpha=.2, label="school zone")
    ax.set_ylim(-2, 7); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=7, loc="upper left")
