"""CARLA env: rollouts of a perturbed BehaviorAgent, collected offline by envs/carla/collect.py.

Importing this module does not require a running server or the carla package; it only
loads collected .npz shards and defines observations / violation detectors over the
per-step privileged state that collect.py logs (see STEP_KEYS).
"""
import glob

import numpy as np

DT = 0.1  # logging period (s)

# per-step [N, T] arrays logged by collect.py
STEP_KEYS = ["x", "y", "yaw", "v", "a_lon", "a_lat", "yaw_rate", "steer", "throttle", "brake",
             "speed_limit", "lane_offset", "in_junction", "lane_id", "road_id", "lead_gap", "lead_closing",
             "ped_dist", "ped_dx", "ped_dy", "at_red", "red_run", "collision"]

FEATURES = ["wobble", "hard_brake", "tailgate", "lane_offset", "ped_close", "speeding",
            "red_light", "too_slow"]

THR_RANGES = dict(
    wobble=(12.0, 20.0),      # |yaw rate - 1 s moving average| deg/s, moving, outside junctions
    hard_brake=(0.55, 0.65),  # brake command while moving (agent: <=0.3 normal, 0.5 at every red light)
    tailgate=(1.0, 2.0),      # time headway s
    lane_offset=(0.6, 1.0),   # m from lane center (outside junctions and lane changes)
    ped_close=(3.0, 5.0),     # m to nearest pedestrian while moving
    speeding=(0.5, 2.5),      # m/s over the speed limit
    red_light=(0.5, 2.0),     # speed m/s while running a red
    too_slow=(4.0, 6.0),      # mean speed m/s (global)
)

GROUPS = [
    ("smooth", ["wobble", "hard_brake"], 4),
    ("safety", ["tailgate", "red_light"], 4),
    ("eager", ["too_slow"], 3),
    ("calm", ["speeding", "hard_brake"], 2),
    ("rare", ["ped_close"], 1),
]
N_RANDOM = 6


def _moving_avg(a, w):
    k = np.ones(w) / w
    pad = np.pad(a, ((0, 0), (w // 2, w // 2)), mode="edge")
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, pad)


def feature_mask(tr, feat, thr):
    v = tr["v"]
    if feat == "wobble":
        hp = np.abs(tr["yaw_rate"] - _moving_avg(tr["yaw_rate"], 11))
        return (hp > thr) & (tr["in_junction"] < 0.5) & (v > 3)  # junction turns aren't wobble
    if feat == "hard_brake":
        # commanded brake, not measured decel: CARLA snaps to a stop at low speed, which
        # makes kinematic decel spike to ~25 m/s^2 at the end of every stop
        return (tr["brake"] > thr) & (v > 3)
    if feat == "tailgate":
        return (tr["lead_gap"] < 40) & (v > 2) & (tr["lead_gap"] / np.maximum(v, 0.1) < thr)
    if feat == "lane_offset":
        lane = tr["road_id"] * 100 + tr["lane_id"]
        switch = np.zeros(v.shape)
        switch[:, 1:] = lane[:, 1:] != lane[:, :-1]
        near_switch = _moving_avg(switch, 31) > 0      # within 1.5 s of a lane/road change
        return (np.abs(tr["lane_offset"]) > thr) & (tr["in_junction"] < 0.5) & ~near_switch
    if feat == "ped_close":
        return (tr["ped_dist"] < thr) & (v > 1)
    if feat == "speeding":
        return v > tr["speed_limit"] + thr
    if feat == "red_light":
        return (tr["red_run"] > 0.5) & (v > thr)
    if feat == "too_slow":
        return np.broadcast_to((v.mean(1) < thr)[:, None], v.shape).copy()
    raise ValueError(feat)


def observations(tr):
    """Per-step privileged state [N, T, D]. Thresholds and event structure must be learned."""
    return np.stack([
        tr["v"], tr["a_lon"], tr["a_lat"], tr["yaw_rate"], tr["steer"], tr["throttle"], tr["brake"],
        tr["speed_limit"], tr["lane_offset"], tr["in_junction"],
        np.minimum(tr["lead_gap"], 40), np.where(tr["lead_gap"] < 40, tr["lead_closing"], 0),
        np.minimum(tr["ped_dist"], 30), np.clip(tr["ped_dx"], -30, 30), np.clip(tr["ped_dy"], -30, 30),
        tr["at_red"],
    ], -1).astype(np.float32)


def add_args(ap):
    ap.add_argument("--data", default="data/carla/train_*.npz", help="glob of collected shards")
    ap.add_argument("--rerank-data", default="data/carla/rerank_*.npz",
                    help="glob of shards with several candidates per scenario ('' to skip)")


def load_shards(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load(split, args, rng):
    pattern = args.data if split == "train" else args.rerank_data
    tr = load_shards(pattern) if pattern else None
    if tr is None and split == "train":
        raise FileNotFoundError(f"no shards match {pattern}; run python -m envs.carla.collect first")
    return tr


def plot_topdown(ax, tr, i):
    ax.plot(tr["x"][i], tr["y"][i], "k-", lw=1)
    ax.plot(tr["x"][i, 0], tr["y"][i, 0], "go", label="start")
    peds = tr["ped_dist"][i] < 8
    ax.plot(tr["x"][i][peds], tr["y"][i][peds], "r.", ms=3, label="ped < 8 m")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=7)
