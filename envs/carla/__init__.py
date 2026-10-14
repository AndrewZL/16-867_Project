"""CARLA env: BehaviorAgent rollouts, collected offline by envs/carla/collect.py.

Training data mixes clean rollouts (stock 'normal' BehaviorAgent) with perturbed ones
(sampled style + injected events); --perturbed-frac sets the mix. User thresholds are
calibrated so the clean driver violates each feature in only ~3-10% of episodes.

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
            "red_light", "too_slow", "collision"]

# Ranges chosen so the clean driver fires each feature in ~3% (lenient end) to ~10% (strict
# end) of episodes; hard_brake, speeding, red_light never fire on the clean driver.
THR_RANGES = dict(
    wobble=(15.0, 30.0),      # |yaw rate - 1 s moving average| deg/s, moving, outside junctions
    hard_brake=(0.55, 0.65),  # brake command while moving (agent: <=0.3 normal, 0.5 at every red light)
    tailgate=(0.5, 0.7),      # time headway s
    lane_offset=(0.8, 1.4),   # m from lane center (outside junctions and lane changes)
    ped_close=(2.0, 3.2),     # m to nearest pedestrian while moving
    speeding=(0.5, 2.5),      # m/s over the speed limit
    red_light=(0.5, 2.0),     # speed m/s while running a red
    too_slow=(5.9, 6.5),      # cruising speed m/s on clear road (global; default driver ~7.2)
    collision=(0.5, 0.5),     # any contact
)
SHARED_FEATURES = ["collision"]  # every user dislikes these

# BehaviorAgent's stock 'normal' profile (km/h): the unperturbed reference driver
DEFAULT_STYLE = dict(max_speed=50.0, speed_lim_dist=3.0, speed_decrease=10.0, safety_time=3.0,
                     braking_distance=5.0, min_proximity_threshold=10.0, ignore_lights=False)

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
        # cruising speed: moving on clear road (no lead car, red light, junction or nearby
        # pedestrian), so queues and stops aren't blamed on the driver; needs >= 3 s of it
        free = ((tr["lead_gap"] > 20) & (tr["at_red"] < 0.5) & (tr["in_junction"] < 0.5)
                & (tr["ped_dist"] > 10) & (v > 0.5))
        n = free.sum(1)
        slow = (n >= 30) & ((v * free).sum(1) / np.maximum(n, 1) < thr)
        return np.broadcast_to(slow[:, None], v.shape).copy()
    if feat == "collision":
        return tr["collision"] > 0.5
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
    ap.add_argument("--data", default="data/carla/clean_*.npz,data/carla/train_*.npz",
                    help="comma-separated globs of collected shards")
    ap.add_argument("--perturbed-frac", type=float, default=0.3,
                    help="fraction of training episodes that are perturbed (0 = clean driving only)")
    ap.add_argument("--rerank-data", default="data/carla/rerank2_*.npz",
                    help="glob of shards with several candidates per scenario ('' to skip)")


def load_shards(patterns):
    files = sorted(f for p in patterns.split(",") for f in glob.glob(p))
    if not files:
        return None
    parts = [dict(np.load(f)) for f in files]
    keys = set.intersection(*(set(p) for p in parts))
    tr = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    tr["perturbed"] = is_perturbed(tr)
    return tr


def is_perturbed(tr):
    ev = np.stack([tr[f"event_{k}"] >= 0 for k in ("wobble", "brake", "drift")]).any(0)
    style = np.stack([~np.isclose(tr[f"style_{k}"], float(v)) for k, v in DEFAULT_STYLE.items()]).any(0)
    return ev | style


def mix(tr, frac, rng):
    """Subsample so that a `frac` share of episodes is perturbed, keeping as many as possible."""
    pert, clean = np.where(tr["perturbed"])[0], np.where(~tr["perturbed"])[0]
    if frac <= 0:
        keep = clean
    elif frac >= 1:
        keep = pert
    else:
        n_clean = min(len(clean), int(len(pert) * (1 - frac) / frac))
        n_pert = min(len(pert), int(round(n_clean * frac / (1 - frac))))
        keep = np.concatenate([rng.choice(clean, n_clean, replace=False), rng.choice(pert, n_pert, replace=False)])
    return {k: v[np.sort(keep)] for k, v in tr.items()}


def load(split, args, rng):
    if split == "train":
        tr = load_shards(args.data)
        if tr is None:
            raise FileNotFoundError(f"no shards match {args.data}; run scripts/collect.sh first")
        tr = mix(tr, args.perturbed_frac, rng)
        print(f"  carla train: {len(tr['v'])} episodes, {tr['perturbed'].mean():.2f} perturbed")
        return tr
    return load_shards(args.rerank_data) if args.rerank_data else None


def plot_topdown(ax, tr, i):
    ax.plot(tr["x"][i], tr["y"][i], "k-", lw=1)
    ax.plot(tr["x"][i, 0], tr["y"][i, 0], "go", label="start")
    peds = tr["ped_dist"][i] < 8
    ax.plot(tr["x"][i][peds], tr["y"][i][peds], "r.", ms=3, label="ped < 8 m")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=7)
