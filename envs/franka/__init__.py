import glob

import numpy as np

from .planner import BLOCK_HALF, DT, T

FEATURES = ["fast", "too_slow", "jerky", "low_clearance", "high_clearance", "cup_close", "cup_far", "collision", "drop", "incomplete"]
THR_RANGES = dict(
    fast=(.28, .38), 
    too_slow=(.10, .14), 
    jerky=(5., 8.),
    low_clearance=(.13, .17), 
    high_clearance=(.23, .27),
    cup_close=(.045, .085), 
    cup_far=(.12, .17),
    collision=(.5, .5), 
    drop=(.5, .5), 
    incomplete=(.5, .5))

SHARED_FEATURES = ["collision", "drop", "incomplete"]             

GROUPS = [
    ("smooth", ["jerky"], 4),
    ("eager", ["too_slow"], 3),
    ("cautious", ["fast", "low_clearance"], 3),
    ("compact", ["high_clearance", "cup_far"], 3),
    ("rare", ["cup_close"], 1),
]
N_RANDOM = 6

OBS_KEYS = ["qpos", "qvel", "ctrl", "tcp", "tcp_vel", "block_pos", "block_quat",
            "block_vel", "cup_pos", "cup_quat", "cup_vel", "bin_pos", "contact_force"]

def observations(tr):
    return np.concatenate([tr[k] for k in OBS_KEYS], axis=-1).astype(np.float32)


def feature_mask(tr, feat, thr):
    phase = tr["phase"]
    transport = (phase >= 3) & (phase <= 6)
    crossing = (phase >= 4) & (phase <= 6)
    speed = np.linalg.norm(tr["tcp_vel"], axis=-1)
    if feat == "fast":
        return transport & (speed > thr)
    if feat == "too_slow":
        mean = (speed * transport).sum(1) / np.maximum(transport.sum(1), 1)
        return transport & (mean[:, None] < thr)
    if feat == "jerky":
        acc = np.gradient(tr["tcp_vel"], DT, axis=1)
        jerk = np.linalg.norm(np.gradient(acc, DT, axis=1), axis=-1)
        return transport & (jerk > thr)
    if feat in ("low_clearance", "high_clearance"):
        height = tr["block_pos"][..., 2] - BLOCK_HALF
        return crossing & ((height < thr) if feat == "low_clearance" else (height > thr))
    if feat == "cup_close":
        return transport & (tr["cup_distance"] < thr)
    if feat == "cup_far":
        closest = np.where(crossing, tr["cup_distance"], np.inf).min(1)
        return crossing & (closest[:, None] > thr)
    if feat in SHARED_FEATURES:
        return tr[feat] > .5
    raise ValueError(feat)


def add_args(ap):
    ap.add_argument("--data", default="data/franka/train*.npz")
    ap.add_argument("--rerank-data", default="data/franka/rerank*.npz",
                    help="comma-separated shard globs; empty string skips reranking")


def load(split, args, rng):
    patterns = args.data if split == "train" else args.rerank_data
    if not patterns and split == "rerank":
        return None
    files = sorted(set(f for p in patterns.split(",") for f in glob.glob(p)))
    if not files:
        raise FileNotFoundError(f"No Franka shards match {patterns}; run python -m envs.franka.collect")
    parts = []
    for file in files:
        with np.load(file, allow_pickle=False) as data:
            parts.append({k: data[k] for k in data.files})
    if any(set(p) != set(parts[0]) for p in parts):
        raise ValueError("Franka shards have different schemas")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def plot_topdown(ax, tr, i):
    from matplotlib.patches import Circle, Rectangle
    xy = tr["tcp"][i, :, :2]
    ax.plot(*xy.T, "k-", label="gripper")
    ax.plot(*tr["block_pos"][i, :, :2].T, color="tab:orange", label="block")
    bx, by = tr["bin_pos"][i, 0, :2]
    ax.add_patch(Rectangle((bx-.10, by-.10), .20, .20, fill=False, color="tab:blue", label="bin"))
    ax.add_patch(Circle(tr["cup_pos"][i, 0, :2], .04, color="tab:cyan", alpha=.5, label="glass mug"))
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=7)
