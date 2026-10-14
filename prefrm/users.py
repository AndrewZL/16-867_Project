"""Simulated raters with consistent, known preferences over trajectory features.

A user rates 1 iff none of their features is violated anywhere in the trajectory
(noisy-AND), then the label is flipped with probability eps.

Environment-agnostic: the env module supplies FEATURES, THR_RANGES (per-feature threshold
range users sample from), GROUPS [(name, feats, count)], N_RANDOM, and
feature_mask(tr, feat, thr) -> [N, T] bool, and optionally SHARED_FEATURES that every user
dislikes. Features used by the "rare" group are kept out of the random users so that group
stays a genuine long tail.
"""
import numpy as np


def make_population(env, rng, eps=0.05, scale=1):
    users = []
    for name, feats, count in env.GROUPS:
        users += [dict(group=name, feats=list(feats)) for _ in range(count * scale)]
    shared = list(getattr(env, "SHARED_FEATURES", []))
    rare = {f for name, feats, _ in env.GROUPS if name == "rare" for f in feats}
    common = [f for f in env.FEATURES if f not in rare and f not in shared]
    for _ in range(env.N_RANDOM * scale):
        feats = list(rng.choice(common, size=rng.integers(1, 4), replace=False))
        users.append(dict(group="random", feats=feats))
    for u in users:
        u["feats"] += [f for f in shared if f not in u["feats"]]
        u["thr"] = {f: float(rng.uniform(*env.THR_RANGES[f])) for f in u["feats"]}
        u["eps"] = eps
    return users


def user_feature_masks(env, tr, user):
    return {f: env.feature_mask(tr, f, user["thr"][f]) for f in user["feats"]}


def clean_labels(env, tr, users):
    """Returns clean labels [K, N] and union violation masks [K, N, T]."""
    masks = np.stack([np.any(list(user_feature_masks(env, tr, u).values()), axis=0) for u in users])
    return (~masks.any(-1)).astype(np.float32), masks


def collect_ratings(y_clean, users, raters_per_traj, rng):
    """Each trajectory is rated by a random subset of users. Returns (traj, user, y) rows."""
    K, N = y_clean.shape
    rows = []
    for i in range(N):
        for k in rng.choice(K, size=min(raters_per_traj, K), replace=False):
            y = y_clean[k, i]
            if rng.random() < users[k]["eps"]:
                y = 1.0 - y
            rows.append((i, k, y))
    return np.array(rows, dtype=np.float64)
