"""2D toy driving domain.

Straight two-lane road. Ego starts in the right lane (y=0) behind a slow lead car and
must change into the passing lane (y=3.5). A pedestrian stands on the left shoulder and
a school zone (limit 8 m/s) sits somewhere along the road.

Trajectories come from a parameterized scripted planner with randomly injected, randomly
timed attribute violations, so per-step ground-truth masks exist for every feature.
Scenario (world layout) and planner params are sampled separately so the same scenario
can be re-rolled with many candidate params (used for reward-based reranking).
"""
import numpy as np

DT, T = 0.1, 120
LANE_W = 3.5
LEFT_Y = 3.5                        # center of the passing lane
PED_Y = LEFT_Y + LANE_W / 2 + 0.5   # pedestrian on the left shoulder
ZONE_LEN, ZONE_LIMIT = 30.0, 8.0

FEATURES = ["wobble", "hard_brake", "tailgate", "ped_close", "zone_speed", "fast", "too_slow"]


def sample_scenarios(n, rng):
    return dict(
        lead_x0=rng.uniform(25, 45, n),
        lead_v=rng.uniform(5, 8, n),
        ped_x=rng.uniform(40, 140, n),
        zone_x=rng.uniform(50, 130, n),
    )


def sample_params(n, rng):
    inject = lambda p: rng.random(n) < p
    return dict(
        cruise=rng.uniform(9, 17, n),
        h_trigger=rng.uniform(1.2, 3.5, n),        # headway (s) at which the lane change starts
        lc_dur=rng.uniform(3.5, 5.0, n),
        wobble_amp=np.where(inject(0.3), rng.uniform(0.35, 0.6, n), 0.0),
        wobble_freq=rng.uniform(0.5, 0.8, n),
        brake_t=np.where(inject(0.25), rng.uniform(1.0, T * DT - 1.0, n), np.inf),
        brake_mag=rng.uniform(5, 7, n),
        ped_drift=np.where(inject(0.25), rng.uniform(0.8, 1.6, n), 0.0),
        zone_comply=~inject(0.3),
        zone_v=rng.uniform(6, 8.5, n),
        zone_anticip=rng.uniform(5, 60, n),        # distance (m) over which ego slows for the zone
    )


def rollout(sc, pr):
    n = len(sc["lead_x0"])
    x, v = np.zeros(n), pr["cruise"].copy()
    lead_x = sc["lead_x0"].copy()
    lc_start = np.full(n, np.inf)
    X, V, A, LX = (np.zeros((n, T)) for _ in range(4))
    zone_target = np.where(pr["zone_comply"], pr["zone_v"], pr["cruise"])

    for t in range(T):
        tt = t * DT
        gap = lead_x - x
        trig = np.isinf(lc_start) & (gap > 0) & (gap < pr["h_trigger"] * np.maximum(v, 0.1))
        lc_start[trig] = tt

        # target speed ramps down (linear in v^2 over distance) ahead of and through the zone
        d_zone = sc["zone_x"] - x
        frac = np.clip(d_zone / pr["zone_anticip"], 0, 1)
        v_ramp = np.sqrt(zone_target**2 + frac * np.maximum(pr["cruise"]**2 - zone_target**2, 0))
        v_des = np.where(d_zone > -ZONE_LEN, v_ramp, pr["cruise"])

        a = np.clip(2.0 * (v_des - v), -8.0, 3.0)
        a = np.where((tt >= pr["brake_t"]) & (tt < pr["brake_t"] + 0.6), -pr["brake_mag"], a)

        X[:, t], V[:, t], A[:, t], LX[:, t] = x, v, a, lead_x
        v = np.maximum(v + a * DT, 0.0)
        x = x + v * DT
        lead_x = lead_x + sc["lead_v"] * DT

    tg = np.arange(T)[None] * DT
    s = np.clip((tg - lc_start[:, None]) / pr["lc_dur"][:, None], 0, 1)
    y = LEFT_Y * (10 * s**3 - 15 * s**4 + 6 * s**5)

    # wobble: decaying oscillation after the lane change settles
    te = lc_start + pr["lc_dur"]
    dt_e = np.where(np.isfinite(te)[:, None], tg - np.where(np.isfinite(te), te, 0)[:, None], -1.0)
    y = y + np.where(dt_e > 0, pr["wobble_amp"][:, None] * np.exp(-dt_e / 1.5)
                     * np.sin(2 * np.pi * pr["wobble_freq"][:, None] * dt_e), 0.0)

    # drift toward the pedestrian while passing them
    passed = (X >= sc["ped_x"][:, None]).any(1)
    t_ped = np.where(passed, np.argmax(X >= sc["ped_x"][:, None], 1) * DT, np.inf)
    y = y + pr["ped_drift"][:, None] * np.exp(-0.5 * ((tg - t_ped[:, None]) / 1.2) ** 2)

    vy = np.gradient(y, DT, axis=1)
    ay = np.gradient(vy, DT, axis=1)
    return dict(x=X, v=V, a=A, y=y, vy=vy, ay=ay, lead_x=LX,
                lead_v=sc["lead_v"], ped_x=sc["ped_x"], zone_x=sc["zone_x"])


def observations(tr):
    """Per-step observation [N, T, D]. Deliberately raw: thresholds must be learned."""
    x, v, y = tr["x"], tr["v"], tr["y"]
    gap = tr["lead_x"] - x
    same_lane = (y < LANE_W / 2) & (gap > 0)
    lead_gap = np.where(same_lane, np.minimum(gap, 50.0), 50.0)
    closing = np.where(same_lane, v - tr["lead_v"][:, None], 0.0)
    ped_dx = np.clip(tr["ped_x"][:, None] - x, -30, 30)
    zone_dx = np.clip(tr["zone_x"][:, None] - x, -40, 60)
    in_zone = ((zone_dx <= 0) & (zone_dx > -ZONE_LEN)).astype(float)
    return np.stack([v, tr["a"], y, tr["vy"], tr["ay"], lead_gap, closing,
                     ped_dx, PED_Y - y, zone_dx, in_zone], -1).astype(np.float32)


def feature_mask(tr, feat, thr):
    """Per-step bool mask [N, T] of where `feat` is violated under threshold `thr`."""
    x, v, y = tr["x"], tr["v"], tr["y"]
    if feat == "wobble":
        return np.abs(tr["ay"]) > thr
    if feat == "hard_brake":
        return tr["a"] < -thr
    if feat == "tailgate":
        gap = tr["lead_x"] - x
        return (y < LANE_W / 2) & (gap > 0) & (gap / np.maximum(v, 0.1) < thr)
    if feat == "ped_close":
        return (np.abs(x - tr["ped_x"][:, None]) < 5) & (PED_Y - y < thr)
    if feat == "zone_speed":
        zx = tr["zone_x"][:, None]
        return (x >= zx) & (x < zx + ZONE_LEN) & (v > ZONE_LIMIT + thr)
    if feat == "fast":
        return v > thr
    if feat == "too_slow":
        return np.broadcast_to((v.mean(1) < thr)[:, None], v.shape).copy()
    raise ValueError(feat)


def repeat_scenarios(sc, n_cand):
    return {k: np.repeat(v, n_cand) for k, v in sc.items()}
