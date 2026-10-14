"""Collect perturbed-BehaviorAgent rollouts from a running CARLA server into .npz shards.

A scenario = (town, ego spawn, destination, background traffic, pedestrians), fixed by a
seed. Each scenario is rolled out --cands times with independently sampled style/events,
so shards with --cands > 1 double as candidate sets for reranking.

    scripts/carla_server.sh 2000 0 &
    uv run python -m envs.carla.collect --town Town03 --out data/carla/train_Town03.npz --scenarios 200

One town per process: switching maps can segfault the server, which aborts the client from
C++. scripts/collect.sh runs one fresh server per town.
"""
import argparse
import os
import time

import carla
import numpy as np

from envs.carla import DT, STEP_KEYS
from envs.carla.agent import PerturbedBehaviorAgent, default_style, sample_events, sample_style

from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: E402  (path set by agent.py)

PHYS_DT = 0.05
SUBSTEPS = int(round(DT / PHYS_DT))
EGO_BP = "vehicle.lincoln.mkz_2020"


class StateLogger:
    """Per-step privileged state of the ego relative to lane, traffic, pedestrians, lights."""

    def __init__(self, world, ego, cmap, others, walkers):
        self.world, self.ego, self.map = world, ego, cmap
        self.others, self.walkers = others, walkers
        self.ext = {a.id: a.bounding_box.extent.x for a in others}
        self.ego_ext = ego.bounding_box.extent.x
        self.prev_in_junction, self.recent_red, self.running_red = False, 0, False
        self.collided = False

    def on_collision(self, _event):
        self.collided = True

    def __call__(self):
        snap = self.world.get_snapshot()
        ego = self.ego
        tf = ego.get_transform()
        loc, fwd, right = tf.location, tf.get_forward_vector(), tf.get_right_vector()
        vel = ego.get_velocity()
        ctrl = ego.get_control()
        v = vel.length()

        wp = self.map.get_waypoint(loc)
        wtf = wp.transform
        lane_offset = (loc - wtf.location).dot(wtf.get_right_vector())
        in_junction = wp.is_junction

        f2, r2 = np.array([fwd.x, fwd.y]), np.array([right.x, right.y])
        p0 = np.array([loc.x, loc.y])

        lead_gap, lead_closing = 100.0, 0.0
        for a in self.others:
            s = snap.find(a.id)
            if s is None:
                continue
            al = s.get_transform().location
            rel = np.array([al.x, al.y]) - p0
            dx, dy = rel @ f2, rel @ r2
            if 0 < dx < 60 and abs(dy) < 1.8:
                gap = dx - self.ego_ext - self.ext[a.id]
                if gap < lead_gap:
                    av = s.get_velocity()
                    lead_gap, lead_closing = gap, v - (av.x * fwd.x + av.y * fwd.y)

        ped_dist, ped_dx, ped_dy = 100.0, 100.0, 100.0
        for a in self.walkers:
            s = snap.find(a.id)
            if s is None:
                continue
            al = s.get_transform().location
            rel = np.array([al.x, al.y]) - p0
            d = float(np.hypot(*rel))
            if d < ped_dist:
                ped_dist, ped_dx, ped_dy = d, rel @ f2, rel @ r2

        at_red = ego.is_at_traffic_light() and ego.get_traffic_light_state() == carla.TrafficLightState.Red
        # running a red = entering a junction within 1 s of being at a red light
        self.recent_red = 10 if at_red else max(self.recent_red - 1, 0)
        if in_junction and not self.prev_in_junction:
            self.running_red = self.recent_red > 0 and v > 0.5
        if not in_junction:
            self.running_red = False
        self.prev_in_junction = in_junction

        out = dict(x=loc.x, y=loc.y, yaw=tf.rotation.yaw, v=v, a_lon=0.0, a_lat=0.0,
                   yaw_rate=ego.get_angular_velocity().z, steer=ctrl.steer, throttle=ctrl.throttle,
                   brake=ctrl.brake, speed_limit=ego.get_speed_limit() / 3.6, lane_offset=lane_offset,
                   in_junction=float(in_junction), lane_id=wp.lane_id, road_id=wp.road_id, lead_gap=lead_gap, lead_closing=lead_closing,
                   ped_dist=ped_dist, ped_dx=ped_dx, ped_dy=ped_dy, at_red=float(at_red),
                   red_run=float(in_junction and self.running_red), collision=float(self.collided))
        self.collided = False
        return out


def finalize(log):
    """Kinematic accelerations from logged speed/yaw rate (CARLA's get_acceleration is spiky)."""
    log["a_lon"][:] = np.gradient(log["v"], DT)
    log["a_lat"][:] = log["v"] * np.deg2rad(log["yaw_rate"])


def spawn_walkers(client, world, srng, n, near, radius=60.0, max_tries=3000):
    """Spawn walkers at sidewalk locations within `radius` of the points in `near`."""
    bps = world.get_blueprint_library().filter("walker.pedestrian.*")
    locs = []
    for _ in range(max_tries):
        if len(locs) >= n:
            break
        l = world.get_random_location_from_navigation()
        if l is not None and np.min(np.hypot(near[:, 0] - l.x, near[:, 1] - l.y)) < radius:
            locs.append(l)
    batch = []
    for l in locs:
        bp = bps[int(srng.integers(len(bps)))]
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        batch.append(carla.command.SpawnActor(bp, carla.Transform(l + carla.Location(z=1.0))))
    walkers = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
    ctrl_bp = world.get_blueprint_library().find("controller.ai.walker")
    ctrls = [r.actor_id for r in client.apply_batch_sync(
        [carla.command.SpawnActor(ctrl_bp, carla.Transform(), w) for w in walkers], True) if not r.error]
    world.tick()
    for c in world.get_actors(ctrls):
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(float(1 + srng.random()))
    return walkers, ctrls


def run_episode(client, world, tm, grp, scen, style, events, args):
    cmap = world.get_map()
    spawns = cmap.get_spawn_points()
    srng = np.random.default_rng(scen["seed"])  # same draws for every candidate of a scenario
    world.set_pedestrians_seed(int(scen["seed"]))
    tm.set_random_device_seed(int(scen["seed"]))
    actors, ctrls = [], []
    try:
        ego = world.spawn_actor(world.get_blueprint_library().find(EGO_BP), spawns[scen["spawn"]])
        actors.append(ego.id)

        # background traffic at other spawn points
        vbps = [b for b in world.get_blueprint_library().filter("vehicle.*")
                if int(b.get_attribute("number_of_wheels")) == 4]
        others_idx = srng.permutation([i for i in range(len(spawns)) if i != scen["spawn"]])[:args.vehicles]
        batch = []
        for i in others_idx:
            bp = vbps[int(srng.integers(len(vbps)))]
            batch.append(carla.command.SpawnActor(bp, spawns[i]).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
        vids = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
        actors += vids

        agent = PerturbedBehaviorAgent(ego, style, events, map_inst=cmap, grp_inst=grp)
        agent.set_destination(spawns[scen["dest"]].location)
        route = np.array([[w.transform.location.x, w.transform.location.y]
                          for w, _ in agent.get_local_planner()._waypoints_queue])
        walkers, ctrls = spawn_walkers(client, world, srng, args.walkers, route)
        actors += walkers

        col = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"),
                                carla.Transform(), attach_to=ego)
        actors.append(col.id)
        logger = StateLogger(world, ego, cmap, list(world.get_actors(vids)), list(world.get_actors(walkers)))
        col.listen(logger.on_collision)

        T = int(round(args.seconds / DT))
        warm = int(round(args.warmup / DT))
        agent.t = -args.warmup
        log = {k: np.zeros(T, np.float32) for k in STEP_KEYS}
        for step in range(warm + T):
            # extend the route before it runs out: BehaviorAgent crashes when its
            # look-ahead waypoint falls past the end of the plan
            if len(agent.get_local_planner()._waypoints_queue) < 50:
                agent.set_destination(spawns[int(srng.integers(len(spawns)))].location, clean_queue=False)
            for _ in range(SUBSTEPS):
                ego.apply_control(agent.run_step(PHYS_DT))
                world.tick()
            if step >= warm:
                for k, val in logger().items():
                    log[k][step - warm] = val
        col.stop()
        finalize(log)
        return log
    finally:
        for c in world.get_actors(ctrls):
            c.stop()
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in ctrls + actors], True)
        world.tick()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--town", default="Town10HD_Opt")
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--cands", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--vehicles", type=int, default=40)
    ap.add_argument("--walkers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0, help="shard seed; scenario ids are seed*1e5 + index")
    ap.add_argument("--style", choices=["sampled", "default"], default="sampled")
    ap.add_argument("--event-scale", type=float, default=1.0, help="multiplier on event probabilities (0 = none)")
    ap.add_argument("--clean", action="store_true", help="shorthand for --style default --event-scale 0")
    ap.add_argument("--ref-cand", action="store_true",
                    help="candidate 0 of every scenario is the clean reference driver")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.clean:
        args.style, args.event_scale = "default", 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    client = connect(args.host, args.port)
    world = client.get_world()
    if not world.get_map().name.endswith(args.town):
        world = client.load_world(args.town)
    world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=PHYS_DT,
                                             no_rendering_mode=True))
    tm = client.get_trafficmanager(args.tm_port)
    tm.set_synchronous_mode(True)
    world.set_pedestrians_cross_factor(0.2)
    grp = GlobalRoutePlanner(world.get_map(), 2.0)
    spawns = world.get_map().get_spawn_points()

    data = {k: [] for k in STEP_KEYS}
    meta = {k: [] for k in ["scenario", "town", "cand"]}
    style_keys = list(sample_style(np.random.default_rng(0)))
    meta.update({f"style_{k}": [] for k in style_keys})
    meta.update({f"event_{k}": [] for k in ["wobble", "brake", "drift"]})  # event start time (s), -1 if none
    try:
        for si in range(args.scenarios):
            sid = args.seed * 100000 + si
            srng = np.random.default_rng(sid)
            scen = dict(seed=sid, spawn=int(srng.integers(len(spawns))), dest=int(srng.integers(len(spawns))))
            for c in range(args.cands):
                prng = np.random.default_rng([sid, c])
                ref = args.ref_cand and c == 0
                style = default_style() if ref or args.style == "default" else sample_style(prng)
                events = {} if ref else sample_events(prng, args.seconds, args.event_scale)
                t0 = time.time()
                try:
                    log = run_episode(client, world, tm, grp, scen, style, events, args)
                except RuntimeError as e:
                    print(f"  scenario {sid} cand {c} failed: {e}")
                    continue
                for k in STEP_KEYS:
                    data[k].append(log[k])
                meta["scenario"].append(sid); meta["town"].append(args.town); meta["cand"].append(c)
                for k in style_keys:
                    meta[f"style_{k}"].append(float(style[k]))
                for k in ["wobble", "brake", "drift"]:
                    meta[f"event_{k}"].append(events[k]["t0"] if k in events else -1.0)
                print(f"  {args.town} scen {sid} cand {c}: {time.time() - t0:.1f}s  mean v {log['v'].mean():.1f} "
                      f"collisions {int(log['collision'].sum())}", flush=True)
            # rewrite the shard after every scenario so partial runs are usable
            np.savez_compressed(args.out, **{k: np.stack(v) for k, v in data.items()},
                                **{k: np.array(v) for k, v in meta.items()})
    finally:
        tm.set_synchronous_mode(False)
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
    print(f"wrote {args.out}: {len(meta['scenario'])} episodes")


def connect(host, port, wait=120.0):
    """Connect, retrying while the server boots."""
    t0 = time.time()
    while True:
        try:
            client = carla.Client(host, port)
            client.set_timeout(10.0)
            client.get_server_version()
            client.set_timeout(120.0)
            return client
        except RuntimeError:
            if time.time() - t0 > wait:
                raise
            time.sleep(2.0)


if __name__ == "__main__":
    main()
