"""BehaviorAgent wrapper with per-episode driving style and randomly timed injected events.

Style (whole episode) overrides BehaviorAgent's behavior params: target speed relative to
the limit, following time/distance, and whether red lights are ignored. Events are short,
randomly timed perturbations applied on top of the agent's control: steering wobble, a hard
brake pulse, and a lateral drift within the lane. Together they give decorrelated,
controllable violations that the detectors in envs/carla/__init__.py pick up.
"""
import os
import sys

import numpy as np

CARLA_ROOT = os.environ.get("CARLA_ROOT", os.path.expanduser("~/carla/CARLA_0.9.16"))
sys.path.append(os.path.join(CARLA_ROOT, "PythonAPI", "carla"))

from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402
from agents.navigation.behavior_types import Normal  # noqa: E402


def sample_style(rng):
    # speeds in km/h, as BehaviorAgent expects
    return dict(
        max_speed=rng.uniform(30, 70),
        speed_lim_dist=rng.uniform(-12, 10),          # target = limit - this; negative => speeding
        speed_decrease=rng.uniform(4, 12),
        safety_time=rng.uniform(0.3, 3.0),
        braking_distance=rng.uniform(1.0, 6.0),
        min_proximity_threshold=rng.uniform(5, 12),
        ignore_lights=bool(rng.random() < 0.3),
    )


def default_style():
    """BehaviorAgent's stock 'normal' profile: the unperturbed reference driver."""
    return dict(max_speed=50.0, speed_lim_dist=3.0, speed_decrease=10.0, safety_time=3.0,
                braking_distance=5.0, min_proximity_threshold=10.0, ignore_lights=False)


def sample_events(rng, duration):
    """Each event type occurs at most once, at a uniformly random time (s)."""
    ev = {}
    if rng.random() < 0.35:
        ev["wobble"] = dict(t0=rng.uniform(1, duration - 3), dur=rng.uniform(1.5, 3.0),
                            amp=rng.uniform(0.15, 0.3), freq=rng.uniform(0.6, 1.2))
    if rng.random() < 0.3:
        ev["brake"] = dict(t0=rng.uniform(1, duration - 1), dur=rng.uniform(0.5, 0.9),
                           mag=rng.uniform(0.7, 1.0))
    if rng.random() < 0.4:
        ev["drift"] = dict(t0=rng.uniform(1, duration - 4), dur=rng.uniform(2.0, 4.0),
                           off=rng.choice([-1, 1]) * rng.uniform(0.8, 1.5))
    return ev


class PerturbedBehaviorAgent(BehaviorAgent):
    def __init__(self, vehicle, style, events, **kw):
        super().__init__(vehicle, behavior="normal", **kw)
        self._behavior = Normal()
        for k in ("max_speed", "speed_lim_dist", "speed_decrease", "safety_time",
                  "braking_distance", "min_proximity_threshold"):
            setattr(self._behavior, k, style[k])
        self.ignore_traffic_lights(style["ignore_lights"])
        self.events = events
        self.t = 0.0  # seconds since perturbations were enabled; negative = warmup

    @staticmethod
    def _active(e, t):
        return e is not None and e["t0"] <= t < e["t0"] + e["dur"]

    def run_step(self, dt):
        t = self.t
        self.t += dt
        d = self.events.get("drift")
        if d is not None:
            # smooth sin^2 bump in lateral offset
            s = (t - d["t0"]) / d["dur"]
            self.set_offset(d["off"] * np.sin(np.pi * s) ** 2 if 0 <= s < 1 else 0.0)

        control = super().run_step()

        w = self.events.get("wobble")
        if self._active(w, t):
            control.steer = float(np.clip(
                control.steer + w["amp"] * np.sin(2 * np.pi * w["freq"] * (t - w["t0"])), -1, 1))
        b = self.events.get("brake")
        if self._active(b, t):
            control.throttle, control.brake = 0.0, float(b["mag"])
        return control
