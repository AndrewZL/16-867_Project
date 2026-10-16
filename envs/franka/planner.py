from dataclasses import dataclass

import numpy as np

DT, T = 0.1, 260
BLOCK_HALF = 0.02
BIN_HALF = 0.09
CUP_RADIUS = 0.04


@dataclass(frozen=True)
class Style:
    speed: float = 0.16       # nominal average Cartesian speed (m/s)
    jerkiness: float = 0.0    # 0: minimum jerk; 1: stop/start progress modulation
    clearance: float = 0.20  # block bottom height above tabletop (m)
    cup_margin: float = 0.12 # horizontal gap from cup shell to carried block (m)

    def __post_init__(self):
        if not (0.07 <= self.speed <= 0.32 and 0 <= self.jerkiness <= 1
                and 0.10 <= self.clearance <= 0.30 and 0 <= self.cup_margin <= 0.20):
            raise ValueError("Style outside supported workspace/timing bounds")


def sample_style(rng):
    return Style(speed=rng.uniform(0.07, 0.32), jerkiness=rng.uniform(0, 1),
                 clearance=rng.uniform(0.10, 0.30), cup_margin=rng.uniform(0, 0.20))


def sample_scenario(rng):
    return dict(block=np.array([rng.uniform(.43, .47), rng.uniform(-.25, -.21), BLOCK_HALF]),
                bin=np.array([rng.uniform(.44, .48), rng.uniform(.23, .27), 0.0]),
                cup=np.array([rng.uniform(.49, .53), rng.uniform(-.01, .03), 0.0]))


class Planner:
    def __init__(self, scenario, style, start):
        self.style = style
        block, goal, cup = (scenario[k] for k in ("block", "bin", "cup"))
        z = style.clearance + BLOCK_HALF
        # Pass on the robot side of the cup; two waypoints maintain the margin.
        x = cup[0] - CUP_RADIUS - BLOCK_HALF - style.cup_margin
        above = block + [0, 0, .14]
        lift = np.array([block[0], block[1], z])
        near = np.array([x, cup[1] - .10, z])
        past = np.array([x, cup[1] + .10, z])
        over = np.array([goal[0], goal[1], z])
        place = np.array([goal[0], goal[1], .055])
        self.points = np.array([start, above, block, block, lift, near, past, over,
                                place, place, over])
        # Fixed careful approach/grasp/place; style acts on lift and transport.
        distances = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        durations = np.array([1.3, 1.2, .8, 0, 0, 0, 0, 1.2, .6, 1.0])
        durations[3:7] = np.maximum(.4, distances[3:7] / style.speed)
        self.ends = np.cumsum(durations)
        self.starts = np.r_[0, self.ends[:-1]]
        self.durations = durations
        if self.ends[-1] + 1.0 > DT * T:
            raise ValueError("Plan exceeds fixed observation horizon")

    def target(self, t):
        i = min(np.searchsorted(self.ends, t, side="right"), len(self.ends) - 1)
        s = np.clip((t - self.starts[i]) / self.durations[i], 0, 1)
        u = 10*s**3 - 15*s**4 + 6*s**5
        if 3 <= i <= 6:
            # Monotone warp: same path/endpoints/duration, variable stop/start speed.
            u -= self.style.jerkiness * np.sin(8 * np.pi * u) / (8 * np.pi)
        target = (1 - u) * self.points[i] + u * self.points[i + 1]
        gripper = 255.0 if i < 2 or i >= 8 else 0.0
        return target, gripper, i
