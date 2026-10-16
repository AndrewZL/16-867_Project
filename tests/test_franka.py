from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

import envs.franka as F
from envs.franka.planner import Planner, Style, sample_scenario
from envs.franka.setup_assets import DEFAULT_MODEL
from prefrm import users as U


class PlannerTests(unittest.TestCase):
    def test_timing_geometry_and_bounds(self):
        sc = sample_scenario(np.random.default_rng(0))
        start = sc["block"] + [0, 0, .2]
        smooth = Planner(sc, Style(), start)
        jerky = Planner(sc, replace(Style(), jerkiness=1), start)
        np.testing.assert_array_equal(smooth.points, jerky.points)
        np.testing.assert_array_equal(smooth.ends, jerky.ends)
        for t in smooth.ends:
            np.testing.assert_allclose(smooth.target(t)[0], jerky.target(t)[0])
        for speed in (.07, .32):
            for clearance in (.1, .3):
                for margin in (0, .2):
                    p = Planner(sc, Style(speed, 1, clearance, margin), start)
                    self.assertLess(p.ends[-1] + 1, F.T*F.DT)
        with self.assertRaises(ValueError):
            Style(speed=0)


@unittest.skipUnless(Path(DEFAULT_MODEL).exists(), "run python -m envs.franka.setup_assets first")
class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from envs.franka.env import FrankaEnv
        cls.env = FrankaEnv()
        cls.sc = sample_scenario(np.random.default_rng(0))
        cls.styles = dict(reference=Style(), slow=replace(Style(), speed=.075),
                          jerky=replace(Style(), jerkiness=1),
                          collision=replace(Style(), clearance=.1, cup_margin=0),
                          drop=replace(Style(), speed=.32, jerkiness=1))
        cls.tr = {k: cls.env.rollout(cls.sc, style) for k, style in cls.styles.items()}
        cls.batch = {k: np.stack([t[k] for t in cls.tr.values()]) for k in cls.tr["reference"]}

    def test_reference_places_real_block(self):
        tr = self.tr["reference"]
        self.assertTrue(tr["success"])
        self.assertGreater(tr["block_pos"][:, 2].max(), .15)
        self.assertLess(np.linalg.norm(tr["block_pos"][-1, :2] - self.sc["bin"][:2]), .07)
        self.assertFalse(tr["collision"].any() or tr["drop"].any())
        self.assertTrue(self.tr["jerky"]["success"], "jerk is a preference, not automatic failure")

    def test_style_affects_measured_motion(self):
        def mean_speed(tr):
            mask = (tr["phase"] >= 3) & (tr["phase"] <= 6)
            return np.linalg.norm(tr["tcp_vel"][mask], axis=-1).mean()
        self.assertLess(mean_speed(self.tr["slow"]), .6*mean_speed(self.tr["reference"]))
        def max_jerk(tr):
            mask = (tr["phase"] >= 3) & (tr["phase"] <= 6)
            j = np.gradient(np.gradient(tr["tcp_vel"], F.DT, axis=0), F.DT, axis=0)
            return np.linalg.norm(j[mask], axis=-1).max()
        self.assertGreater(max_jerk(self.tr["jerky"]), 1.2*max_jerk(self.tr["reference"]))

    def test_failures_are_shared_and_observations_do_not_leak(self):
        self.assertTrue(self.tr["collision"]["collision"].any())
        self.assertTrue(self.tr["drop"]["drop"].any())
        pop = U.make_population(F, np.random.default_rng(1), eps=0)
        labels, _ = U.clean_labels(F, self.batch, pop)
        self.assertTrue((labels[:, -2:] == 0).all())
        obs = F.observations(self.batch)
        self.assertEqual(obs.shape, (5, F.T, 65))
        self.assertTrue(np.isfinite(obs).all())
        altered = dict(self.batch)
        for key in ("phase", "collision", "drop", "incomplete", "success", "cup_distance"):
            altered[key] = np.zeros_like(altered[key])
        altered["style_speed"] = np.ones(5)*999
        np.testing.assert_array_equal(F.observations(altered), obs)
        for feat in F.FEATURES:
            mask = F.feature_mask(self.batch, feat, np.mean(F.THR_RANGES[feat]))
            self.assertEqual(mask.shape, (5, F.T))
            self.assertEqual(mask.dtype, np.bool_)

    def test_reset_is_deterministic(self):
        again = self.env.rollout(self.sc, Style())
        np.testing.assert_array_equal(again["qpos"], self.tr["reference"]["qpos"])
        np.testing.assert_array_equal(again["block_pos"], self.tr["reference"]["block_pos"])


if __name__ == "__main__":
    unittest.main()
