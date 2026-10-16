import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .env import FrankaEnv
from .planner import Style, sample_scenario, sample_style
from .setup_assets import DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--cands", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="use distinct seeds for train/rerank and each shard")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="data/franka/train.npz")
    ap.add_argument("--reference-only", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="show each rollout in MuJoCo's passive viewer")
    args = ap.parse_args()
    if args.scenarios < 1 or args.cands < 1 or args.seed < 0:
        ap.error("scenarios/cands must be positive and seed nonnegative")
    env = FrankaEnv(args.model)
    revision_file = Path(args.model).parent / "REVISION"
    revision = revision_file.read_text().strip() if revision_file.exists() else "external"
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = []
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    try:
        for si in range(args.scenarios):
            scenario = sample_scenario(np.random.default_rng([args.seed, si]))
            for cand in range(args.cands):
                style = Style() if cand == 0 or args.reference_only else sample_style(
                    np.random.default_rng([args.seed, si, cand]))
                tr = env.rollout(scenario, style, callback=(lambda e: viewer.sync()) if viewer else None)
                tr.update(scenario=np.array(f"{args.seed}:{si}"), cand=np.array(cand),
                          menagerie_revision=np.array(revision))
                tr.update({f"style_{k}": np.array(v) for k, v in asdict(style).items()})
                episodes.append(tr)
                print(f"scenario {si} candidate {cand}: success={bool(tr['success'])} "
                      f"collision={bool(tr['collision'].any())} drop={bool(tr['drop'].any())}", flush=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **{k: np.stack([e[k] for e in episodes]) for k in episodes[0]})
            tmp.replace(path)
    finally:
        if viewer:
            viewer.close()
    print(f"Wrote {len(episodes)} episodes to {path}")


if __name__ == "__main__":
    main()
