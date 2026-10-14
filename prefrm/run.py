"""Train per-user reward heads vs a single pooled head from sparse binary ratings, then evaluate.

Usage: python -m prefrm.run --env toy [--eps 0.05] [env-specific args]
       python -m prefrm.run --env carla --data data/carla_train.npz --rerank-data data/carla_rerank.npz

An env is a package under envs/ exposing:
  FEATURES, THR_RANGES, GROUPS, N_RANDOM     simulated-user spec (see prefrm/users.py)
  add_args(parser)                           env-specific CLI args
  load(split, args, rng) -> tr               split in {"train", "rerank"}; tr is a dict of arrays
                                             with leading dim N, including "scenario" [N]
  observations(tr) -> [N, T, D] float32
  feature_mask(tr, feat, thr) -> [N, T] bool
  plot_topdown(ax, tr, i)                    optional
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch

from prefrm import eval as E
from prefrm import users as U
from prefrm.model import RewardModel, train


def take(tr, idx):
    return {k: v[idx] for k, v in tr.items()}


def plot_examples(path, env, tr, pen_pu, pen_si, pop, y_clean, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first = {}
    for k, u in enumerate(pop):
        first.setdefault(u["group"], k)
    show = [first[g] for g in ("rare", "smooth", "eager") if g in first]
    topdown = hasattr(env, "plot_topdown")
    fig, axes = plt.subplots(len(idx), 1 + topdown, figsize=(6.5 * (1 + topdown), 3.2 * len(idx)), squeeze=False)
    for row, i in enumerate(idx):
        if topdown:
            env.plot_topdown(axes[row, 0], tr, i)
        ax = axes[row, -1]
        t = np.arange(pen_pu.shape[1])
        one = take(tr, slice(i, i + 1))
        for k in show:
            m = np.any(list(U.user_feature_masks(env, one, pop[k]).values()), axis=0)[0]
            l, = ax.plot(t, pen_pu[i, :, k],
                         label=f"user {k} ({pop[k]['group']}: {','.join(pop[k]['feats'])}) y={int(y_clean[k, i])}")
            ax.fill_between(t, 0, m * pen_pu[i].max() * 1.05, color=l.get_color(), alpha=.12, step="mid")
        ax.plot(t, pen_si[i], "k:", label="single head")
        ax.set_xlabel("step"); ax.set_ylabel("-log σ(r_t)")
        ax.legend(fontsize=7)
    fig.suptitle("Per-step penalty by head (shaded = that user's ground-truth violation steps)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="toy")
    ap.add_argument("--raters", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--labels-per-user", type=int, default=0, help="cap on training labels per user (0 = no cap)")
    ap.add_argument("--pop-scale", type=int, default=1, help="multiply the number of users in every group")
    ap.add_argument("--prior", type=float, default=0.1,
                    help="penalty on per-user corrections around the population head (shared model)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: out/<env>")
    known, _ = ap.parse_known_args()
    env = importlib.import_module(f"envs.{known.env}")
    env.add_args(ap)
    args = ap.parse_args()
    out = args.out or os.path.join("out", args.env)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = env.load("train", args, rng)
    pop = U.make_population(env, rng, eps=args.eps, scale=args.pop_scale)
    y_clean, _ = U.clean_labels(env, tr, pop)
    rows = U.collect_ratings(y_clean, pop, args.raters, rng)
    N = len(tr["scenario"])

    print(f"[{args.env}] {len(pop)} users, {N} trajectories, {len(rows)} ratings, eps={args.eps}")
    for k, u in enumerate(pop):
        print(f"  user {k:2d} {u['group']:7s} {','.join(u['feats']):40s} approval {y_clean[k].mean():.2f}")
    per_traj = {}
    for i, k, y in rows:
        per_traj.setdefault(int(i), []).append(y)
    print(f"  trajectories with conflicting ratings: {np.mean([0 < np.mean(v) < 1 for v in per_traj.values()]):.2f}")

    # split by scenario so held-out trajectories come from unseen scenarios
    scen = np.unique(tr["scenario"])
    test_scen = rng.choice(scen, size=max(1, len(scen) // 5), replace=False)
    is_test = np.isin(tr["scenario"], test_scen)
    obs_np = env.observations(tr)
    flat = obs_np[~is_test].reshape(-1, obs_np.shape[-1])
    norm = (flat.mean(0), flat.std(0) + 1e-6)
    to_t = lambda o: torch.as_tensor((o - norm[0]) / norm[1], dtype=torch.float32, device=device)
    obs = to_t(obs_np)
    rows_tr = rows[~is_test[rows[:, 0].astype(int)]]
    if args.labels_per_user:
        rows_tr = np.concatenate([rng.permutation(rows_tr[rows_tr[:, 1] == k])[:args.labels_per_user]
                                  for k in range(len(pop))])
    print(f"  training labels: {len(rows_tr)} ({len(rows_tr) / len(pop):.0f} per user)")

    # single: one pooled head; per_user: independent heads; shared: population head + penalized
    # per-user corrections. All share the encoder architecture.
    models = {}
    for name, H, shared in [("single", 1, False), ("per_user", len(pop), False), ("shared", len(pop), True)]:
        print(f"training {name}")
        model = RewardModel(obs.shape[-1], H, horizon=obs.shape[1], shared=shared).to(device)
        models[name] = train(model, obs, rows_tr, epochs=args.epochs, single_head=(H == 1),
                             prior=args.prior if shared else 0.0)

    te = np.where(is_test)[0]
    tr_te = take(tr, te)
    pen = {n: E.step_penalties(m, obs[te]) for n, m in models.items()}
    results = dict(args=vars(args), users=[dict(group=u["group"], feats=u["feats"], thr=u["thr"]) for u in pop])
    rare = [k for k, u in enumerate(pop) if u["group"] == "rare"]

    print("\n(i) held-out rating accuracy vs clean labels")
    for name in models:
        m = E.rating_metrics(pen[name], y_clean[:, te])
        results[f"acc_{name}"] = m["acc"].tolist()
        results[f"rating_{name}"] = {k: v.tolist() for k, v in m.items()}
        print(f"  {name:9s} acc {m['acc'].mean():.3f} (majority {m['majority'].mean():.3f})  "
              f"balanced {m['bal_acc'].mean():.3f}  AUROC mean {np.nanmean(m['auc']):.3f} "
              f"worst {np.nanmin(m['auc']):.3f} rare {np.nanmean(m['auc'][rare]):.3f}")

    print("\n(ii) credit assignment (pairs where exactly one of the user's features is violated)")
    print(f"  {'feature':12s} " + "  ".join(f"{n + ' AUC':>14s} {n + ' IoU':>14s}" for n in models))
    for name in models:
        results[f"credit_{name}"] = E.credit_assignment(env, pen[name], tr_te, pop)
    for f in env.FEATURES:
        print(f"  {f:12s} " + "  ".join(
            f"{results[f'credit_{n}'][f]['auc']:14.3f} {results[f'credit_{n}'][f]['iou']:14.3f}" for n in models))

    tr_rr = env.load("rerank", args, rng)
    if tr_rr is not None:
        obs_rr = to_t(env.observations(tr_rr))
        rr = E.rerank(env, E.step_penalties(models["single"], obs_rr),
                      {n: E.step_penalties(models[n], obs_rr) for n in ("per_user", "shared")}, tr_rr, pop, rng)
        results["rerank"] = rr
        print(f"\n(iii) reranking on {rr.pop('_n_scenarios')} unseen scenarios (per-user satisfaction)")
        print(f"  scenarios where some candidate satisfies everyone: {rr.pop('_feasible_all_satisfied'):.2f}")
        print(f"  {'selector':24s} {'mean':>6s} {'worst':>6s} {'rare':>6s} {'all-sat':>8s}")
        for name, m in rr.items():
            print(f"  {name:24s} {m['mean']:6.3f} {m['worst_user']:6.3f} {m['rare_user']:6.3f} {m['all_satisfied']:8.3f}")

    ex = np.where(y_clean[rare[0], te] == 0)[0][:3] if rare else np.arange(3)
    tag = f"eps{args.eps}"
    plot_examples(os.path.join(out, f"example_{tag}.png"), env, tr_te,
                  pen["shared"], pen["single"][..., 0], pop, y_clean[:, te], ex)
    with open(os.path.join(out, f"results_{tag}.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {out}/results_{tag}.json and example_{tag}.png")


if __name__ == "__main__":
    main()
