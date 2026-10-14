"""Metrics.

  (i)   held-out rating accuracy per user (vs clean labels)
  (ii)  credit assignment: per-step AUROC and IoU of flagged steps vs ground-truth violations
  (iii) reranking: among candidate trajectories that share a scenario id, pick the best under
        each reward and measure per-user satisfaction (mean, worst user, rare user)
"""
import numpy as np
import torch
import torch.nn.functional as F

from prefrm import users as U

FLAG_PENALTY = 0.05  # a step is "flagged" if -log sigmoid(r_t) exceeds this


def auroc(scores, labels):
    labels = labels.astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def step_penalties(model, obs, bs=2048):
    """-log sigmoid(r) per step, [N, T, H]."""
    out = [-F.logsigmoid(model(obs[s:s + bs])) for s in range(0, len(obs), bs)]
    return torch.cat(out).cpu().numpy()


def _head(pen, k):
    """Penalties of user k's head; a single-head model serves every user."""
    return pen[..., 0] if pen.shape[-1] == 1 else pen[..., k]


def rating_accuracy(pen, y_clean):
    return np.array([((np.exp(-_head(pen, k).sum(1)) > 0.5) == (y_clean[k] > 0.5)).mean()
                     for k in range(len(y_clean))])


def credit_assignment(env, pen, tr, pop):
    """Per-feature AUROC/IoU over (traj, user) pairs whose only violated feature is f."""
    res = {f: dict(auc=[], iou=[]) for f in env.FEATURES}
    for k, u in enumerate(pop):
        fm = U.user_feature_masks(env, tr, u)
        viol = np.stack([m.any(1) for m in fm.values()])          # [F_k, N]
        p = _head(pen, k)
        for j, f in enumerate(fm):
            sel = viol[j] & (viol.sum(0) == 1)
            if not sel.any():
                continue
            res[f]["auc"].append(auroc(p[sel].ravel(), fm[f][sel].ravel()))
            flag, truth = p[sel] > FLAG_PENALTY, fm[f][sel]
            res[f]["iou"].extend(((flag & truth).sum(1) / np.maximum((flag | truth).sum(1), 1)).tolist())
    # AUROC is undefined (nan) for global features, where every step is a violation
    return {f: dict(auc=float(np.nanmean(v["auc"])) if not np.all(np.isnan(v["auc"])) else float("nan"),
                    iou=float(np.mean(v["iou"])) if v["iou"] else float("nan"))
            for f, v in res.items()}


def rerank(env, pen_single, pen_per_user, tr, pop, rng):
    """pen_*: step penalties on the candidate set, [N, T, H]."""
    y_clean, _ = U.clean_labels(env, tr, pop)                     # [K, N]
    groups = [g for g in (np.where(tr["scenario"] == s)[0] for s in np.unique(tr["scenario"])) if len(g) > 1]

    traj_lp = -pen_per_user.sum(1)                                # [N, K]
    scores = {
        "single head": -pen_single[..., 0].sum(1),
        "per-user mean": traj_lp.mean(1),
        "per-user min (traj)": traj_lp.min(1),
        "per-user min (step)": -pen_per_user.max(2).sum(1),      # r = min_k r_k per step
    }
    picks = {name: [g[np.argmax(s[g])] for g in groups] for name, s in scores.items()}
    picks["random"] = [rng.choice(g) for g in groups]
    picks["oracle (max #sat)"] = [g[np.argmax(y_clean[:, g].sum(0))] for g in groups]

    rare = [k for k, u in enumerate(pop) if u["group"] == "rare"]
    out = {"_n_scenarios": len(groups),
           "_feasible_all_satisfied": float(np.mean([y_clean[:, g].all(0).any() for g in groups]))}
    for name, idx in picks.items():
        sat = y_clean[:, idx]                                     # [K, S]
        rate = sat.mean(1)
        out[name] = dict(mean=float(rate.mean()), worst_user=float(rate.min()),
                         rare_user=float(rate[rare].mean()) if rare else float("nan"),
                         all_satisfied=float(sat.all(0).mean()), per_user=rate.round(3).tolist())
    return out
