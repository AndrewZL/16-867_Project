"""Label-efficiency sweep: how many ratings per user does each model need?

    uv run python scripts/sweep_labels.py [--env toy] [--budgets 10,25,50,100,200] [--scales 1,5] [--seeds 0,1,2]

Writes out/<env>/sweep_labels/{summary.json, curves.png}; extra args go to prefrm.run.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

MODELS = ("single", "per_user", "shared")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="toy")
    ap.add_argument("--budgets", default="10,25,50,100,200")
    ap.add_argument("--scales", default="1,5")
    ap.add_argument("--seeds", default="0,1,2")
    args, extra = ap.parse_known_args()
    root = os.path.join("out", args.env, "sweep_labels")

    rows = []
    for scale in map(int, args.scales.split(",")):
        for L in map(int, args.budgets.split(",")):
            for seed in map(int, args.seeds.split(",")):
                out = os.path.join(root, f"x{scale}_L{L}_s{seed}")
                res_path = os.path.join(out, "results_eps0.05.json")
                if not os.path.exists(res_path):
                    subprocess.run([sys.executable, "-m", "prefrm.run", "--env", args.env, "--seed", str(seed),
                                    "--labels-per-user", str(L), "--pop-scale", str(scale), "--out", out, *extra],
                                   check=True, stdout=subprocess.DEVNULL)
                r = json.load(open(res_path))
                rare = [k for k, u in enumerate(r["users"]) if u["group"] == "rare"]
                row = dict(scale=scale, L=L, seed=seed, n_users=len(r["users"]))
                for m in MODELS:
                    auc = np.array(r[f"rating_{m}"]["auc"])     # per-user AUROC of P(good): base-rate free
                    row[f"{m}_acc"], row[f"{m}_worst"], row[f"{m}_rare"] = np.nanmean(auc), np.nanmin(auc), np.nanmean(auc[rare])
                    row[f"{m}_credit"] = np.nanmean([v["auc"] for v in r[f"credit_{m}"].values()])
                rr = r.get("rerank", {})
                for sel in ("single head", "per_user min (step)", "shared min (step)", "oracle (max #sat)"):
                    if sel in rr:
                        row[f"rr_{sel}"] = rr[sel]["mean"]
                        row[f"rr_worst_{sel}"] = rr[sel]["worst_user"]
                rows.append(row)
                print(f"x{scale} L={L:4d} seed {seed}: rating AUROC " + "  ".join(
                    f"{m} {row[m + '_acc']:.3f} (worst {row[m + '_worst']:.3f})" for m in MODELS), flush=True)

    # aggregate over seeds
    keys = [k for k in rows[0] if k not in ("scale", "L", "seed", "n_users")]
    summary = []
    for scale in sorted({r["scale"] for r in rows}):
        for L in sorted({r["L"] for r in rows}):
            g = [r for r in rows if r["scale"] == scale and r["L"] == L]
            summary.append(dict(scale=scale, L=L, n_users=g[0]["n_users"],
                                **{k: float(np.mean([r[k] for r in g])) for k in keys},
                                **{k + "_sd": float(np.std([r[k] for r in g])) for k in keys}))
    json.dump(dict(runs=rows, summary=summary), open(os.path.join(root, "summary.json"), "w"), indent=1)

    print(f"\nper-user rating AUROC, mean / worst user / rare user (credit = per-step AUROC)")
    print(f"{'users':>5s} {'L':>4s} | " + " | ".join(f"{m:>26s}" for m in MODELS))
    for s in summary:
        print(f"{s['n_users']:5d} {s['L']:4d} | " + " | ".join(
            f"{s[m + '_acc']:.3f} {s[m + '_worst']:.3f} {s[m + '_rare']:.3f} c{s[m + '_credit']:.2f}" for m in MODELS))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for scale in sorted({s["scale"] for s in summary}):
        g = [s for s in summary if s["scale"] == scale]
        L = [s["L"] for s in g]
        n = g[0]["n_users"]
        for ax, key, title in zip(axes, ["acc", "worst", "rare"], ["mean user", "worst user", "rare user"]):
            for m, ls in (("shared", "-"), ("per_user", "--"), ("single", ":")):
                ax.errorbar(L, [s[f"{m}_{key}"] for s in g], yerr=[s[f"{m}_{key}_sd"] for s in g],
                            ls=ls, marker="o", capsize=3, label=f"{m}, {n} users")
            ax.set_xscale("log"); ax.set_xlabel("training labels per user"); ax.set_title(f"rating AUROC: {title}")
    axes[0].set_ylabel("per-user AUROC of P(good)"); axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "curves.png"), dpi=110)
    print(f"\nwrote {root}/summary.json and curves.png")


if __name__ == "__main__":
    main()
