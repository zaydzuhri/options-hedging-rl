"""
Why not just use RL? Pathwise vs score-function gradients on the same hedging problem.

    uv run gradient_estimators.py --config configs/gbm_deep.json

Both estimate the gradient of J(θ) = E[exp(-a * P&L)] for a Gaussian policy u_t = π_θ(x_t) + s·ξ_t
(clipped to the position limits), on the same episodes and noise:

  pathwise (what deep_hedge.py uses):  ∇θ exp(-a P&L), backpropagated through every trade and state
  score function (REINFORCE / PPO):    exp(-a P&L) · Σ_t ∇θ log N(u_t; π_θ(x_t), s²), with a
                                        leave-one-out baseline (still unbiased)

The score function never differentiates the P&L; it only learns which random actions happened to
come with good outcomes, so its variance grows with the horizon and with 1/s. We measure
(1) the relative error of each estimator against a large-batch reference, as a function of
batch size, and (2) learning curves with the same optimizer, batch and number of episodes.
Outputs go to results/<name>/gradient_estimators.{csv,png}.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backtest import load_market_data, make_episodes, split_train_test
from plots import SLOTS as PALETTE   # the repo's categorical colors
from deep_hedge import ARCHS, DEFAULTS, DTYPE, FEATURES, _market, episode_pnl, episode_tensors, risk


def gradient(policy, ep, m, a, noise, estimator, gen) -> torch.Tensor:
    """One estimate of ∇θ E[exp(-a P&L)] from the episodes in `ep`."""
    policy.zero_grad()
    dt = 1.0 / m["steps_per_year"]
    if estimator == "pathwise":
        act = lambda f, d, x: policy(f, d, x) + noise * torch.randn(len(x), generator=gen, dtype=DTYPE)
        torch.exp(-a * episode_pnl(act, ep, m["cost_rate"], m["rate"], dt, m["limits"])).mean().backward()
    else:
        logp = []

        def act(f, d, x):
            mean = policy(f, d, x.detach())
            u = (mean + noise * torch.randn(len(x), generator=gen, dtype=DTYPE)).detach()
            logp.append(-0.5 * ((u - mean) / noise) ** 2)
            return u

        pnl = episode_pnl(act, ep, m["cost_rate"], m["rate"], dt, m["limits"])
        utility = torch.exp(-a * pnl.detach())
        baseline = (utility.sum() - utility) / (len(utility) - 1)        # leave-one-out
        ((utility - baseline) * torch.stack(logp).sum(0)).mean().backward()
    return torch.cat([p.grad.flatten() for p in policy.parameters()]).clone()


def new_policy(fit, m, p, seed=0):
    torch.manual_seed(seed)
    flat = fit["features"].reshape(-1, len(FEATURES))
    return ARCHS[p["arch"]](flat.mean(0), flat.std(0), m["limits"], p["hidden"], p["layers"]).to(DTYPE)


def subset(ep, idx):
    return {k: v[idx] for k, v in ep.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", default=None, help="which deep_hedge entry to take the policy from (default: first)")
    parser.add_argument("--noise", type=float, nargs="+", default=[0.01, 0.05], help="exploration std, in shares")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--steps", type=int, default=1500, help="optimizer steps for the learning curves")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    name = cfg.get("name", Path(args.config).stem)
    entry = next(s for s in cfg["strategies"] if s["name"] == "deep_hedge" and args.label in (None, s.get("label")))
    p = {**DEFAULTS, **{k: v for k, v in entry["params"].items() if k != "model"}}
    a = p["objective"].get("risk_aversion", 1.0)
    m = _market(cfg)
    train_df, _ = split_train_test(load_market_data(cfg), cfg["split"])
    cut = int(len(train_df) * (1 - p["val_frac"]))
    tensors = lambda df: episode_tensors(make_episodes(df, m["n_steps"], 1), m["kind"], m["moneyness"], m["rate"],
                                         m["steps_per_year"])
    fit, val = tensors(train_df.iloc[:cut]), tensors(train_df.iloc[cut:])
    n_fit = len(fit["premium"])
    out = Path(cfg.get("output_dir", "results")) / name
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(1)
    rows, curves = [], []

    # (1) estimator error vs batch size, at the policy training starts from
    policy = new_policy(fit, m, p)
    for noise in args.noise:
        ref = torch.stack([gradient(policy, fit, m, a, noise, "pathwise", gen) for _ in range(4)]).mean(0)
        check = gradient(policy, fit, m, a, noise, "score", gen)
        print(f"[{name}] noise {noise}: |reference grad| {ref.norm():.4f}; score function on all {n_fit} training "
              f"episodes: relative error {(check - ref).norm() / ref.norm():.2f}")
        for batch in [64, 256, 1024, 4096]:
            for est in ["pathwise", "score"]:
                errs = []
                for _ in range(args.repeats):
                    idx = torch.randint(n_fit, (batch,), generator=gen)
                    errs.append(((gradient(policy, subset(fit, idx), m, a, noise, est, gen) - ref).norm() / ref.norm()).item())
                rows.append(dict(noise=noise, batch=batch, estimator=est, rel_error=float(np.mean(errs))))
                print(f"  batch {batch:5d}  {est:9s} relative error {np.mean(errs):.3f}")

    # (2) learning curves: same optimizer, batch and steps; only the gradient estimator differs
    noise = args.noise[0]
    for est in ["pathwise", "score"]:
        policy = new_policy(fit, m, p)
        optim = torch.optim.Adam(policy.parameters(), lr=p["lr"])
        for step in range(args.steps + 1):
            if step % 50 == 0:
                with torch.no_grad():
                    v = risk(episode_pnl(policy, val, m["cost_rate"], m["rate"], 1.0 / m["steps_per_year"], m["limits"]),
                             p["objective"]).item()
                curves.append(dict(estimator=est, step=step, episodes=step * p["batch"], val_risk=v))
            if step == args.steps:
                break
            idx = torch.randint(n_fit, (p["batch"],), generator=gen)
            gradient(policy, subset(fit, idx), m, a, noise, est, gen)
            optim.step()
        print(f"  {est:9s} learning curve: val risk {curves[-1]['val_risk']:.4f} after {args.steps} steps")

    errors, curves = pd.DataFrame(rows), pd.DataFrame(curves)
    errors.to_csv(out / "gradient_estimators.csv", index=False)
    curves.to_csv(out / "gradient_learning_curves.csv", index=False)
    with torch.no_grad():
        delta_risk = risk(episode_pnl(lambda f, d, x: d, val, m["cost_rate"], m["rate"], 1.0 / m["steps_per_year"],
                                      m["limits"]), p["objective"]).item()
    plot(errors, curves, delta_risk, noise, f"{name}: pathwise vs score-function gradients", out / "gradient_estimators.png")
    print(f"saved {out / 'gradient_estimators.png'}")


def plot(errors, curves, delta_risk, noise, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
    colors = {"pathwise": PALETTE[0], "score": PALETTE[1]}
    for (noise_, est), g in errors.groupby(["noise", "estimator"]):
        style = "-" if noise_ == errors.noise.min() else "--"
        ax1.plot(g.batch, g.rel_error, style, marker="o", color=colors[est], linewidth=2,
                 label=f"{'pathwise' if est == 'pathwise' else 'score function'}, s={noise_:g}")
    ax1.set(xscale="log", yscale="log", xlabel="episodes per gradient estimate", ylabel="relative error of the gradient")
    ax1.set_title("Gradient error at the initial policy", loc="left", fontsize=10)
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(color="#e7e6e2", linewidth=0.8)
    for est, g in curves.groupby("estimator"):
        ax2.plot(g.episodes / 1e6, g.val_risk, color=colors[est], linewidth=2,
                 label="pathwise (deep hedging)" if est == "pathwise" else "score function (REINFORCE)")
    ax2.axhline(delta_risk, color="#b5b4ae", linestyle=":", linewidth=1.5, label="daily delta hedge")
    ax2.set(xlabel="training episodes (millions)", ylabel="validation risk (lower is better)")
    ax2.set_title(f"Same optimizer and batch, exploration s={noise:g}", loc="left", fontsize=10)
    ax2.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax2.grid(axis="y", color="#e7e6e2", linewidth=0.8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
