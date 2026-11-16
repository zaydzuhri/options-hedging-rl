"""
Train the learned (deep hedging) policies, and compare gradient estimators.

    uv run train.py --config configs/gbm_bench.json               # train every "deep_hedge" strategy in the config
    uv run backtest.py --config configs/gbm_bench.json            # then evaluate them out of sample
    uv run train.py --config configs/gbm_bench.json --gradients   # pathwise vs score-function (REINFORCE) gradients

Why no RL machinery is needed: our trades don't move the market, so once a price path is fixed, the
final P&L is a deterministic, differentiable function of the positions we pick. We backpropagate a risk
measure of that P&L straight through the episode (the "pathwise" or reparameterization gradient) instead
of estimating the gradient from sampled actions the way policy-gradient methods do (Buehler et al., 2019,
"Deep hedging"). --gradients measures how much that matters.

Training uses only data before the config's split.test_start: its last `val_frac` picks the checkpoint
(chronologically), and with a "simulator" entry the policy trains on paths from a GJR-GARCH model fitted to
the rest (strategies/models.py). The policies themselves live in strategies/learned.py.

File layout (top to bottom):
  1. Episode tensors       -> price paths + Black-Scholes features, precomputed (they don't depend on actions)
  2. Differentiable P&L    -> exactly HedgingEnv's accounting, vectorized over a batch of episodes
  3. Risk measures         -> CVaR, entropic
  4. Training              -> Adam on minibatches, best validation checkpoint, backtester parity check
  5. Gradient estimators   -> pathwise vs score function, same episodes and noise (--gradients)
  6. CLI
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backtest import (Episode, HedgingEnv, black_scholes, load_market_data, make_episodes, run_episode,
                      split_train_test)
from strategies.learned import ARCHS, DTYPE, FEATURES, DeepHedge, load_policy, save_policy
from strategies.models import fit_gjr_garch, simulate_fitted_gjr_garch

_bs = np.vectorize(black_scholes, otypes=[float, float, float], excluded={"kind"})

DEFAULTS = dict(
    arch="band",                                   # "band" or "mlp" (strategies/learned.py)
    objective={"type": "cvar", "alpha": 0.05},     # or {"type": "entropic", "risk_aversion": 1.0}
    hidden=32, layers=2,
    steps=3000, batch=1024, lr=1e-3, seed=0,
    train_stride=1,                                # overlapping training episodes: more data
    simulator=None,                                # or {"type": "gjr_garch", "n_paths": 100000, "seed": 0}
    val_frac=0.2,                                  # last part of the training period picks the checkpoint
    eval_every=100,
)


# ---------------------------------------------------------------------------
# 1. Episode tensors
# ---------------------------------------------------------------------------

def market(cfg) -> dict:
    opt, mkt = cfg["option"], cfg["market"]
    return dict(kind=opt.get("type", "call"), moneyness=opt.get("moneyness", 1.0), rate=mkt.get("rate", 0.0),
                cost_rate=mkt.get("cost_bps", 0.0) / 1e4, steps_per_year=mkt.get("steps_per_year", 252),
                n_steps=opt["maturity_steps"], limits=mkt.get("position_limits"))


def episode_tensors(episodes, m) -> dict:
    """Stack episodes into tensors. Nothing here depends on our actions, so features are computed
    once, with the backtester's own Black-Scholes."""
    spots = np.stack([ep.spots for ep in episodes])            # [E, N+1]
    vols = np.stack([ep.vols for ep in episodes])
    n = spots.shape[1] - 1
    strike = m["moneyness"] * spots[:, :1]                      # [E, 1]; spots start at 100
    tau = np.broadcast_to((n - np.arange(n + 1)) * (1.0 / m["steps_per_year"]), spots.shape)
    price, delta, _ = _bs(spots, strike, tau, vols, m["rate"], kind=m["kind"])
    feats = np.stack([np.log(spots / strike), np.sqrt(tau), vols, delta], axis=-1)
    as_t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=DTYPE)
    return dict(spots=as_t(spots), features=as_t(feats[:, :n]), delta=as_t(delta[:, :n]),
                premium=as_t(price[:, 0]), payoff=as_t(price[:, n]))   # tau = 0 at step n: price = payoff


# ---------------------------------------------------------------------------
# 2. Differentiable P&L
# ---------------------------------------------------------------------------

def episode_pnl(policy, ep, m) -> torch.Tensor:
    """Final P&L of each episode in the batch. Same accounting as HedgingEnv.step, but on
    tensors, so gradients flow from the P&L back through every trade into the policy."""
    spots, n = ep["spots"], ep["spots"].shape[1] - 1
    lo, hi = m["limits"] if m["limits"] is not None else (-math.inf, math.inf)
    c, growth = m["cost_rate"], math.exp(m["rate"] / m["steps_per_year"])
    position, cash = torch.zeros_like(ep["premium"]), ep["premium"]   # we sold the option
    for t in range(n):
        target = policy(ep["features"][:, t], ep["delta"][:, t], position).clamp(lo, hi)
        trade = target - position
        cash = (cash - trade * spots[:, t] - c * trade.abs() * spots[:, t]) * growth
        position = target
    cash = cash + position * spots[:, n] - c * position.abs() * spots[:, n]   # sell off at expiry
    return cash - ep["payoff"]


# ---------------------------------------------------------------------------
# 3. Risk measures: the loss to minimize (bigger = worse)
# ---------------------------------------------------------------------------

def risk(pnl, objective: dict) -> torch.Tensor:
    """cvar:     average loss over the worst `alpha` fraction of episodes (expected shortfall).
    entropic: (1/a) log E[exp(-a * pnl)] = -CE, the certainty-equivalent loss of an investor with
              exponential utility and risk aversion a: what dp, mpc (cara) and ww_band optimize too."""
    kind = objective.get("type", "cvar")
    if kind == "cvar":
        k = max(1, math.ceil(objective.get("alpha", 0.05) * len(pnl)))
        return torch.topk(-pnl, k).values.mean()
    if kind == "entropic":
        a = objective.get("risk_aversion", 1.0)
        return (torch.logsumexp(-a * pnl, dim=0) - math.log(len(pnl))) / a
    raise ValueError(f"unknown objective {kind!r} (use 'cvar' or 'entropic')")


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def new_policy(fit, m, p, seed=0):
    torch.manual_seed(seed)
    flat = fit["features"].reshape(-1, len(FEATURES))
    limits = m["limits"] if m["limits"] is not None else (-math.inf, math.inf)
    return ARCHS[p["arch"]](flat.mean(0), flat.std(0), limits, p["hidden"], p["layers"]).to(DTYPE)


def train(fit, val, p, m, log=print):
    """Adam on minibatches of training episodes; keeps the checkpoint with the best validation risk."""
    policy = new_policy(fit, m, p, p["seed"])
    optim = torch.optim.Adam(policy.parameters(), lr=p["lr"])
    with torch.no_grad():
        ref = risk(episode_pnl(lambda f, d, x: d, val, m), p["objective"]).item()
    log(f"  val risk of the daily delta hedge (reference): {ref:.4f}")
    gen = torch.Generator().manual_seed(p["seed"])
    best_risk, best_state = math.inf, None
    for step in range(1, p["steps"] + 1):
        idx = torch.randint(len(fit["premium"]), (p["batch"],), generator=gen)
        loss = risk(episode_pnl(policy, {k: v[idx] for k, v in fit.items()}, m), p["objective"])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % p["eval_every"] == 0 or step == p["steps"]:
            with torch.no_grad():
                val_risk = risk(episode_pnl(policy, val, m), p["objective"]).item()
            if val_risk < best_risk:
                best_risk, best_state = val_risk, copy.deepcopy(policy.state_dict())
            if step % (10 * p["eval_every"]) == 0 or step == p["steps"]:
                log(f"  step {step:5d}  batch risk {loss.item():.4f}  val risk {val_risk:.4f}")
    policy.load_state_dict(best_state)
    log(f"  best val risk {best_risk:.4f} (delta hedge: {ref:.4f})")
    return policy.eval()


def check_parity(model_path, params, episodes, m, n=50):
    """The backtester and the differentiable P&L must agree, or we trained on the wrong problem."""
    env = HedgingEnv(m["kind"], m["moneyness"], m["rate"], m["cost_rate"] * 1e4, m["steps_per_year"], m["limits"])
    strategy = DeepHedge(**params)
    episodes = episodes[:n]
    env_pnl = np.array([run_episode(env, strategy, ep)["pnl"] for ep in episodes])
    with torch.no_grad():
        torch_pnl = episode_pnl(load_policy(model_path), episode_tensors(episodes, m), m)
    gap = np.abs(env_pnl - torch_pnl.numpy()).max()
    assert gap < 1e-8, f"backtester and differentiable P&L disagree by {gap:.2e}"
    return gap


def split_fit_val(train_df, p, m):
    cut = int(len(train_df) * (1 - p["val_frac"]))   # chronological: validation is the latest training data
    return (make_episodes(train_df.iloc[:cut], m["n_steps"], p["train_stride"]),
            make_episodes(train_df.iloc[cut:], m["n_steps"], p["train_stride"]), cut)


def train_entry(entry, train_df, m, name, log=print):
    """Train one "deep_hedge" config entry on `train_df`, save it, check parity. Returns (policy, parity gap)."""
    label, params = entry.get("label", entry["name"]), entry["params"]
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k != "model"}}
    fit_eps, val_eps, cut = split_fit_val(train_df, p, m)
    if p["simulator"]:
        sim = p["simulator"]
        if sim.get("type") != "gjr_garch":
            raise SystemExit(f"unknown simulator {sim.get('type')!r} (use 'gjr_garch')")
        part = train_df.iloc[:cut]
        model = fit_gjr_garch(part["spot"], part["vol"], m["steps_per_year"])
        spots, vols = simulate_fitted_gjr_garch(model, sim.get("n_paths", 100_000), m["n_steps"], sim.get("seed", 0))
        fit_eps = [Episode(f"gjr-garch-{i}", s, v) for i, (s, v) in enumerate(zip(spots, vols))]
        log(f"  GJR-GARCH fitted on {train_df.index[0]} -> {train_df.index[cut - 1]}: "
            + ", ".join(f"{k} {model[k]:.4g}" for k in ["omega", "alpha", "gamma", "beta", "persistence"]))
    log(f"[{name}] {label}: {p['arch']} policy, objective {p['objective']}, "
        f"{len(fit_eps)} training / {len(val_eps)} validation episodes")
    policy = train(episode_tensors(fit_eps, m), episode_tensors(val_eps, m), p, m, log)
    save_policy(policy, p, params["model"])
    gap = check_parity(params["model"], params, val_eps, m)
    log(f"  saved {params['model']} (backtester parity: max |Δpnl| = {gap:.1e})")
    return policy, gap


# ---------------------------------------------------------------------------
# 5. Gradient estimators: why not plain RL?
# ---------------------------------------------------------------------------

def gradient(policy, ep, m, a, noise, estimator, gen) -> torch.Tensor:
    """One estimate of the gradient of E[exp(-a P&L)] for the Gaussian policy u_t = π_θ(x_t) + noise·ξ_t:
      pathwise:        ∇θ exp(-a P&L), backpropagated through every trade and state (what train() uses)
      score function:  exp(-a P&L) · Σ_t ∇θ log N(u_t; π_θ(x_t), noise²) (REINFORCE / PPO), with a
                       leave-one-out baseline (still unbiased). It never differentiates the P&L, so its
                       variance grows with the horizon and with 1/noise."""
    policy.zero_grad()
    if estimator == "pathwise":
        act = lambda f, d, x: policy(f, d, x) + noise * torch.randn(len(x), generator=gen, dtype=DTYPE)
        torch.exp(-a * episode_pnl(act, ep, m)).mean().backward()
    else:
        logp = []

        def act(f, d, x):
            mean = policy(f, d, x.detach())
            u = (mean + noise * torch.randn(len(x), generator=gen, dtype=DTYPE)).detach()
            logp.append(-0.5 * ((u - mean) / noise) ** 2)
            return u

        utility = torch.exp(-a * episode_pnl(act, ep, m).detach())
        baseline = (utility.sum() - utility) / (len(utility) - 1)        # leave-one-out
        ((utility - baseline) * torch.stack(logp).sum(0)).mean().backward()
    return torch.cat([q.grad.flatten() for q in policy.parameters()]).clone()


def gradient_study(cfg, out, label=None, noises=(0.01, 0.05), repeats=30, steps=1500):
    """(1) relative error of each estimator against a large-batch reference vs batch size, at the initial
    policy; (2) learning curves with the same optimizer, batch and episodes. Writes CSVs and a figure."""
    name = cfg.get("name")
    entry = next(s for s in cfg["strategies"] if s["name"] == "deep_hedge" and label in (None, s.get("label")))
    p = {**DEFAULTS, **{k: v for k, v in entry["params"].items() if k != "model"}}
    a, m = p["objective"].get("risk_aversion", 1.0), market(cfg)
    train_df, _ = split_train_test(load_market_data(cfg), cfg["split"])
    fit_eps, val_eps, _ = split_fit_val(train_df, {**p, "train_stride": 1}, m)
    fit, val = episode_tensors(fit_eps, m), episode_tensors(val_eps, m)
    n_fit = len(fit["premium"])
    sub = lambda ep, idx: {k: v[idx] for k, v in ep.items()}
    gen = torch.Generator().manual_seed(1)
    rows, curves = [], []
    policy = new_policy(fit, m, p)
    for noise in noises:
        ref = torch.stack([gradient(policy, fit, m, a, noise, "pathwise", gen) for _ in range(4)]).mean(0)
        print(f"[{name}] noise {noise}: |reference grad| {ref.norm():.4f}")
        for batch in [64, 256, 1024, 4096]:
            for est in ["pathwise", "score"]:
                errs = [((gradient(policy, sub(fit, torch.randint(n_fit, (batch,), generator=gen)), m, a, noise, est, gen)
                          - ref).norm() / ref.norm()).item() for _ in range(repeats)]
                rows.append(dict(noise=noise, batch=batch, estimator=est, rel_error=float(np.mean(errs))))
                print(f"  batch {batch:5d}  {est:9s} relative error {np.mean(errs):.3f}")
    noise = noises[0]
    for est in ["pathwise", "score"]:
        policy = new_policy(fit, m, p)
        optim = torch.optim.Adam(policy.parameters(), lr=p["lr"])
        for step in range(steps + 1):
            if step % 50 == 0:
                with torch.no_grad():
                    v = risk(episode_pnl(policy, val, m), p["objective"]).item()
                curves.append(dict(estimator=est, step=step, episodes=step * p["batch"], val_risk=v))
            if step == steps:
                break
            gradient(policy, sub(fit, torch.randint(n_fit, (p["batch"],), generator=gen)), m, a, noise, est, gen)
            optim.step()
        print(f"  {est:9s} learning curve: val risk {curves[-1]['val_risk']:.4f} after {steps} steps")
    errors, curves = pd.DataFrame(rows), pd.DataFrame(curves)
    errors.to_csv(out / "gradient_estimators.csv", index=False)
    curves.to_csv(out / "gradient_learning_curves.csv", index=False)
    with torch.no_grad():
        delta_risk = risk(episode_pnl(lambda f, d, x: d, val, m), p["objective"]).item()
    plot_gradients(errors, curves, delta_risk, noise, f"{name}: pathwise vs score-function gradients",
                   out / "gradient_estimators.png")
    print(f"saved {out / 'gradient_estimators.png'}")


def plot_gradients(errors, curves, delta_risk, noise, title, path):
    from plots import AXIS, INK, SLOTS, _style
    plt = _style()
    colors = {"pathwise": SLOTS[2], "score": SLOTS[4]}
    names = {"pathwise": "pathwise (deep hedging)", "score": "score function (REINFORCE)"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for (noise_, est), g in errors.groupby(["noise", "estimator"]):
        filled = noise_ == errors.noise.min()
        ax1.plot(g.batch, g.rel_error, marker="o", color=colors[est], lw=2, mfc=colors[est] if filled else "white",
                 label=f"{names[est]}, noise {noise_:g}")
    ax1.set(xscale="log", yscale="log", xlabel="episodes per gradient estimate", ylabel="relative error of the gradient")
    ax1.set_title("Gradient error at the initial policy")
    ax1.legend(fontsize=8.5)
    for est, g in curves.groupby("estimator"):
        ax2.plot(g.episodes / 1e6, g.val_risk, color=colors[est], lw=2, label=names[est])
    ax2.axhline(delta_risk, color=AXIS, lw=1.5, label="daily delta hedge")
    ax2.set(xlabel="training episodes (millions)", ylabel="validation risk (lower is better)")
    ax2.set_title(f"Same optimizer and batch, noise {noise:g}")
    ax2.legend(fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="a config with a 'split' and 'deep_hedge' strategies")
    parser.add_argument("--gradients", action="store_true", help="run the gradient-estimator study instead of training")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    name = cfg.get("name", Path(args.config).stem)
    if "split" not in cfg:
        raise SystemExit("train.py needs a 'split' in the config, so training never sees the test period")
    jobs = [s for s in cfg["strategies"] if s["name"] == "deep_hedge"]
    if not jobs:
        raise SystemExit("no 'deep_hedge' strategies in this config")
    out = Path(cfg.get("output_dir", "results")) / name
    out.mkdir(parents=True, exist_ok=True)
    if args.gradients:
        return gradient_study(cfg, out)
    train_df, _ = split_train_test(load_market_data(cfg), cfg["split"])
    for s in jobs:
        train_entry(s, train_df, market(cfg), name)


if __name__ == "__main__":
    main()
