"""
Deep hedging: learn a hedging policy by differentiating the P&L of whole episodes.

    uv run deep_hedge.py --config configs/gbm_deep.json   # train every "deep_hedge" strategy in the config
    uv run backtest.py   --config configs/gbm_deep.json   # evaluate them out of sample next to the baselines

Why no RL machinery is needed: our trades don't move the market, so once a price path is
fixed, the final P&L is a deterministic, differentiable function of the positions we pick.
We backpropagate a risk measure of that P&L straight through the episode (the "pathwise" or
reparameterization gradient) instead of estimating the gradient from sampled actions the way
policy-gradient methods do. See Buehler et al. (2019), "Deep hedging".

File layout (top to bottom):
  1. Episode tensors       -> price paths + Black-Scholes features, precomputed (they don't depend on actions)
  2. Policies              -> NoTransactionBand (default) and MLPHedge
  3. Differentiable P&L    -> exactly HedgingEnv's accounting, vectorized over a batch of episodes
  4. Risk measures         -> CVaR, entropic
  5. Training, checkpoints, band plot, CLI
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backtest import (HedgingEnv, black_scholes, load_market_data, make_episodes, run_episode,
                      split_train_test)
from plots import SLOTS as PALETTE   # the repo's categorical colors

DTYPE = torch.float64   # same precision as the numpy backtester, so the two agree to ~1e-12
FEATURES = ["log_moneyness", "sqrt_tau", "vol", "delta"]   # what a policy sees, besides its position
_bs = np.vectorize(black_scholes, otypes=[float, float, float], excluded={"kind"})


# ---------------------------------------------------------------------------
# 1. Episode tensors
# ---------------------------------------------------------------------------

def episode_tensors(episodes, kind, moneyness, rate, steps_per_year) -> dict:
    """Stack episodes into tensors. Nothing here depends on our actions, so features are computed
    once, with the backtester's own Black-Scholes."""
    spots = np.stack([ep.spots for ep in episodes])            # [E, N+1]
    vols = np.stack([ep.vols for ep in episodes])
    n = spots.shape[1] - 1
    strike = moneyness * spots[:, :1]                           # [E, 1]; spots start at 100
    tau = np.broadcast_to((n - np.arange(n + 1)) * (1.0 / steps_per_year), spots.shape)
    price, delta, _ = _bs(spots, strike, tau, vols, rate, kind=kind)
    feats = np.stack([np.log(spots / strike), np.sqrt(tau), vols, delta], axis=-1)
    as_t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=DTYPE)
    return dict(spots=as_t(spots), features=as_t(feats[:, :n]), delta=as_t(delta[:, :n]),
                premium=as_t(price[:, 0]), payoff=as_t(price[:, n]))   # tau = 0 at step n: price = payoff


# ---------------------------------------------------------------------------
# 2. Policies: (features, delta, current position) -> target position
# ---------------------------------------------------------------------------

def _mlp(n_in, n_out, hidden, layers):
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    last = nn.Linear(d, n_out)
    nn.init.zeros_(last.weight)   # start from a constant output, see each policy's docstring
    return nn.Sequential(*mods, last)


class Policy(nn.Module):
    """Base: standardizes features with training-set statistics and projects the output onto the
    position limits, so a policy never asks for an infeasible position (both stored in the checkpoint)."""

    def __init__(self, feat_mean, feat_std, limits):
        super().__init__()
        self.register_buffer("feat_mean", feat_mean)
        self.register_buffer("feat_std", torch.where(feat_std > 1e-8, feat_std, 1.0))   # constant vol -> std 0
        self.register_buffer("limits", torch.as_tensor(limits, dtype=feat_mean.dtype))

    def norm(self, feats):
        return (feats - self.feat_mean) / self.feat_std

    def project(self, target):
        return torch.minimum(torch.maximum(target, self.limits[0]), self.limits[1])


class NoTransactionBand(Policy):
    """Learn a band [delta - lower, delta + upper] around Black-Scholes delta: inside it, don't
    trade; outside it, trade to the nearest edge. The band depends only on the market state, so
    "don't trade" is built into the architecture (Imaki et al., 2021): the same shape as
    Whalley-Wilmott, with a learned (and possibly asymmetric) width. Starts as a 0.05 band."""

    def __init__(self, feat_mean, feat_std, limits, hidden=32, layers=2, init_width=0.05):
        super().__init__(feat_mean, feat_std, limits)
        self.net = _mlp(len(FEATURES), 2, hidden, layers)
        nn.init.constant_(self.net[-1].bias, math.log(math.expm1(init_width)))   # softplus^-1

    def band(self, feats, delta):
        lower, upper = nn.functional.softplus(self.net(self.norm(feats))).unbind(-1)
        return delta - lower, delta + upper

    def forward(self, feats, delta, position):
        lo, hi = self.band(feats, delta)
        return self.project(torch.minimum(torch.maximum(position, lo), hi))


class MLPHedge(Policy):
    """Unconstrained: target = delta + correction(features, current position).
    The correction starts at zero, so training starts from the daily delta hedge."""

    def __init__(self, feat_mean, feat_std, limits, hidden=32, layers=2):
        super().__init__(feat_mean, feat_std, limits)
        self.net = _mlp(len(FEATURES) + 1, 1, hidden, layers)

    def forward(self, feats, delta, position):
        x = torch.cat([self.norm(feats), position.unsqueeze(-1)], dim=-1)
        return self.project(delta + self.net(x).squeeze(-1))


ARCHS = {"band": NoTransactionBand, "mlp": MLPHedge}


def act(policy, obs) -> float:
    """One step of a trained policy on a backtester Observation (same features as in training)."""
    feats = torch.tensor([[math.log(obs.spot / obs.strike), math.sqrt(obs.tau), obs.vol, obs.delta]], dtype=DTYPE)
    with torch.no_grad():
        target = policy(feats, torch.tensor([obs.delta], dtype=DTYPE), torch.tensor([obs.position], dtype=DTYPE))
    return target.item()


# ---------------------------------------------------------------------------
# 3. Differentiable P&L
# ---------------------------------------------------------------------------

def episode_pnl(policy, ep, cost_rate, rate, dt, limits=None) -> torch.Tensor:
    """Final P&L of each episode in the batch. Same accounting as HedgingEnv.step, but on
    tensors, so gradients flow from the P&L back through every trade into the policy."""
    spots = ep["spots"]
    n = spots.shape[1] - 1
    lo, hi = limits if limits is not None else (-math.inf, math.inf)
    position, cash = torch.zeros_like(ep["premium"]), ep["premium"]   # we sold the option
    growth = math.exp(rate * dt)
    for t in range(n):
        target = policy(ep["features"][:, t], ep["delta"][:, t], position).clamp(lo, hi)
        trade = target - position
        cash = (cash - trade * spots[:, t] - cost_rate * trade.abs() * spots[:, t]) * growth
        position = target
    cash = cash + position * spots[:, n] - cost_rate * position.abs() * spots[:, n]   # sell off at expiry
    return cash - ep["payoff"]


# ---------------------------------------------------------------------------
# 4. Risk measures: the loss to minimize (bigger = worse)
# ---------------------------------------------------------------------------

def risk(pnl, objective: dict) -> torch.Tensor:
    """cvar:     average loss over the worst `alpha` fraction of episodes (expected shortfall).
    entropic: (1/a) log E[exp(-a * pnl)], the certainty-equivalent loss of an investor with
              exponential utility and risk aversion a: the objective Whalley-Wilmott is built for."""
    kind = objective.get("type", "cvar")
    if kind == "cvar":
        k = max(1, math.ceil(objective.get("alpha", 0.05) * len(pnl)))
        return torch.topk(-pnl, k).values.mean()
    if kind == "entropic":
        a = objective.get("risk_aversion", 1.0)
        return (torch.logsumexp(-a * pnl, dim=0) - math.log(len(pnl))) / a
    raise ValueError(f"unknown objective {kind!r} (use 'cvar' or 'entropic')")


# ---------------------------------------------------------------------------
# 5. Training, checkpoints, band plot, CLI
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    arch="band",                                   # "band" or "mlp"
    objective={"type": "cvar", "alpha": 0.05},     # or {"type": "entropic", "risk_aversion": 1.0}
    hidden=32, layers=2,
    steps=3000, batch=1024, lr=1e-3, seed=0,
    train_stride=1,                                # overlapping training episodes: more data
    simulator=None,                                # or {"type": "gjr_garch", "n_paths": 100000, "seed": 0}:
                                                   #   train on paths from a model fitted to the training period
    val_frac=0.2,                                  # last part of the training period picks the checkpoint
    eval_every=100,
)


def _market(cfg):
    opt, mkt = cfg["option"], cfg["market"]
    return dict(kind=opt.get("type", "call"), moneyness=opt.get("moneyness", 1.0), rate=mkt.get("rate", 0.0),
                cost_rate=mkt.get("cost_bps", 0.0) / 1e4, steps_per_year=mkt.get("steps_per_year", 252),
                n_steps=opt["maturity_steps"], limits=mkt.get("position_limits"))


def train(fit, val, p, m, log=print):
    """Adam on minibatches of training episodes; keeps the checkpoint with the best validation risk."""
    torch.manual_seed(p["seed"])
    flat = fit["features"].reshape(-1, len(FEATURES))
    limits = m["limits"] if m["limits"] is not None else (-math.inf, math.inf)
    policy = ARCHS[p["arch"]](flat.mean(0), flat.std(0), limits, p["hidden"], p["layers"]).to(DTYPE)
    optim = torch.optim.Adam(policy.parameters(), lr=p["lr"])
    pnl_of = lambda pol, ep: episode_pnl(pol, ep, m["cost_rate"], m["rate"], 1.0 / m["steps_per_year"], m["limits"])
    with torch.no_grad():
        ref = risk(pnl_of(lambda f, d, x: d, val), p["objective"]).item()
    log(f"  val risk of the daily delta hedge (reference): {ref:.4f}")

    gen = torch.Generator().manual_seed(p["seed"])
    best_risk, best_state = math.inf, None
    for step in range(1, p["steps"] + 1):
        idx = torch.randint(len(fit["premium"]), (p["batch"],), generator=gen)
        loss = risk(pnl_of(policy, {k: v[idx] for k, v in fit.items()}), p["objective"])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % p["eval_every"] == 0 or step == p["steps"]:
            with torch.no_grad():
                val_risk = risk(pnl_of(policy, val), p["objective"]).item()
            if val_risk < best_risk:
                best_risk, best_state = val_risk, copy.deepcopy(policy.state_dict())
            if step % (10 * p["eval_every"]) == 0 or step == p["steps"]:
                log(f"  step {step:5d}  batch risk {loss.item():.4f}  val risk {val_risk:.4f}")
    policy.load_state_dict(best_state)
    log(f"  best val risk {best_risk:.4f} (delta hedge: {ref:.4f})")
    return policy.eval()


def save_policy(policy, p, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(arch=p["arch"], hidden=p["hidden"], layers=p["layers"], params=p,
                    state_dict=policy.state_dict()), path)


def load_policy(path):
    ckpt = torch.load(path, weights_only=True)
    state = ckpt["state_dict"]
    policy = ARCHS[ckpt["arch"]](state["feat_mean"], state["feat_std"], state["limits"], ckpt["hidden"],
                                 ckpt["layers"]).to(DTYPE)
    policy.load_state_dict(state)
    return policy.eval()


def check_parity(model_path, params, episodes, m, n=50):
    """The backtester and the differentiable P&L must agree, or we trained on the wrong problem."""
    from strategies.learned import DeepHedge

    env = HedgingEnv(m["kind"], m["moneyness"], m["rate"], m["cost_rate"] * 1e4, m["steps_per_year"], m["limits"])
    strategy = DeepHedge(**params)
    episodes = episodes[:n]
    env_pnl = np.array([run_episode(env, strategy, ep)["pnl"] for ep in episodes])
    ep = episode_tensors(episodes, m["kind"], m["moneyness"], m["rate"], m["steps_per_year"])
    with torch.no_grad():
        torch_pnl = episode_pnl(load_policy(model_path), ep, m["cost_rate"], m["rate"], 1.0 / m["steps_per_year"],
                                m["limits"])
    gap = np.abs(env_pnl - torch_pnl.numpy()).max()
    assert gap < 1e-8, f"backtester and differentiable P&L disagree by {gap:.2e}"
    return gap


def plot_band(policy, m, vol, risk_aversion, title, path):
    """Learned no-trade band vs Whalley-Wilmott's, around delta, at a few times to expiry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, dt, strike = m["n_steps"], 1.0 / m["steps_per_year"], 100.0 * m["moneyness"]
    log_m = np.linspace(-0.12, 0.12, 241)
    spot = strike * np.exp(log_m)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    left_list = sorted({n, max(1, n // 2), max(1, n // 7)}, reverse=True)
    fig, axes = plt.subplots(1, len(left_list), figsize=(4.2 * len(left_list), 3.6), sharey=True,
                             facecolor="#fcfcfb")
    for ax, left in zip(np.atleast_1d(axes), left_list):
        tau = left * dt
        _, delta, gamma = _bs(spot, strike, tau, vol, m["rate"], kind=m["kind"])
        feats = torch.tensor(np.stack([log_m, np.full_like(log_m, math.sqrt(tau)), np.full_like(log_m, vol), delta], -1),
                             dtype=DTYPE)
        with torch.no_grad():
            lo, hi = policy.band(feats, torch.tensor(delta, dtype=DTYPE))
        ww = (1.5 * math.exp(-m["rate"] * tau) * m["cost_rate"] * spot * gamma**2 / risk_aversion) ** (1 / 3)
        ax.fill_between(log_m, lo.numpy() - delta, hi.numpy() - delta, color=PALETTE[0], alpha=0.35,
                        linewidth=0, label="learned band")
        ax.plot(log_m, ww, color="#0b0b0b", linestyle="--", linewidth=1.5, label=f"Whalley-Wilmott (γ={risk_aversion:g})")
        ax.plot(log_m, -ww, color="#0b0b0b", linestyle="--", linewidth=1.5)
        ax.axhline(0, color="#b5b4ae", linewidth=1, zorder=0)
        ax.set_title(f"{left} steps to expiry", loc="left", fontsize=10)
        ax.set_xlabel("log(spot / strike)")
        ax.grid(axis="y", color="#e7e6e2", linewidth=0.8)
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("allowed position − delta (shares)")
    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="a config with a 'split' and 'deep_hedge' strategies")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    name = cfg.get("name", Path(args.config).stem)
    if "split" not in cfg:
        raise SystemExit("deep_hedge.py needs a 'split' in the config, so training never sees the test period")
    jobs = [s for s in cfg["strategies"] if s["name"] == "deep_hedge"]
    if not jobs:
        raise SystemExit("no 'deep_hedge' strategies in this config")

    m = _market(cfg)
    train_df, _ = split_train_test(load_market_data(cfg), cfg["split"])
    out = Path(cfg.get("output_dir", "results")) / name
    out.mkdir(parents=True, exist_ok=True)
    for s in jobs:
        train_entry(s, train_df, m, name, out)


def train_entry(entry, train_df, m, name, out, log=print):
    """Train one "deep_hedge" config entry on `train_df`, save it, check parity, plot its band."""
    label, params = entry.get("label", entry["name"]), entry["params"]
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k != "model"}}
    cut = int(len(train_df) * (1 - p["val_frac"]))   # chronological: validation is the latest training data
    fit_eps = make_episodes(train_df.iloc[:cut], m["n_steps"], p["train_stride"])
    val_eps = make_episodes(train_df.iloc[cut:], m["n_steps"], p["train_stride"])
    if p["simulator"]:
        sim = p["simulator"]
        if sim.get("type") != "gjr_garch":
            raise SystemExit(f"unknown simulator {sim.get('type')!r} (use 'gjr_garch')")
        import garch
        model = garch.fit(train_df.iloc[:cut])
        fit_eps = garch.simulate(model, sim.get("n_paths", 100_000), m["n_steps"], sim.get("seed", 0))
        log(f"  GJR-GARCH fitted on {train_df.index[0]} -> {train_df.index[cut - 1]}: "
            + ", ".join(f"{k} {model[k]:.4g}" for k in ["omega", "alpha", "gamma", "beta", "persistence"]))
    log(f"[{name}] {label}: {p['arch']} policy, objective {p['objective']}, "
        f"{len(fit_eps)} training / {len(val_eps)} validation episodes")
    tensors = lambda eps: episode_tensors(eps, m["kind"], m["moneyness"], m["rate"], m["steps_per_year"])
    policy = train(tensors(fit_eps), tensors(val_eps), p, m, log)
    save_policy(policy, p, params["model"])
    gap = check_parity(params["model"], params, val_eps, m)
    log(f"  saved {params['model']} (backtester parity: max |Δpnl| = {gap:.1e})")
    if p["arch"] == "band":
        obj = p["objective"]
        ww_gamma = obj.get("risk_aversion", 1.0) if obj.get("type") == "entropic" else 1.0
        vol = float(np.median(train_df["vol"]))
        out.mkdir(parents=True, exist_ok=True)
        plot_band(policy, m, vol, ww_gamma, f"{name} / {label}: no-trade band at vol {vol:.2f}", out / f"band_{label}.png")
        log(f"  band plot: {out / f'band_{label}.png'}")
    return policy, gap

if __name__ == "__main__":
    main()
