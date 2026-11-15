"""
Build the project website's data and the poster figures from real runs.

    uv run deep_hedge.py --config configs/spy_vix_deep.json            # SPY models (if not trained yet)
    uv run gradient_estimators.py --config configs/gbm_costs.json      # gradient CSVs (if not run yet)
    uv run make_site.py                                                # ~6 min; writes docs/data.js, docs/figures/
    uv run make_site.py --figures-only                                 # redraw docs/figures/ from docs/data.js

What it computes, with the same code as the rest of the repo:
  - Black-Scholes world (configs/gbm_costs.json) at 0, 2, 5, 10 and 20 bps: solve the DP, train the
    learned band (entropic risk, a = 1), evaluate delta / Whalley-Wilmott / DP / learned in the
    backtester on the 20,476 test episodes, and record the bands and three example episodes.
  - SPY (configs/spy_vix_deep.json): out-of-sample metrics with block-bootstrap CIs, and replays of
    the five worst test episodes for the daily delta hedge plus two typical ones.
  - Verification numbers: DP grid refinement, DP value vs simulation, parity, Kamal & Derman.
"""

import copy
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backtest import (PALETTE, HedgingEnv, black_scholes, load_market_data, make_episodes, run_episode,
                      split_train_test)
from deep_hedge import DTYPE, _bs, _market, load_policy, train_entry
from dp_optimal import save as save_dp, solve_from_config
from strategies import STRATEGIES

DOCS = Path("docs")
FIGS = DOCS / "figures"
COSTS = [0, 2, 5, 10, 20]
LOG_M = np.round(np.linspace(-0.12, 0.12, 49), 4)
SERIES = {"delta": "#0b0b0b", "ww": PALETTE[1], "dp": PALETTE[2], "learned": PALETTE[0]}   # entity -> color
NAMES = {"delta": "daily delta", "ww": "Whalley-Wilmott", "dp": "DP optimal", "learned": "learned band"}


def r4(x):
    return np.round(np.asarray(x, dtype=float), 4).tolist()


def metrics(pnl, cost, trades):
    pnl = np.asarray(pnl)
    worst = np.sort(pnl)[: max(1, int(len(pnl) * 0.05))]
    return dict(entropic=float(np.log(np.mean(np.exp(-pnl)))), mean=float(pnl.mean()), std=float(pnl.std(ddof=1)),
                cvar=float(worst.mean()), cost=float(np.mean(cost)), trades=float(np.mean(trades)))


def trace(env, strategy, ep):
    res = run_episode(env, strategy, ep)
    h = env.history[:-1]                                     # drop the expiry row (sell-off)
    return dict(position=r4([x["position"] for x in h]), pnl=round(res["pnl"], 4), cost=round(res["cost"], 4),
                trades=res["n_trades"])


def pick_examples(eps):
    """Three test paths with different shapes: a rally, a chop around the strike, a sell-off."""
    ret = np.array([e.spots[-1] / e.spots[0] - 1 for e in eps])
    crossings = np.array([int(np.sum(np.diff(np.sign(e.spots - 100.0)) != 0)) for e in eps])
    rally = int(np.flatnonzero((ret > 0.06) & (ret < 0.09))[0])
    chop = int(np.flatnonzero((np.abs(ret) < 0.01) & (crossings >= 6))[0])
    selloff = int(np.flatnonzero((ret < -0.06) & (ret > -0.09))[0])
    return [(rally, f"Rally ({ret[rally]:+.0%})"), (chop, f"Chop around the strike ({crossings[chop]} crossings)"),
            (selloff, f"Sell-off ({ret[selloff]:+.0%})")]


def gbm_world():
    base = json.loads(Path("configs/gbm_costs.json").read_text())
    df = load_market_data(base)
    train_df, test_df = split_train_test(df, base["split"])
    n = base["option"]["maturity_steps"]
    test_eps = make_episodes(test_df, n, base["episodes"]["stride"])
    examples = pick_examples(test_eps)
    out = dict(costs=COSTS, log_m=LOG_M.tolist(), n_steps=n, n_test=len(test_eps), bands={}, risk={}, dp_value={},
               examples=[dict(name=name, spot=r4(test_eps[i].spots),
                              delta=r4([black_scholes(s, 100.0, (n - t) / 252, 0.2, 0.0)[1]
                                        for t, s in enumerate(test_eps[i].spots[:-1])]))
                         for i, name in examples],
               traces={}, checks={})
    pnl_at = {}
    parity = 0.0
    for c in COSTS:
        cfg = copy.deepcopy(base)
        cfg["market"]["cost_bps"] = c
        m = _market(cfg)
        dp_path, band_path = f"models/site/c{c}/dp.npz", f"models/site/c{c}/deep_band.pt"
        sol = solve_from_config(cfg, {"risk_aversion": 1.0})
        save_dp(sol, dp_path)
        entry = {"name": "deep_hedge", "label": "deep_band", "params": {
            "model": band_path, "arch": "band", "objective": {"type": "entropic", "risk_aversion": 1.0}}}
        _, gap = train_entry(entry, train_df, m, f"site c{c}", Path("results/site") / f"c{c}", log=lambda *a: None)
        parity = max(parity, gap)
        strategies = {"delta": STRATEGIES["delta_hedge"](), "ww": STRATEGIES["whalley_wilmott"](risk_aversion=1.0),
                      "dp": STRATEGIES["dp_optimal"](model=dp_path), "learned": STRATEGIES["deep_hedge"](model=band_path)}
        env = HedgingEnv("call", 1.0, 0.0, c, 252, cfg["market"]["position_limits"])
        risk, pnl_at[c] = {}, {}
        for key, strat in strategies.items():
            rows = [run_episode(env, strat, ep) for ep in test_eps]
            pnl_at[c][key] = np.array([r["pnl"] for r in rows])
            risk[key] = metrics(pnl_at[c][key], [r["cost"] for r in rows], [r["n_trades"] for r in rows])
        out["risk"][str(c)], out["dp_value"][str(c)] = risk, float(sol["risk"])
        out["traces"][str(c)] = [{k: trace(env, strat, test_eps[i]) for k, strat in strategies.items()}
                                 for i, _ in examples]
        out["bands"][str(c)] = bands(sol, load_policy(band_path), m)
        print(f"[site] {c:2d} bps: " + "  ".join(f"{k} {v['entropic']:.4f}" for k, v in risk.items())
              + f"  (DP value {sol['risk']:.4f})")

    # verification numbers
    cfg = copy.deepcopy(base)
    out["checks"]["dp_refinement"] = [[ns, nq, float(solve_from_config(cfg, {"n_spot": ns, "n_pos": nq})["risk"])]
                                      for ns, nq in [(201, 101), (401, 201), (801, 401)]]
    dp = pnl_at[10]["dp"]
    boot = [np.log(np.mean(np.exp(-dp[i]))) for i in np.random.default_rng(0).integers(0, len(dp), (2000, len(dp)))]
    out["checks"]["dp_mc"] = dict(value=out["dp_value"]["10"], mc=out["risk"]["10"]["dp"]["entropic"],
                                  lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))
    gaps = {}
    idx = np.random.default_rng(1).integers(0, len(dp), (2000, len(dp)))
    for key in ["learned", "delta", "ww"]:
        x = pnl_at[10][key]
        d = [np.log(np.mean(np.exp(-x[i]))) - np.log(np.mean(np.exp(-dp[i]))) for i in idx]
        gaps[key] = dict(gap=out["risk"]["10"][key]["entropic"] - out["risk"]["10"]["dp"]["entropic"],
                         lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
    out["checks"]["gap_to_dp"] = gaps
    out["checks"]["parity"] = parity
    tau, sigma = n / 252, 0.2
    vega = 100 * math.exp(-0.5 * (0.5 * sigma * math.sqrt(tau)) ** 2) / math.sqrt(2 * math.pi) * math.sqrt(tau)
    out["checks"]["kamal_derman"] = dict(predicted=math.sqrt(math.pi / 4) * vega * sigma / math.sqrt(n),
                                         observed=float(np.std(pnl_at[0]["delta"], ddof=1)))
    return out


def bands(sol, policy, m):
    """Position - delta at the band edges, for every steps-to-expiry and log-moneyness on the grid."""
    n, strike = m["n_steps"], 100.0
    spot = strike * np.exp(LOG_M)
    res = {"dp_lo": [], "dp_hi": [], "nn_lo": [], "nn_hi": [], "ww": []}
    for left in range(1, n + 1):
        t, tau = n - left, left / m["steps_per_year"]
        _, delta, gamma = _bs(spot, strike, tau, 0.2, m["rate"], kind="call")
        lo = np.interp(np.log(spot), np.log(sol["spots"]), sol["lower"][t])
        hi = np.interp(np.log(spot), np.log(sol["spots"]), sol["upper"][t])
        feats = torch.tensor(np.stack([LOG_M, np.full_like(LOG_M, math.sqrt(tau)), np.full_like(LOG_M, 0.2), delta], -1),
                             dtype=DTYPE)
        with torch.no_grad():
            nlo, nhi = (x.clamp(0, 1).numpy() for x in policy.band(feats, torch.tensor(delta, dtype=DTYPE)))
        half = (1.5 * m["cost_rate"] * spot * gamma**2 / 1.0) ** (1 / 3)
        res["dp_lo"].append(r4(lo - delta)), res["dp_hi"].append(r4(hi - delta))
        res["nn_lo"].append(r4(nlo - delta)), res["nn_hi"].append(r4(nhi - delta))
        res["ww"].append(r4(half))
    return res


def spy_world():
    cfg = json.loads(Path("configs/spy_vix_deep.json").read_text())
    _, test_df = split_train_test(load_market_data(cfg), cfg["split"])
    eps = make_episodes(test_df, cfg["option"]["maturity_steps"], cfg["episodes"]["stride"])
    mkt = cfg["market"]
    env = HedgingEnv("call", 1.0, mkt["rate"], mkt["cost_bps"], 252, mkt["position_limits"])
    entries = {s.get("label", s["name"]): s for s in cfg["strategies"]}
    labels = ["delta_daily", "delta_band_0.1", "ww_g1", "deep_band_cvar", "deep_mlp_cvar",
              "deep_band_cvar_garch", "deep_mlp_cvar_garch"]
    make = lambda lab: STRATEGIES[entries[lab]["name"]](**entries[lab].get("params", {}))
    pnl, rows = {}, []
    for lab in labels:
        res = [run_episode(env, make(lab), ep) for ep in eps]
        pnl[lab] = np.array([r["pnl"] for r in res])
        rows.append(dict(label=lab, **metrics(pnl[lab], [r["cost"] for r in res], [r["n_trades"] for r in res])))
    cvar = lambda x: np.sort(x)[: max(1, int(len(x) * 0.05))].mean()
    B, rng = 8, np.random.default_rng(0)
    idx = (rng.integers(0, len(eps) - B + 1, (5000, len(eps) // B))[:, :, None] + np.arange(B)).reshape(5000, -1)
    for row in rows:
        x, base = pnl[row["label"]], pnl["delta_daily"]
        own = np.array([cvar(x[i]) for i in idx])
        diff = np.array([cvar(x[i]) - cvar(base[i]) for i in idx])
        row.update(cvar_lo=float(np.percentile(own, 2.5)), cvar_hi=float(np.percentile(own, 97.5)),
                   gain=float(cvar(x) - cvar(base)), gain_lo=float(np.percentile(diff, 2.5)),
                   gain_hi=float(np.percentile(diff, 97.5)))
    order = np.argsort(pnl["delta_daily"])
    median = np.argsort(np.abs(pnl["delta_daily"] - np.median(pnl["delta_daily"])))
    chosen = list(order[:5]) + [i for i in median if i not in order[:5]][:2]
    shown = ["delta_daily", "delta_band_0.1", "ww_g1", "deep_band_cvar", "deep_mlp_cvar"]
    replays = []
    for i in chosen:
        ep = eps[i]
        tr = {lab: trace(env, make(lab), ep) for lab in shown}
        delta = [black_scholes(s, 100.0, (21 - t) / 252, v, mkt["rate"])[1]
                 for t, (s, v) in enumerate(zip(ep.spots[:-1], ep.vols[:-1]))]
        replays.append(dict(start=ep.start, kind="worst" if i in order[:5] else "typical", spot=r4(ep.spots),
                            vix=r4(ep.vols * 100), delta=r4(delta), traces=tr))
    return dict(n_test=len(eps), test_start=str(test_df.index[0].date()), test_end=str(test_df.index[-1].date()),
                summary=rows, replays=replays)


def gradients():
    root = Path("results/gbm_costs")
    errors = pd.read_csv(root / "gradient_estimators.csv")
    curves = pd.read_csv(root / "gradient_learning_curves.csv")
    shutil.copy(root / "gradient_estimators.png", FIGS / "gradient_estimators.png")
    return dict(errors=errors.to_dict("records"), curves=curves.round(5).to_dict("records"))


def poster_figures(gbm, spy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    b = gbm["bands"]["10"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, facecolor="#fcfcfb")
    for ax, left in zip(axes, [21, 10, 3]):
        k = left - 1
        ax.fill_between(LOG_M, b["dp_lo"][k], b["dp_hi"][k], color=SERIES["dp"], alpha=0.35, linewidth=0,
                        label=NAMES["dp"])
        ax.plot(LOG_M, b["nn_lo"][k], color=SERIES["learned"], linewidth=2, label=NAMES["learned"])
        ax.plot(LOG_M, b["nn_hi"][k], color=SERIES["learned"], linewidth=2)
        ax.plot(LOG_M, b["ww"][k], color=SERIES["ww"], linestyle="--", linewidth=2, label=f"{NAMES['ww']} (a = 1)")
        ax.plot(LOG_M, -np.array(b["ww"][k]), color=SERIES["ww"], linestyle="--", linewidth=2)
        ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=0)
        ax.set_title(f"{left} trading days to expiry", loc="left", fontsize=11)
        ax.set_xlabel("log(spot / strike)")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    axes[0].set_ylabel("shares held − delta")
    axes[-1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("The learned band is close to the optimal one; Whalley-Wilmott's is far too wide (10 bps)",
                 x=0.01, y=1.04, ha="left", fontsize=13, fontweight="bold")
    fig.savefig(FIGS / "bands.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#fcfcfb")
    for key in ["delta", "ww", "dp", "learned"]:
        y = [gbm["risk"][str(c)][key]["entropic"] for c in COSTS]
        ax.plot(COSTS, y, marker="o", markersize=7, linewidth=2, color=SERIES[key], label=NAMES[key],
                linestyle=":" if key == "delta" else "-")
    ax.set(xlabel="transaction cost (bps of each $ traded)", ylabel="entropic risk, a = 1 (lower is better)")
    ax.set_xticks(COSTS)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=10)
    ax.set_title(f"Out of sample, {gbm['n_test']:,} Black-Scholes episodes", loc="left", fontsize=11)
    fig.savefig(FIGS / "risk_vs_cost.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    names = {"delta_daily": "daily delta", "delta_band_0.1": "fixed band 0.1", "ww_g1": "Whalley-Wilmott",
             "deep_band_cvar": "learned band", "deep_mlp_cvar": "learned MLP",
             "deep_band_cvar_garch": "learned band (GARCH-trained)", "deep_mlp_cvar_garch": "learned MLP (GARCH-trained)"}
    rows = spy["summary"][::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor="#fcfcfb")
    for y, row in enumerate(rows):
        learned = row["label"].startswith("deep")
        color = PALETTE[0] if learned else "#52514e"
        ax.plot([row["cvar_lo"], row["cvar_hi"]], [y, y], color=color, linewidth=2, alpha=0.5)
        ax.plot(row["cvar"], y, "o", color=color, markersize=8)
    ax.set_yticks(range(len(rows)), [names[r["label"]] for r in rows])
    ax.set_xlabel(r"CVaR 5% of final P&L, \$ per \$100 of stock (higher is better)")
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_title(f"SPY, unseen {spy['test_start'][:4]}-{spy['test_end'][:4]} ({spy['n_test']} options); "
                 "95% block-bootstrap CI", loc="left", fontsize=11)
    fig.savefig(FIGS / "spy_cvar.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    if "--figures-only" in sys.argv:   # redraw the poster figures from the last run's docs/data.js
        text = (DOCS / "data.js").read_text()
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        poster_figures(data["gbm"], data["spy"])
        print(f"[site] redrew {FIGS}/ from {DOCS / 'data.js'}")
        return
    gbm = gbm_world()
    spy = spy_world()
    data = dict(gbm=gbm, spy=spy, gradients=gradients())
    # a script, not JSON: the page then also works when opened straight from disk (file://)
    (DOCS / "data.js").write_text("window.SITE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    poster_figures(gbm, spy)
    print(f"[site] wrote {DOCS / 'data.js'} ({(DOCS / 'data.js').stat().st_size / 1e3:.0f} kB) and {FIGS}/")


if __name__ == "__main__":
    main()
