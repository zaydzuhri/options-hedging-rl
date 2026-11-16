"""
Build the project website's data (docs/data.js) and the poster figures (docs/figures/) from real runs.

    uv run train.py --config configs/gbm_bench.json --gradients   # the gradient study (if not run yet)
    uv run make_site.py [--workers 10]       # ~2.5 h the first time (MPC on 5 x 20,476 episodes); cached after
    uv run make_site.py --figures-only       # redraw docs/figures/ from docs/data.js

Every number comes from the backtester (backtest.run_strategies). Per-strategy results are cached in
results/site/cache/ by (world, strategy entry, model file), so a re-run only recomputes what changed, and
learned policies that are missing are trained first (train.py).
  - Black-Scholes world (configs/gbm_bench.json) at 0, 2, 5, 10 and 20 bps: daily delta, Whalley-Wilmott,
    DP (the exact optimum), MPC band and the learned band (trained for each cost) on 20,476 unseen episodes;
    their no-trade bands (MPC's probed from both sides); three example paths.
  - Real data (configs/{spy,ko,jnj,aapl}_bench.json): every strategy out of sample on 2019-2025, with paired
    block-bootstrap 95% CIs against daily delta, and replays of the worst and two typical test months.
  - Verification numbers: DP grid refinement, DP value vs simulation, gaps to the optimum, parity, Kamal & Derman.
"""

import argparse
import copy
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backtest import load_episodes, load_market_data, make_env, run_episode, run_strategies, split_train_test
from plots import AXIS, GRID, INK, INK2, SLOTS, _style, compare, run_from
from strategies import STRATEGIES
from strategies.dp import h_grid, optimal_risk, solve_gbm
from strategies.learned import DTYPE, load_policy
from strategy import Observation, black_scholes
from train import check_parity, market, split_fit_val, train_entry

DOCS, FIGS, CACHE = Path("docs"), Path("docs/figures"), Path("results/site/cache")
COSTS = [0, 2, 5, 10, 20]
LOG_M = np.round(np.linspace(-0.12, 0.12, 49), 4)
ASSETS = [("spy", "SPY", "VIX"), ("ko", "KO", "21-day realized vol"), ("jnj", "JNJ", "21-day realized vol"),
          ("aapl", "AAPL", "21-day realized vol")]
SHOWN = ["delta_daily", "ww", "dp", "mpc_band", "deep_band"]           # policies drawn in the real-data replays
COLOR = {"mpc": SLOTS[0], "dp": SLOTS[1], "learned": SLOTS[2], "ww": SLOTS[3], "mlp": SLOTS[4], "delta": INK}
NAME = {"delta": "daily delta", "ww": "Whalley-Wilmott", "dp": "DP (optimal)", "mpc": "MPC band", "learned": "learned band"}
REAL = {"delta_daily": ("delta", "daily delta"), "ww": ("ww", "Whalley-Wilmott"), "dp": ("dp", "DP (Black-Scholes model)"),
        "mpc_band": ("mpc", "MPC band"), "deep_band": ("learned", "learned band"), "deep_mlp": ("mlp", "learned MLP"),
        "deep_band_garch": ("learned", "learned band, GARCH-trained")}


def r4(x):
    return np.round(np.asarray(x, dtype=float), 4).tolist()


# ---------------------------------------------------------------------------- evaluation (cached)
def evaluate(cfg, entry, workers=10) -> pd.DataFrame:
    """One strategy's per-episode results in one world, cached in results/site/cache/ by (world, strategy
    entry, model file contents), so re-running the site only recomputes what changed."""
    world = {k: cfg[k] for k in ("data", "vol", "option", "market", "split", "episodes") if k in cfg}
    model = entry.get("params", {}).get("model")
    stamp = hashlib.sha1(Path(model).read_bytes()).hexdigest() if model else None
    key = hashlib.sha1(json.dumps([world, entry, stamp], sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE / f"{entry.get('label', entry['name'])}_{key}.csv"
    if path.exists():
        return pd.read_csv(path)
    rows, _ = run_strategies({**cfg, "strategies": [entry]}, workers)
    CACHE.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)
    return rows


def ensure_trained(cfg, entries, log=print):
    """Train the learned entries whose checkpoint is missing; returns the largest backtester parity gap."""
    train_df, m, gap = None, market(cfg), 0.0
    for e in entries:
        if e["name"] != "deep_hedge":
            continue
        if train_df is None:
            train_df, _ = split_train_test(load_market_data(cfg), cfg["split"])
        if not Path(e["params"]["model"]).exists():
            train_entry(e, train_df, m, cfg["name"], log=lambda *a: None)
        p = {"val_frac": 0.2, "train_stride": 1}
        gap = max(gap, check_parity(e["params"]["model"], e["params"], split_fit_val(train_df, p, m)[1], m))
    return gap


def trace(env, strategy, ep):
    res = run_episode(env, strategy, ep)
    h = env.history[:-1]                                     # drop the expiry row (sell-off)
    return dict(position=r4([x["position"] for x in h]), pnl=round(res["pnl"], 4), cost=round(res["cost"], 4),
                trades=res["n_trades"])


def metrics(run, base, n_boot=2000):
    """Point values and paired (block) bootstrap 95% CIs vs `base` from plots.compare: entropic risk = -CE."""
    table, block = compare(run, base, n_boot)
    t = table.set_index(["metric", "strategy"])
    out = {}
    for lab in run["labels"]:
        ce, cv = t.loc[("ce", lab)], t.loc[("cvar", lab)]
        out[lab] = dict(entropic=-ce.value, gain=ce["diff"], gain_lo=ce.ci_lo, gain_hi=ce.ci_hi,   # gain > 0: less risk
                        cvar=cv.value, cvar_gain=cv["diff"], cvar_lo=cv.ci_lo, cvar_hi=cv.ci_hi,
                        mean=t.loc[("mean_pnl", lab)].value, std=t.loc[("std_pnl", lab)].value,
                        cost=t.loc[("mean_cost", lab)].value, trades=t.loc[("mean_trades", lab)].value)
    return out, block


# ---------------------------------------------------------------------------- MPC's band, probed
def _probe(job):
    """MPC's action for one state from position 0 (-> lower edge of today's no-trade band) or 1 (-> upper)."""
    entry, cost, left, lm, position = job
    tau, spot = left / 252, 100.0 * math.exp(lm)
    price, delta, gamma = black_scholes(spot, 100.0, tau, 0.2, 0.0)
    obs = Observation(step=21 - left, steps_left=left, tau=tau, spot=spot, strike=100.0, is_call=True, vol=0.2,
                      rate=0.0, cost_rate=cost / 1e4, option_price=float(price), delta=float(delta), gamma=float(gamma),
                      position=position, cash=float(price) - position * spot, pnl=0.0)
    mpc = STRATEGIES["mpc"](**entry["params"])
    mpc.reset()
    return min(max(mpc.act(obs), 0.0), 1.0)


def mpc_bands(entry, cost, workers):
    key = hashlib.sha1(json.dumps([entry, cost, LOG_M.tolist()], sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE / f"mpc_bands_{key}.npy"
    if path.exists():
        return np.load(path)
    jobs = [(entry, cost, left, lm, pos) for pos in (0.0, 1.0) for left in range(1, 22) for lm in LOG_M]
    with ProcessPoolExecutor(workers) as pool:
        out = np.array(list(pool.map(_probe, jobs, chunksize=16))).reshape(2, 21, len(LOG_M))
    np.save(path, out)
    return out


# ---------------------------------------------------------------------------- Black-Scholes world
def pick_examples(eps):
    """Three test paths with different shapes: a rally, a chop around the strike, a sell-off."""
    ret = np.array([e.spots[-1] / e.spots[0] - 1 for e in eps])
    crossings = np.array([int(np.sum(np.diff(np.sign(e.spots - 100.0)) != 0)) for e in eps])
    rally = int(np.flatnonzero((ret > 0.06) & (ret < 0.09))[0])
    chop = int(np.flatnonzero((np.abs(ret) < 0.01) & (crossings >= 6))[0])
    selloff = int(np.flatnonzero((ret < -0.06) & (ret > -0.09))[0])
    return [(rally, f"Rally ({ret[rally]:+.0%})"), (chop, f"Chop around the strike ({crossings[chop]} crossings)"),
            (selloff, f"Sell-off ({ret[selloff]:+.0%})")]


def gbm_world(workers):
    base = json.loads(Path("configs/gbm_bench.json").read_text())
    by_label = {s.get("label", s["name"]): s for s in base["strategies"]}
    test_eps, _ = load_episodes(base)
    n = base["option"]["maturity_steps"]
    examples = pick_examples(test_eps)
    out = dict(costs=COSTS, log_m=LOG_M.tolist(), n_steps=n, n_test=len(test_eps), bands={}, risk={}, dp_value={},
               examples=[dict(name=name, spot=r4(test_eps[i].spots),
                              delta=r4([black_scholes(s, 100.0, (n - t) / 252, 0.2, 0.0)[1]
                                        for t, s in enumerate(test_eps[i].spots[:-1])]))
                         for i, name in examples],
               traces={}, checks={})
    parity, pnl10 = 0.0, {}
    for c in COSTS:
        cfg = copy.deepcopy(base)
        cfg["market"]["cost_bps"] = c
        learned = copy.deepcopy(by_label["deep_band"])
        learned["params"]["model"] = f"models/site/c{c}/deep_band.pt"
        entries = {"delta": by_label["delta_daily"], "ww": by_label["ww"], "dp": by_label["dp"],
                   "mpc": by_label["mpc_band"], "learned": learned}
        parity = max(parity, ensure_trained(cfg, [learned]))
        eps = pd.concat([evaluate(cfg, e, workers).assign(strategy=k) for k, e in entries.items()])
        run = run_from({**cfg, "strategies": [{"name": k, "label": k} for k in entries], "eval": {"risk_aversion": 1.0}}, eps)
        risk, _ = metrics(run, "dp")
        out["risk"][str(c)] = risk
        if c == 10:
            pnl10 = {k: run["P"][:, i] for i, k in enumerate(run["labels"])}
        # the exact optimum and the bands
        sol = solve_gbm(100.0, True, n, 1 / 252, 0.2, 0.0, 0.0, c / 1e4, 1.0, h_grid(True, (0, 1), 401))
        out["dp_value"][str(c)] = optimal_risk(sol, 100.0, float(black_scholes(100.0, 100.0, n / 252, 0.2, 0.0)[0]))
        out["bands"][str(c)] = bands(sol, load_policy(learned["params"]["model"]), mpc_bands(entries["mpc"], c, workers),
                                     c / 1e4)
        env = make_env(cfg)
        strategies = {k: STRATEGIES[e["name"]](**e.get("params", {})) for k, e in entries.items()}
        out["traces"][str(c)] = [{k: trace(env, s, test_eps[i]) for k, s in strategies.items()} for i, _ in examples]
        print(f"[site] {c:2d} bps: " + "  ".join(f"{k} {v['entropic']:.4f}" for k, v in risk.items())
              + f"  (DP value {out['dp_value'][str(c)]:.4f})", flush=True)

    # verification numbers
    ck = out["checks"]
    prem = float(black_scholes(100.0, 100.0, n / 252, 0.2, 0.0)[0])
    ck["dp_refinement"] = [[nx, nh, optimal_risk(solve_gbm(100.0, True, n, 1 / 252, 0.2, 0.0, 0.0, 0.001, 1.0,
                                                           h_grid(True, (0, 1), nh), nx), 100.0, prem)]
                           for nx, nh in [(201, 101), (401, 201), (801, 401)]]
    dp = pnl10["dp"]
    boot = [np.log(np.mean(np.exp(-dp[i]))) for i in np.random.default_rng(0).integers(0, len(dp), (2000, len(dp)))]
    ck["dp_mc"] = dict(value=out["dp_value"]["10"], mc=out["risk"]["10"]["dp"]["entropic"],
                       lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))
    ck["gap_to_dp"] = {k: dict(gap=-v["gain"], lo=-v["gain_hi"], hi=-v["gain_lo"])
                       for k, v in out["risk"]["10"].items() if k != "dp"}
    ck["parity"] = parity
    tau, sigma = n / 252, 0.2
    vega = 100 * math.exp(-0.5 * (0.5 * sigma * math.sqrt(tau)) ** 2) / math.sqrt(2 * math.pi) * math.sqrt(tau)
    zero = evaluate({**base, "market": {**base["market"], "cost_bps": 0}}, by_label["delta_daily"], workers)
    ck["kamal_derman"] = dict(predicted=math.sqrt(math.pi / 4) * vega * sigma / math.sqrt(n),
                              observed=float(np.std(zero.pnl, ddof=1)))
    return out


def bands(sol, policy, probed, cost):
    """Position - delta at the band edges, for every steps-to-expiry and log-moneyness on the grid."""
    spot = 100.0 * np.exp(LOG_M)
    res = {k: [] for k in ("dp_lo", "dp_hi", "mpc_lo", "mpc_hi", "nn_lo", "nn_hi", "ww")}
    for left in range(1, 22):
        t, tau = 21 - left, left / 252
        _, delta, gamma = black_scholes(spot, 100.0, tau, 0.2, 0.0)
        feats = torch.tensor(np.stack([LOG_M, np.full_like(LOG_M, math.sqrt(tau)), np.full_like(LOG_M, 0.2), delta], -1),
                             dtype=DTYPE)
        with torch.no_grad():
            nlo, nhi = (x.clamp(0, 1).numpy() for x in policy.band(feats, torch.tensor(delta, dtype=DTYPE)))
        res["dp_lo"].append(r4(np.interp(np.log(spot), sol["x"], sol["lo"][t]) - delta))
        res["dp_hi"].append(r4(np.interp(np.log(spot), sol["x"], sol["hi"][t]) - delta))
        res["mpc_lo"].append(r4(probed[0, left - 1] - delta))
        res["mpc_hi"].append(r4(probed[1, left - 1] - delta))
        res["nn_lo"].append(r4(nlo - delta))
        res["nn_hi"].append(r4(nhi - delta))
        res["ww"].append(r4((1.5 * cost * spot * gamma**2 / 1.0) ** (1 / 3)))
    return res


# ---------------------------------------------------------------------------- real data
def real_world(key, ticker, pricing, workers):
    cfg = json.loads(Path(f"configs/{key}_bench.json").read_text())
    entries = [s for s in cfg["strategies"] if s["name"] != "no_hedge"]
    ensure_trained(cfg, entries)
    eps = pd.concat([evaluate(cfg, e, workers) for e in entries])
    run = run_from({**cfg, "strategies": entries}, eps)
    rows, block = metrics(run, "delta_daily")
    test, _ = load_episodes(cfg)
    df = load_market_data(cfg)
    train_df, test_df = split_train_test(df, cfg["split"])
    pnl = run["P"][:, run["labels"].index("delta_daily")]
    order = np.argsort(pnl)
    typical = [i for i in np.argsort(np.abs(pnl - np.median(pnl))) if i not in order[:5]][:2]
    env = make_env(cfg)
    by = {e["label"]: e for e in entries}
    strategies = {lab: STRATEGIES[by[lab]["name"]](**by[lab].get("params", {})) for lab in SHOWN}
    replays = []
    for i in list(order[:5]) + typical:
        ep = test[i]
        tr = {lab: trace(env, s, ep) for lab, s in strategies.items()}
        replays.append(dict(start=ep.start, kind="worst" if i in order[:5] else "typical", spot=r4(ep.spots),
                            vol=r4(ep.vols * 100), delta=r4([black_scholes(s, 100.0, (21 - t) / 252, v, cfg["market"]["rate"])[1]
                                                             for t, (s, v) in enumerate(zip(ep.spots[:-1], ep.vols[:-1]))]),
                            traces=tr))
    print(f"[site] {ticker}: " + "  ".join(f"{lab} {r['gain']:+.3f}" for lab, r in rows.items()), flush=True)
    return dict(key=key, ticker=ticker, pricing=pricing, n_test=run["n"], block=block,
                train_start=str(train_df.index[0].date()), test_start=str(test_df.index[0].date()),
                test_end=str(test_df.index[-1].date()), risk_aversion=cfg["eval"]["risk_aversion"],
                summary=[dict(label=lab, **{k: float(v) for k, v in rows[lab].items()}) for lab in run["labels"]],
                replays=replays)


def gradients():
    root = Path("results/gbm_bench")
    errors = pd.read_csv(root / "gradient_estimators.csv")
    curves = pd.read_csv(root / "gradient_learning_curves.csv")
    FIGS.mkdir(parents=True, exist_ok=True)
    (FIGS / "gradient_estimators.png").write_bytes((root / "gradient_estimators.png").read_bytes())
    return dict(errors=errors.to_dict("records"), curves=curves.round(5).to_dict("records"))


# ---------------------------------------------------------------------------- poster figures
def poster_figures(gbm, real):
    plt = _style()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5})

    # 1. the bands at 10 bps
    b = gbm["bands"]["10"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    for ax, left in zip(axes, [21, 10, 3]):
        k = left - 1
        ax.fill_between(LOG_M, b["dp_lo"][k], b["dp_hi"][k], color=COLOR["dp"], alpha=0.30, lw=0, label="DP (optimal)")
        for key, lo, hi in [("mpc", "mpc_lo", "mpc_hi"), ("learned", "nn_lo", "nn_hi")]:
            ax.plot(LOG_M, b[lo][k], color=COLOR[key], lw=2, label=NAME[key])
            ax.plot(LOG_M, b[hi][k], color=COLOR[key], lw=2)
        ax.plot(LOG_M, b["ww"][k], color=COLOR["ww"], lw=2, label="Whalley-Wilmott")
        ax.plot(LOG_M, -np.array(b["ww"][k]), color=COLOR["ww"], lw=2)
        ax.axhline(0, color=AXIS, lw=1, zorder=0)
        ax.set_title(f"{left} trading days to expiry")
        ax.set_xlabel("log(spot / strike)")
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("shares held − delta")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Where to stop trading (10 bps): MPC and the learned band follow the optimal band; Whalley-Wilmott's is too wide",
                 x=0.01, y=1.07, ha="left", va="bottom", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 1.02, va="bottom", s="MPC's band is read off its decisions (its trade from 0 shares and from 1 share), so it carries the "
             "sampling noise of a fresh plan over 1,024 random futures at every point.", ha="left", fontsize=10, color=INK2)
    fig.savefig(FIGS / "bands.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. gap to the optimum vs cost: everything, then a zoom on the two near-optimal methods
    fig, (ax, az) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    gap = lambda key, c: gbm["risk"][str(c)][key]["entropic"] - gbm["risk"][str(c)]["dp"]["entropic"]
    for key in ["delta", "ww", "mpc", "learned"]:
        ax.plot(COSTS, [gap(key, c) for c in COSTS], marker="o", ms=7, lw=2, color=COLOR[key], label=NAME[key],
                mec="white", mew=1.2)
    for a_ in (ax, az):
        a_.axhline(0, color=COLOR["dp"], lw=2, label="DP (the optimum)")
        a_.set_xticks(COSTS)
        a_.set_xlabel("trading cost (bps of each $ traded)")
        a_.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel("extra risk over the optimum (entropic, a = 1)")
    ax.set_title("All methods")
    ax.legend(fontsize=9.5)
    for key, dx in [("mpc", -0.18), ("learned", 0.18)]:
        r = [gbm["risk"][str(c)][key] for c in COSTS]
        y = np.array([-x["gain"] for x in r])
        err = np.array([[-x["gain"] - (-x["gain_hi"]) for x in r], [-x["gain_lo"] - (-x["gain"]) for x in r]])
        az.errorbar(np.array(COSTS) + dx, y, yerr=err, marker="o", ms=7, lw=2, capsize=0, elinewidth=2,
                    color=COLOR[key], label=NAME[key], mec="white", mew=1.2)
    az.set_ylim(bottom=-0.0005)
    az.set_title("Zoom: MPC and the learned band (95% CI)")
    az.legend(fontsize=9.5, loc="upper left")
    fig.suptitle(f"Out of sample on {gbm['n_test']:,} Black-Scholes paths: MPC (no training) and the learned band "
                 "stay within 0.006 of the optimum", x=0.01, y=1.03, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIGS / "risk_vs_cost.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. real data: gain over daily delta per asset, with 95% CIs
    labels = [lab for lab in REAL if lab != "delta_daily"]
    fig, axes = plt.subplots(1, len(real), figsize=(3.4 * len(real), 3.9), sharey=True)
    for ax, asset in zip(np.atleast_1d(axes), real):
        rows = {r["label"]: r for r in asset["summary"]}
        for y, lab in enumerate(labels[::-1]):
            r, color = rows[lab], COLOR[REAL[lab][0]]
            ax.plot([r["gain_lo"], r["gain_hi"]], [y, y], color=color, lw=2.2, solid_capstyle="round", alpha=0.55)
            ax.plot(r["gain"], y, "o", ms=8, color=color, mec="white", mew=1.2)
        ax.axvline(0, color=AXIS, lw=1.4, zorder=0)
        ax.set_yticks(range(len(labels)), [REAL[lab][1] for lab in labels[::-1]])
        ax.set_title(f"{asset['ticker']} ({asset['n_test']} options)")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_xlabel("gain over daily delta (> 0: less risk)")
    fig.suptitle(f"Unseen {real[0]['test_start'][:4]}–{real[0]['test_end'][:4]}: certainty-equivalent gain over daily "
                 f"delta, \\$ per \\$100 of stock (a = {real[0]['risk_aversion']:g}; 95% block-bootstrap CI)",
                 x=0.01, y=1.04, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.savefig(FIGS / "real_data.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--figures-only", action="store_true", help="redraw docs/figures/ from docs/data.js")
    args = parser.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        text = (DOCS / "data.js").read_text()
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        poster_figures(data["gbm"], data["real"])
        print(f"[site] redrew {FIGS}/ from {DOCS / 'data.js'}")
        return
    real = [real_world(k, t, p, args.workers) for k, t, p in ASSETS]
    gbm = gbm_world(args.workers)
    data = dict(gbm=gbm, real=real, gradients=gradients())
    # a script, not JSON: the page then also works when opened straight from disk (file://)
    (DOCS / "data.js").write_text("window.SITE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    poster_figures(gbm, real)
    print(f"[site] wrote {DOCS / 'data.js'} ({(DOCS / 'data.js').stat().st_size / 1e3:.0f} kB) and {FIGS}/")


if __name__ == "__main__":
    main()
