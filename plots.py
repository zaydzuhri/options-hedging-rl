"""Figures for one backtest run, drawn from its results folder, so a finished run can be re-drawn
without re-running it. backtest.py calls render() at the end of every run. To re-draw old runs:

    uv run plots.py results/gbm_costs results/ko_costs ...

Writes into the folder:
    metrics.png      final P&L across episodes, then every metric per strategy as a box plot
    example.png      one episode: the price, and each strategy's shares next to Black-Scholes delta
    comparison.csv   the numbers behind metrics.png

The baseline is eval.baseline in the config (a strategy label); by default the Whalley-Wilmott band if the
run has one, else delta hedging, else the first strategy.

A metric like the mean or RSQP is ONE number per strategy, so its box shows how much that number would
move with a different draw of episodes: a paired bootstrap that resamples episodes (the same ones for
every strategy) and recomputes each strategy's metric minus the baseline's. Overlapping episodes on real
data share price moves, so there it resamples blocks of consecutive episodes (a "block bootstrap").
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Colours: one per method FAMILY (strategies are told apart by their labels); baselines in gray. The first
# three categorical slots are validated colourblind-safe for every pair; light chart surface only.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY = {"mpc": 0, "dp": 1, "dp_optimal": 1, "deep_hedge": 2}
BASELINES = {"no_hedge", "delta_hedge", "delta_band", "ww_band", "whalley_wilmott"}

# key, title, unit, which way is better
METRICS = [
    ("mean_pnl", "Mean P&L", r"\$", "higher"),
    ("std_pnl", "Std of P&L", r"\$", "lower"),
    ("cvar", "Worst-5% average, CVaR", r"\$", "higher"),
    ("ce", "Certainty equivalent, g = {g:g}", r"\$", "higher"),
    ("rsqp", "RSQP", r"\$", "lower"),
    ("mean_cost", "Trading cost", r"\$", "lower"),
    ("mean_trades", "Trades per episode", "", None),
]


# ---------------------------------------------------------------------------- data
def load_run(folder):
    folder = Path(folder)
    cfg = json.loads((folder / "config.json").read_text())
    eps = pd.read_csv(folder / "episodes.csv")
    labels = [s.get("label", s["name"]) for s in cfg["strategies"]]
    labels = [lab for lab in labels if lab in set(eps.strategy)]
    by = {lab: eps[eps.strategy == lab].reset_index(drop=True) for lab in labels}
    n = len(by[labels[0]])
    assert all(len(d) == n for d in by.values()), "strategies were run on different episodes"
    mat = lambda col: np.column_stack([by[lab][col].to_numpy(float) for lab in labels])
    return dict(folder=folder, cfg=cfg, labels=labels, by=by, n=n,
                P=mat("pnl"), C=mat("cost"), T=mat("n_trades"), premium=float(eps.premium.mean()))


def family_colors(cfg):
    out, spare, extra = {}, iter(range(len(FAMILY), len(SLOTS))), {}
    for s in cfg["strategies"]:
        lab, name = s.get("label", s["name"]), s["name"]
        if name in BASELINES:
            out[lab] = MUTED
        elif name in FAMILY:
            out[lab] = SLOTS[FAMILY[name]]
        else:                                    # a new family (e.g. an RL agent): the next free slot
            if name not in extra:
                extra[name] = next(spare, None)
            out[lab] = INK2 if extra[name] is None else SLOTS[extra[name]]
    return out


def baseline(cfg, labels):
    want = cfg.get("eval", {}).get("baseline")
    if want in labels:
        return want
    for name in ("ww_band", "whalley_wilmott", "delta_hedge"):
        for s in cfg["strategies"]:
            if s["name"] == name and s.get("label", s["name"]) in labels:
                return s.get("label", s["name"])
    return labels[0]


def block_size(cfg):
    """Episodes that share price moves: real data with stride < maturity. Resample blocks of twice the overlap."""
    if cfg["data"].get("source") == "gjr_garch":
        return 1
    n_steps = cfg["option"]["maturity_steps"]
    overlap = n_steps / cfg.get("episodes", {}).get("stride", n_steps)
    return 1 if overlap <= 1 else 2 * math.ceil(overlap)


# ---------------------------------------------------------------------------- metrics
def metric_values(P, C, T, idx, g, n_splits):
    """Every metric for every strategy (columns), on the episodes idx. Same definitions as
    backtest.summarize (RSQP averaged over n_splits consecutive groups when n_splits > 1)."""
    X = P[idx]
    k = max(1, int(len(X) * 0.05))
    loss2 = np.maximum(-X, 0.0) ** 2
    out = {
        "mean_pnl": X.mean(0),
        "std_pnl": X.std(0, ddof=1),
        "cvar": np.partition(X, k - 1, axis=0)[:k].mean(0),
        "rsqp": np.mean([np.sqrt(part.mean(0)) for part in np.array_split(loss2, n_splits)], axis=0),
        "mean_cost": C[idx].mean(0),
        "mean_trades": T[idx].mean(0),
    }
    if g:
        z = -g * X
        m = z.max(0)
        out["ce"] = -(m + np.log(np.exp(z - m).mean(0))) / g
    return out


def compare(run, base, n_boot=1000, seed=0):
    """Point value of each metric, and quantiles (2.5, 25, 75, 97.5%) of its paired (block) bootstrap
    difference from base: diff, ci_lo, q25, q75, ci_hi are all relative to base."""
    ev = run["cfg"].get("eval", {})
    g, n_splits, block = ev.get("risk_aversion"), ev.get("n_splits", 1), block_size(run["cfg"])
    P, C, T, n, j = run["P"], run["C"], run["T"], run["n"], run["labels"].index(base)
    point = metric_values(P, C, T, np.arange(n), g, n_splits)
    rng, draws = np.random.default_rng(seed), {k: [] for k in point}
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, math.ceil(n / block))
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        for k, v in metric_values(P, C, T, idx, g, n_splits).items():
            draws[k].append(v - v[j])
    rows = []
    for k, v in point.items():
        lo, q1, q3, hi = np.percentile(np.array(draws[k]), [2.5, 25, 75, 97.5], axis=0)
        for i, lab in enumerate(run["labels"]):
            rows.append(dict(metric=k, strategy=lab, value=v[i], diff=v[i] - v[j],
                             ci_lo=lo[i], q25=q1[i], q75=q3[i], ci_hi=hi[i]))
    return pd.DataFrame(rows), block


# ---------------------------------------------------------------------------- drawing
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": next((f for f in ("Helvetica Neue", "Arial", "Liberation Sans") if f in have), "DejaVu Sans"),
        "font.size": 9.5, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelcolor": INK2, "xtick.major.size": 0,
        "ytick.major.size": 0, "axes.grid": False, "legend.frameon": False,
    })
    return plt


def _title(fig, run):
    cfg = run["cfg"]
    bps = cfg["market"].get("cost_bps", 0)
    head = cfg.get("name", run["folder"].name)
    mk = cfg["option"].get("vol_markup", 0)
    mk = " (Leland markup)" if mk == "leland" else f" (fair vol + {100 * mk:g} vol pts)" if mk else ""
    sub = (f"{run['n']} episodes · short {cfg['option'].get('type', 'call')}, {cfg['option']['maturity_steps']} steps · "
           f"trading cost {bps:g} bps · mean premium \\${run['premium']:.2f}{mk} per option on \\$100 of stock")
    fig.text(0.01, 0.995, head, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.995 - 0.32 / fig.get_figheight(), sub, ha="left", va="top", fontsize=9.5, color=INK2)


def _box(a, y, lo, q1, q3, hi, color, h=0.46):
    """One horizontal box: whiskers lo..hi, box q1..q3."""
    from matplotlib.patches import Rectangle
    a.plot([lo, q1], [y, y], color=color, lw=1.4, solid_capstyle="butt", zorder=2)
    a.plot([q3, hi], [y, y], color=color, lw=1.4, solid_capstyle="butt", zorder=2)
    a.add_patch(Rectangle((q1, y - h / 2), q3 - q1, h, facecolor=color, alpha=0.3, edgecolor=color, lw=1.2, zorder=2))


def plot_metrics(run, table, base, block, path):
    plt = _style()
    # no_hedge is off the scale of every hedged strategy (std ~6x), so it would flatten each panel
    hidden = [s.get("label", s["name"]) for s in run["cfg"]["strategies"] if s["name"] == "no_hedge" and
              s.get("label", s["name"]) in run["labels"] and s.get("label", s["name"]) != base]
    keep = [i for i, lab in enumerate(run["labels"]) if lab not in hidden]
    labels, colors, S = [run["labels"][i] for i in keep], family_colors(run["cfg"]), len(keep)
    g = run["cfg"].get("eval", {}).get("risk_aversion")
    metrics = [m for m in METRICS if m[0] in set(table.metric)]
    ncol = 3
    nrow = math.ceil((len(metrics) + 2) / ncol)                      # + the P&L panel + the note
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, nrow * max(0.27 * S + 1.15, 2.7) + 0.7), squeeze=False)
    panels, y = list(axes.flat), np.arange(S)[::-1]

    def frame(a, title, sub):
        for yy in y:
            a.axhline(yy, color=GRID, lw=0.6, zorder=0)
        a.set_yticks(y, labels if a in axes[:, 0] else [""] * S)
        for tl, lab in zip(a.get_yticklabels(), labels):
            tl.set_fontweight("bold" if lab == base else "normal")
        a.set_ylim(-0.7, S - 0.3)
        a.spines["left"].set_visible(False)
        a.xaxis.set_major_locator(plt.MaxNLocator(4))
        a.set_title(title, pad=15)
        a.text(0, 1.02, sub, transform=a.transAxes, ha="left", va="bottom", fontsize=8.5, color=MUTED)

    # first panel: the spread of outcomes across episodes (a classic box plot)
    a = panels[0]
    for yy, lab, x in zip(y, labels, run["P"][:, keep].T):
        q = np.percentile(x, [5, 25, 50, 75, 95])
        _box(a, yy, q[0], q[1], q[3], q[4], colors[lab])
        a.plot([q[2], q[2]], [yy - 0.23, yy + 0.23], color=INK, lw=1.6, zorder=3)
    a.axvline(0, color=AXIS, lw=1.4, zorder=1)
    frame(a, r"Final P&L across episodes (\$)", "box: middle 50%, whiskers: 5% to 95%, line: median")

    # every metric: the value (dot), and how much it moves under resampling, paired with the baseline
    for a, (key, title, unit, better) in zip(panels[1:], metrics):
        t = table[table.metric == key].set_index("strategy").loc[labels]
        b = t.loc[base, "value"]
        for yy, lab in zip(y, labels):
            r = t.loc[lab]
            if lab != base:
                _box(a, yy, b + r.ci_lo, b + r.q25, b + r.q75, b + r.ci_hi, colors[lab])
            a.plot(r.value, yy, "o", ms=6.5, color=colors[lab], mec=SURFACE, mew=1.2, zorder=3)
        a.axvline(b, color=AXIS, lw=1.4, zorder=1)
        frame(a, title.format(g=g) + (f" ({unit})" if unit else ""), f"{better} is better" if better else "")

    for a in panels[len(metrics) + 1:]:
        a.axis("off")
    note = panels[len(metrics) + 1]
    how = "paired bootstrap" if block == 1 else f"paired block bootstrap,\nblocks of {block} overlapping episodes"
    note.text(0.0, 1.0, "How to read", transform=note.transAxes, ha="left", va="top", fontsize=10,
              fontweight="bold", color=INK)
    note.annotate("First panel: how outcomes spread across episodes.\n\n"
                  "Other panels: one number per strategy (dot); its\n"
                  "box shows how much it would move with other episodes\n"
                  f"(box: middle 50%, whiskers: 95%; {how}).\n"
                  f"Gray line: the baseline, {base}. Whiskers that\n"
                  "don't reach it: the difference isn't luck.\n"
                  "Color = method family; gray = baselines."
                  + (f"\n{', '.join(hidden)}: not drawn (off scale), see summary.csv." if hidden else ""),
                  xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, -20), textcoords="offset points",
                  ha="left", va="top", fontsize=9, color=INK2, linespacing=1.45)
    _title(fig, run)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig.get_figheight()), h_pad=2.2, w_pad=1.5)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_example(run, path):
    folder, cfg = run["folder"], run["cfg"]
    ex = pd.read_csv(folder / "example_episode.csv")
    plt = _style()
    labels = [lab for lab in run["labels"] if lab in set(ex.strategy)]
    colors = family_colors(cfg)
    i_ex = cfg.get("plot_episode", run["n"] // 2)
    first = ex[ex.strategy == labels[0]]
    strike = cfg["option"].get("moneyness", 1.0) * first.spot.iloc[0]
    ncol = 3
    nrow = math.ceil(len(labels) / ncol)
    fig = plt.figure(figsize=(13, 2.3 + 1.75 * nrow))
    gs = fig.add_gridspec(1 + nrow, ncol, height_ratios=[1.25] + [1] * nrow, hspace=0.55, wspace=0.12)
    top = fig.add_subplot(gs[0, :])
    top.plot(first.step, first.spot, color=INK2, lw=2)
    top.axhline(strike, color=AXIS, lw=1)
    top.text(first.step.iloc[-1], strike, "  strike", va="center", color=MUTED, fontsize=9)
    top.set_ylabel(r"price (\$)")
    start = run["by"][labels[0]].start.iloc[i_ex]
    top.set_title(f"Price path, episode {i_ex}" + (f" (starts {start})" if cfg["data"].get("source") == "yahoo" else ""))
    top.spines["left"].set_visible(False)
    top.grid(axis="y", color=GRID, lw=0.6)
    lo, hi = min(ex.position.min(), ex.delta.min()), max(ex.position.max(), ex.delta.max())
    for k, lab in enumerate(labels):
        a = fig.add_subplot(gs[1 + k // ncol, k % ncol], sharex=top)
        h = ex[ex.strategy == lab]
        a.plot(h.step, h.delta, color=AXIS, lw=2.2, label="Black-Scholes delta")
        a.step(h.step, h.position, where="post", color=colors[lab], lw=2, label="shares held")
        r = run["by"][lab].iloc[i_ex]
        a.set_title(lab, fontsize=9.5)
        a.text(1.0, 1.02, f"P&L {r.pnl:+.2f}  ·  {int(r.n_trades)} trades", transform=a.transAxes,
               ha="right", va="bottom", fontsize=8.5, color=INK2)
        a.set_ylim(lo - 0.05, hi + 0.05)
        a.grid(axis="y", color=GRID, lw=0.6)
        a.spines["left"].set_visible(False)
        if k % ncol:
            a.tick_params(labelleft=False)
        if k // ncol < nrow - 1 and k + ncol < len(labels):
            a.tick_params(labelbottom=False)
        else:
            a.set_xlabel("step")
    _title(fig, run)
    fig.text(0.99, 0.995 - 0.32 / fig.get_figheight(), "gray line: Black-Scholes delta    colored steps: shares held",
             ha="right", va="top", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.99, top=1 - 0.95 / fig.get_figheight(), bottom=0.45 / fig.get_figheight())
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------- entry points
def render(folder):
    """Draw all figures (and comparison.csv) for one results folder."""
    run = load_run(folder)
    base = baseline(run["cfg"], run["labels"])
    table, block = compare(run, base)
    table.to_csv(run["folder"] / "comparison.csv", index=False)
    plot_metrics(run, table, base, block, run["folder"] / "metrics.png")
    if (run["folder"] / "example_episode.csv").exists():
        plot_example(run, run["folder"] / "example.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for f in sys.argv[1:]:
        render(f)
        print(f"drew {f}/")
