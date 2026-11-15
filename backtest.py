"""
Backtest option-hedging strategies on real (Yahoo Finance) or simulated price paths.

    uv run backtest.py --config configs/spy_vix.json

File layout (top to bottom):
  1. Black-Scholes pricing          -> option price, delta, gamma
  2. Market data                    -> a price series + a volatility series
  3. Episodes                       -> slice the series into one-option-lifetime chunks
  4. HedgingEnv                     -> step-by-step simulation of hedging one option (gym-like)
  5. Running, metrics, plots, CLI

Why the option is "synthetic": Yahoo only has TODAY's option prices, not historical ones.
So we take the real historical stock path and pretend we sold an option at each start date,
priced with Black-Scholes using a volatility estimate (VIX, trailing realized vol, or a
constant). The real path then decides how well the hedge actually worked.
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from strategy import Observation, Strategy
from strategies import STRATEGIES

CACHE_DIR = Path("data_cache")
_N = NormalDist()


# ---------------------------------------------------------------------------
# 1. Black-Scholes
# ---------------------------------------------------------------------------

def black_scholes(spot, strike, tau, vol, rate, kind="call"):
    """Price, delta and gamma of a European option.
    tau = years to maturity. At tau == 0 this returns the payoff (and a 0/1 delta)."""
    if tau <= 0 or vol <= 0:
        itm = spot > strike if kind == "call" else spot < strike
        payoff = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
        delta = (1.0 if kind == "call" else -1.0) if itm else 0.0
        return payoff, delta, 0.0
    sq = vol * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol**2) * tau) / sq
    d2 = d1 - sq
    disc = math.exp(-rate * tau)
    if kind == "call":
        price = spot * _N.cdf(d1) - strike * disc * _N.cdf(d2)
        delta = _N.cdf(d1)
    else:
        price = strike * disc * _N.cdf(-d2) - spot * _N.cdf(-d1)
        delta = _N.cdf(d1) - 1.0
    gamma = _N.pdf(d1) / (spot * sq)
    return price, delta, gamma


# ---------------------------------------------------------------------------
# 2. Market data: returns a DataFrame with columns [spot, vol]
# ---------------------------------------------------------------------------

def download_close(ticker, start, end, interval="1d") -> pd.Series:
    """Adjusted close prices from Yahoo Finance, cached to data_cache/ as CSV."""
    import yfinance as yf

    CACHE_DIR.mkdir(exist_ok=True)
    safe = ticker.replace("^", "").replace("/", "_")
    path = CACHE_DIR / f"{safe}_{start}_{end}_{interval}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Yahoo returned no data for {ticker!r} ({start} -> {end}, {interval})")
    close = df["Close"].squeeze().rename("close").dropna()
    close.to_csv(path)
    return close


def simulate_gbm(n_steps, s0=100.0, mu=0.0, sigma=0.2, seed=0, steps_per_year=252) -> pd.Series:
    """Geometric Brownian motion: the exact world Black-Scholes assumes. Good sanity check."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / steps_per_year
    logret = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n_steps)
    return pd.Series(s0 * np.exp(np.concatenate([[0.0], np.cumsum(logret)])), name="close")


def load_market_data(cfg) -> pd.DataFrame:
    data, vol = cfg["data"], cfg["vol"]
    spy = cfg["market"].get("steps_per_year", 252)

    if data["source"] == "yahoo":
        spot = download_close(data["ticker"], data.get("start"), data.get("end"), data.get("interval", "1d"))
    elif data["source"] == "gbm":
        spot = simulate_gbm(data["n_steps"], data.get("s0", 100.0), data.get("mu", 0.0),
                            data["sigma"], data.get("seed", 0), spy)
    else:
        raise ValueError(f"unknown data source {data['source']!r} (use 'yahoo' or 'gbm')")
    df = spot.to_frame("spot")

    # The volatility we PRICE and compute delta with. It is only an estimate of the
    # volatility the path will actually have -- that mismatch is part of the problem.
    if vol["type"] == "constant":
        df["vol"] = vol["value"]
    elif vol["type"] == "realized":   # trailing std of log returns, annualized (no look-ahead)
        logret = np.log(df["spot"]).diff()
        df["vol"] = logret.rolling(vol.get("window", 21)).std() * math.sqrt(spy)
    elif vol["type"] == "ticker":     # e.g. ^VIX = market's 30-day implied vol of the S&P 500, in %
        v = download_close(vol["ticker"], data.get("start"), data.get("end"), data.get("interval", "1d"))
        df["vol"] = v.reindex(df.index).ffill() * vol.get("scale", 0.01)
    else:
        raise ValueError(f"unknown vol type {vol['type']!r} (use 'constant', 'realized' or 'ticker')")
    return df.dropna()


def split_train_test(df: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: rows before `test_start` (a date, or a row number for gbm) are for
    training, the rest for testing. Episodes are cut separately on each side, so no option's
    life straddles the boundary. Vol is computed before splitting, which is fine: it only
    ever looks back."""
    start = split["test_start"]
    if isinstance(df.index, pd.DatetimeIndex):
        start = pd.Timestamp(start, tz=df.index.tz)
    is_test = df.index >= start
    if is_test.all() or not is_test.any():
        raise ValueError(f"test_start={split['test_start']!r} leaves one side of the split empty")
    return df[~is_test], df[is_test]


# ---------------------------------------------------------------------------
# 3. Episodes: one option's life = n_steps+1 consecutive prices
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    start: str            # label of the first timestamp (date for real data)
    spots: np.ndarray     # length n_steps+1, rescaled so spots[0] == 100
    vols: np.ndarray      # length n_steps+1


def make_episodes(df: pd.DataFrame, n_steps: int, stride: int) -> list[Episode]:
    """Start a new option every `stride` steps. Prices are rescaled to start at 100 so
    every episode is comparable (P&L is then in dollars per $100 of underlying)."""
    spot, vol = df["spot"].to_numpy(), df["vol"].to_numpy()
    episodes = []
    for i in range(0, len(df) - n_steps, stride):
        s = spot[i:i + n_steps + 1]
        label = df.index[i].strftime("%Y-%m-%d") if hasattr(df.index[i], "strftime") else str(df.index[i])
        episodes.append(Episode(label, 100.0 * s / s[0], vol[i:i + n_steps + 1]))
    if not episodes:
        raise ValueError(f"not enough data ({len(df)} rows) for options of {n_steps} steps")
    return episodes


# ---------------------------------------------------------------------------
# 4. The hedging environment
# ---------------------------------------------------------------------------

class HedgingEnv:
    """We are SHORT one option on one share. Each step the agent chooses a target number
    of shares to hold. Gym-style:  obs = env.reset(episode);  obs, reward, done, info = env.step(a)

    reward = change in mark-to-market P&L over the step (after costs). The rewards of an
    episode sum exactly to its final P&L, so an RL agent can use them directly or reshape.

    position_limits = (lo, hi) is a hard constraint on the shares held (a risk-desk mandate):
    a requested position outside it is clipped to it, and the step counts as a violation."""

    def __init__(self, option_type="call", moneyness=1.0, rate=0.0, cost_bps=0.0, steps_per_year=252,
                 position_limits=None):
        self.kind = option_type
        self.moneyness = moneyness            # strike / initial spot
        self.rate = rate
        self.cost_rate = cost_bps / 1e4        # cost per $ traded
        self.dt = 1.0 / steps_per_year
        self.limits = tuple(position_limits) if position_limits is not None else (-math.inf, math.inf)

    def reset(self, episode: Episode) -> Observation:
        self.ep = episode
        self.n_steps = len(episode.spots) - 1
        self.t = 0
        self.strike = self.moneyness * episode.spots[0]
        self.position = 0.0
        self.premium = self._price()[0]
        self.cash = self.premium               # we sold the option, so we receive its price
        self.pnl = 0.0                          # cash + shares - option value = 0 at the start
        self.total_cost = 0.0
        self.history = []
        return self._observe()

    def step(self, action: float):
        assert self.t < self.n_steps, "episode is over, call reset()"
        spot = self.ep.spots[self.t]

        # 1) trade from current position to the target position (clipped to the limits), paying costs
        requested = float(action)
        target = min(max(requested, self.limits[0]), self.limits[1])
        trade = target - self.position
        cost = self.cost_rate * abs(trade) * spot
        self.cash -= trade * spot + cost
        self.position = target
        self.total_cost += cost
        self.history.append(dict(step=self.t, spot=spot, vol=self.ep.vols[self.t], delta=self._price()[1],
                                 requested=requested, position=self.position, trade=trade, cost=cost, pnl=self.pnl))

        # 2) time passes: interest on cash, market moves
        self.t += 1
        self.cash *= math.exp(self.rate * self.dt)
        spot = self.ep.spots[self.t]
        done = self.t == self.n_steps

        # 3) at expiry: sell the remaining shares (with cost) -- the option pays its payoff
        if done:
            cost = self.cost_rate * abs(self.position) * spot
            self.cash += self.position * spot - cost
            self.total_cost += cost
            self.position = 0.0

        option_value = self._price()[0]       # equals the payoff when done
        new_pnl = self.cash + self.position * spot - option_value
        reward, self.pnl = new_pnl - self.pnl, new_pnl
        info = dict(cost=self.total_cost)
        if done:
            self.history.append(dict(step=self.t, spot=spot, vol=self.ep.vols[self.t], delta=self._price()[1],
                                     requested=0.0, position=0.0, trade=0.0, cost=cost, pnl=self.pnl))
        return self._observe(), reward, done, info

    def _price(self):
        tau = (self.n_steps - self.t) * self.dt
        return black_scholes(self.ep.spots[self.t], self.strike, tau, self.ep.vols[self.t], self.rate, self.kind)

    def _observe(self) -> Observation:
        price, delta, gamma = self._price()
        return Observation(
            step=self.t, steps_left=self.n_steps - self.t, tau=(self.n_steps - self.t) * self.dt,
            spot=self.ep.spots[self.t], strike=self.strike, vol=self.ep.vols[self.t], rate=self.rate,
            cost_rate=self.cost_rate, option_price=price, delta=delta, gamma=gamma,
            position=self.position, cash=self.cash, pnl=self.pnl,
        )


# ---------------------------------------------------------------------------
# 5. Running strategies, metrics, plots
# ---------------------------------------------------------------------------

def run_episode(env: HedgingEnv, strategy: Strategy, episode: Episode) -> dict:
    strategy.reset()
    obs, done = env.reset(episode), False
    while not done:
        obs, _, done, _ = env.step(strategy.act(obs))
    trades = np.array([h["trade"] for h in env.history])
    excess = np.array([abs(h["requested"] - h["position"]) for h in env.history])   # > 0: limit was hit
    return dict(start=episode.start, premium=env.premium, pnl=env.pnl, cost=env.total_cost,
                n_trades=int(np.sum(np.abs(trades) > 1e-12)), turnover=float(np.sum(np.abs(trades))),
                n_violations=int(np.sum(excess > 1e-9)), max_violation=float(excess.max()))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    def cvar(x, q=0.05):  # average of the worst 5% outcomes ("expected shortfall")
        x = np.sort(x.to_numpy())
        return x[: max(1, int(len(x) * q))].mean()

    g = results.groupby("strategy", sort=False)
    return pd.DataFrame({
        "mean_pnl": g["pnl"].mean(),
        "std_pnl": g["pnl"].std(),
        "cvar_5%": g["pnl"].apply(cvar),
        "worst": g["pnl"].min(),
        "mean_cost": g["cost"].mean(),
        "mean_trades": g["n_trades"].mean(),
        "violations": g["n_violations"].mean(),   # steps per episode where the position limit was hit
    })


# Categorical palette (fixed order; validated for colorblind-safe adjacent pairs).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def plot(results: pd.DataFrame, example: dict, labels: list[str], title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {lab: PALETTE[i] if i < len(PALETTE) else "#8a8984" for i, lab in enumerate(labels)}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b5b4ae", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig = plt.figure(figsize=(13, 6.5), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[1, 1.6], hspace=0.35, wspace=0.3)

    # (a) distribution of final P&L per strategy
    ax = fig.add_subplot(gs[:, 0])
    data = [results.loc[results.strategy == lab, "pnl"].to_numpy() for lab in labels][::-1]
    bp = ax.boxplot(data, orientation="horizontal", whis=(5, 95), showfliers=False, patch_artist=True,
                    widths=0.5, medianprops=dict(color="#0b0b0b", linewidth=1.5))
    for patch, lab in zip(bp["boxes"], labels[::-1]):
        patch.set(facecolor=colors[lab], alpha=0.85, edgecolor="none")
    ax.set_yticks(range(1, len(labels) + 1), labels[::-1])
    ax.axvline(0, color="#b5b4ae", linewidth=1, zorder=0)
    ax.set_xlabel(r"final P&L per option (\$, underlying starts at \$100)")
    ax.set_title("Final P&L across episodes\n(box = 25-75%, whiskers = 5-95%)", loc="left", fontsize=10)
    ax.grid(axis="x", color="#e7e6e2", linewidth=0.8)

    # (b) one example episode: the price path
    first = next(iter(example.values()))
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(first.step, first.spot, color="#52514e", linewidth=2)
    ax.axhline(first.attrs["strike"], color="#b5b4ae", linestyle="--", linewidth=1)
    ax.text(first.step.iloc[-1], first.attrs["strike"], " strike", va="center", color="#52514e", fontsize=9)
    ax.set_ylabel(r"spot (\$)")
    ax.set_title(f"Example episode starting {first.attrs['start']}", loc="left", fontsize=10)
    ax.grid(axis="y", color="#e7e6e2", linewidth=0.8)

    # (c) the same episode: hedge position chosen by each strategy vs Black-Scholes delta
    ax = fig.add_subplot(gs[1, 1], sharex=ax)
    ax.plot(first.step, first.delta, color="#0b0b0b", linestyle=":", linewidth=2, label="BS delta")
    for lab, hist in example.items():
        ax.step(hist.step, hist.position, where="post", color=colors[lab], linewidth=2, label=lab)
    ax.set_xlabel("step")
    ax.set_ylabel("shares held")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", color="#e7e6e2", linewidth=0.8)

    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a JSON config, e.g. configs/spy_vix.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    name = cfg.get("name", Path(args.config).stem)

    opt, mkt = cfg["option"], cfg["market"]
    df = load_market_data(cfg)
    if "split" in cfg:   # learned strategies trained on the early part: evaluate everyone on the rest
        train_df, df = split_train_test(df, cfg["split"])
        print(f"[{name}] out-of-sample evaluation: training rows end at {train_df.index[-1]}")
    n_steps = opt["maturity_steps"]
    episodes = make_episodes(df, n_steps, cfg.get("episodes", {}).get("stride", n_steps))
    env = HedgingEnv(opt.get("type", "call"), opt.get("moneyness", 1.0), mkt.get("rate", 0.0),
                     mkt.get("cost_bps", 0.0), mkt.get("steps_per_year", 252), mkt.get("position_limits"))
    print(f"[{name}] {len(df)} data rows ({df.index[0]} -> {df.index[-1]}), {len(episodes)} episodes "
          f"of {n_steps} steps, short 1 {opt.get('type', 'call')} at moneyness {opt.get('moneyness', 1.0)}")

    rows, example, labels = [], {}, []
    ex_idx = cfg.get("plot_episode", len(episodes) // 2)
    for s in cfg["strategies"]:
        label = s.get("label", s["name"])
        strategy = STRATEGIES[s["name"]](**s.get("params", {}))
        labels.append(label)
        for i, ep in enumerate(episodes):
            rows.append(dict(strategy=label, **run_episode(env, strategy, ep)))
            if i == ex_idx:
                hist = pd.DataFrame(env.history)
                hist.attrs.update(start=ep.start, strike=env.strike)
                example[label] = hist
    results = pd.DataFrame(rows)
    summary = summarize(results)

    print(f"mean premium received: {results.premium.mean():.3f}  (all P&L figures in $ per option on $100 of stock)\n")
    print(summary.round(3).to_string())

    out = Path(cfg.get("output_dir", "results")) / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.json")
    results.to_csv(out / "episodes.csv", index=False)
    summary.to_csv(out / "summary.csv")
    pd.concat(example, names=["strategy"]).to_csv(out / "example_episode.csv")
    plot(results, example, labels, f"{name}: {len(episodes)} episodes", out / "pnl.png")
    print(f"\nsaved results to {out}/")


if __name__ == "__main__":
    main()
