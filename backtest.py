"""
Backtest option-hedging strategies on real (Yahoo Finance) or simulated price paths.

    uv run backtest.py --config configs/spy_vix.json [--workers 10]

File layout (top to bottom; Black-Scholes pricing lives in strategy.py):
  1. Market data                    -> a price series + a volatility series
  2. Episodes                       -> slice the series into one-option-lifetime chunks
  3. HedgingEnv                     -> step-by-step simulation of hedging one option (gym-like)
  4. Running, metrics, CLI          (the figures are drawn by plots.py)

Why the option is "synthetic": Yahoo only has TODAY's option prices, not historical ones.
So we take the real historical stock path and pretend we sold an option at each start date,
priced with Black-Scholes using a volatility estimate (VIX, the stock's own implied vol, trailing realized vol, or a
constant). The real path then decides how well the hedge actually worked.
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from strategy import Observation, Strategy, black_scholes
from strategies import STRATEGIES

CACHE_DIR = Path("data_cache")


# ---------------------------------------------------------------------------
# 1. Market data: returns a DataFrame with columns [spot, vol]
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


def download_implied_vol(symbol) -> pd.Series:
    """Daily ~30-day at-the-money implied vol of one US stock (e.g. 0.21 = 21%), from the free
    community database DoltHub post-no-preference/options (table volatility_history, column
    iv_current; coverage starts 2019-02, not every trading day). Cached to data_cache/ as CSV;
    delete the file to refresh.

    One request per year, ~1 s apart. The table is keyed by (date, symbol), so we list every
    date of the year explicitly: that makes each row a direct lookup. A plain date range would
    scan every symbol on every date (~30 s per year, close to the server's timeout)."""
    import datetime
    import time
    import urllib.parse
    import urllib.request

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"dolthub_iv_{symbol}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)["iv"]
    url = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master?q="
    rows = []
    for year in range(2019, datetime.date.today().year + 1):
        dates = ",".join(f"'{d.date()}'" for d in pd.date_range(f"{year}-01-01", f"{year}-12-31"))
        sql = f"SELECT date, iv_current FROM volatility_history WHERE act_symbol = '{symbol}' AND date IN ({dates})"
        req = urllib.request.Request(url + urllib.parse.quote(sql), headers={"User-Agent": "options-hedging-backtester"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.load(resp)
        if out.get("query_execution_status") != "Success":
            raise RuntimeError(f"DoltHub query failed: {out.get('query_execution_message')}")
        rows += out.get("rows") or []
        time.sleep(1)
    if not rows:
        raise ValueError(f"DoltHub has no implied vol for {symbol!r}")
    iv = pd.Series({pd.Timestamp(r["date"]): r.get("iv_current") for r in rows}, name="iv")
    iv = pd.to_numeric(iv, errors="coerce").dropna().sort_index()   # some days have no value
    iv.to_csv(path)
    return iv


def simulate_gbm(n_steps, s0=100.0, mu=0.0, sigma=0.2, seed=0, steps_per_year=252) -> pd.Series:
    """Geometric Brownian motion: the exact world Black-Scholes assumes. Good sanity check."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / steps_per_year
    logret = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rng.standard_normal(n_steps)
    return pd.Series(s0 * np.exp(np.concatenate([[0.0], np.cumsum(logret)])), name="close")


def simulate_gjr_garch(n_paths, n_steps, mu, omega, alpha, gamma, beta, h0, s0=100.0, seed=0) -> np.ndarray:
    """Independent GJR-GARCH(1,1) price paths, shape (n_paths, n_steps+1). Per step (e.g. monthly):
        log-return  Y_t = mu + eps_t,   eps_t = sqrt(h_t) z_t,   z_t ~ N(0, 1)
        variance    h_{t+1} = omega + (alpha + gamma * 1{eps_t < 0}) eps_t^2 + beta h_t,   h_0 = h0
    Unlike GBM, volatility moves: it stays high after big moves (clustering) and rises more after
    drops than after rallies (gamma: the "leverage effect"). The setting of Neagu et al. (2025)."""
    rng = np.random.default_rng(seed)
    log_s = np.zeros((n_paths, n_steps + 1))
    h = np.full(n_paths, float(h0))
    for t in range(n_steps):
        eps = np.sqrt(h) * rng.standard_normal(n_paths)
        log_s[:, t + 1] = log_s[:, t] + mu + eps
        h = omega + (alpha + gamma * (eps < 0)) * eps**2 + beta * h
    return s0 * np.exp(log_s)


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
    elif vol["type"] == "implied":    # the market's own implied vol for this stock (Level A)
        iv = download_implied_vol(vol.get("symbol", data["ticker"]))
        # The data skips some days (and early dates fall on weekends), so carry the last known
        # value forward -- but only for up to max_stale_days calendar days. Longer gaps stay
        # NaN, and make_episodes skips any option whose life touches one.
        days = iv.index.union(df.index)
        last_seen = pd.Series(iv.index, index=iv.index).reindex(days).ffill()
        fresh = (days - pd.DatetimeIndex(last_seen)).days <= vol.get("max_stale_days", 7)
        df["vol"] = iv.reindex(days).ffill()[fresh].reindex(df.index)
    else:
        raise ValueError(f"unknown vol type {vol['type']!r} (use 'constant', 'realized', 'ticker' or 'implied')")
    return df.loc[df["vol"].first_valid_index():]   # drop the warm-up rows before the first vol value


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
# 2. Episodes: one option's life = n_steps+1 consecutive prices
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
        if np.isnan(vol[i:i + n_steps + 1]).any():   # vol data missing somewhere in this option's life
            continue
        label = df.index[i].strftime("%Y-%m-%d") if hasattr(df.index[i], "strftime") else str(df.index[i])
        episodes.append(Episode(label, 100.0 * s / s[0], vol[i:i + n_steps + 1]))
    if not episodes:
        raise ValueError(f"not enough data ({len(df)} rows) for options of {n_steps} steps")
    return episodes


def load_episodes(cfg) -> tuple[list[Episode], str]:
    """Episodes plus a one-line description of where they came from. Two kinds of data:
      - one long price series (yahoo, gbm): sliced into option lifetimes by make_episodes; with a
        "split" in the config, only the test part (learned strategies train on the part before)
      - a set of independent simulated paths (gjr_garch): each path is one episode"""
    data, n_steps = cfg["data"], cfg["option"]["maturity_steps"]
    if data["source"] == "gjr_garch":
        if cfg["vol"]["type"] != "constant" or "split" in cfg:
            raise ValueError("gjr_garch paths only support a constant pricing vol and no train/test split")
        paths = simulate_gjr_garch(data["n_paths"], n_steps, data["mu"], data["omega"], data["alpha"],
                                   data["gamma"], data["beta"], data["h0"], seed=data.get("seed", 0))
        vols = np.full(n_steps + 1, float(cfg["vol"]["value"]))
        return [Episode(f"path_{i}", s, vols) for i, s in enumerate(paths)], f"{len(paths)} simulated GJR-GARCH paths"
    df, note = load_market_data(cfg), ""
    if "split" in cfg:   # learned strategies trained on the early part: evaluate everyone on the rest
        train_df, df = split_train_test(df, cfg["split"])
        note = f", out of sample (training rows end at {train_df.index[-1]})"
    episodes = make_episodes(df, n_steps, cfg.get("episodes", {}).get("stride", n_steps))
    return episodes, f"{len(df)} data rows ({df.index[0]} -> {df.index[-1]}){note}"


# ---------------------------------------------------------------------------
# 3. The hedging environment
# ---------------------------------------------------------------------------

class HedgingEnv:
    """We are SHORT one option on one share. Each step the agent chooses a target number
    of shares to hold. Gym-style:  obs = env.reset(episode);  obs, reward, done, info = env.step(a)

    reward = change in mark-to-market P&L over the step (after costs). The rewards of an
    episode sum exactly to its final P&L minus the day-one markup (0 unless vol_markup is set),
    so an RL agent can use them directly or reshape.

    position_limits = (lo, hi) is a hard constraint on the shares held (a risk-desk mandate):
    a requested position outside it is clipped to it, and the step counts as a violation.

    vol_markup: we charge the buyer more than the fair price, as a market maker would, by pricing
    the premium at vol + vol_markup (0.02 = 2 vol points), or "leland" for Leland's (1985) cost-
    covering vol (see _markup). The liability is still marked at the fair vol, so the extra premium
    shows up as P&L the moment we sell. Every strategy's final P&L shifts up by the same amount."""

    def __init__(self, option_type="call", moneyness=1.0, rate=0.0, cost_bps=0.0, steps_per_year=252,
                 position_limits=None, vol_markup=0.0):
        self.kind = option_type
        self.moneyness = moneyness            # strike / initial spot
        self.rate = rate
        self.cost_rate = cost_bps / 1e4        # cost per $ traded
        self.dt = 1.0 / steps_per_year
        self.limits = tuple(position_limits) if position_limits is not None else (-math.inf, math.inf)
        self.vol_markup = vol_markup

    def reset(self, episode: Episode) -> Observation:
        self.ep = episode
        self.n_steps = len(episode.spots) - 1
        self.t = 0
        self.strike = self.moneyness * episode.spots[0]
        self.position = 0.0
        fair = float(self._price()[0])
        self.premium = float(self._price(self._markup())[0])   # what the buyer pays us
        self.cash = self.premium               # we sold the option, so we receive its price
        self.pnl = self.premium - fair          # cash + shares - option value: 0 unless we charged a markup
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

    def _price(self, extra_vol=0.0):
        tau = (self.n_steps - self.t) * self.dt
        return black_scholes(self.ep.spots[self.t], self.strike, tau, self.ep.vols[self.t] + extra_vol, self.rate, self.kind)

    def _markup(self) -> float:
        """Vol points added to the fair vol when pricing the premium. "leland": Leland (1985),
        sigma_L = sigma * sqrt(1 + sqrt(2/pi) * k / (sigma * sqrt(dt))) with k = 2 * cost_rate (a round trip),
        the vol at which the premium pays, on average, for rebalancing a delta hedge every step."""
        if self.vol_markup != "leland":
            return float(self.vol_markup)
        sig = self.ep.vols[0]
        return sig * (math.sqrt(1 + math.sqrt(2 / math.pi) * 2 * self.cost_rate / (sig * math.sqrt(self.dt))) - 1)

    def _observe(self) -> Observation:
        price, delta, gamma = self._price()
        return Observation(
            step=self.t, steps_left=self.n_steps - self.t, tau=(self.n_steps - self.t) * self.dt,
            spot=self.ep.spots[self.t], strike=self.strike, is_call=self.kind == "call",
            vol=self.ep.vols[self.t], rate=self.rate, cost_rate=self.cost_rate,
            option_price=float(price), delta=float(delta), gamma=float(gamma),
            position=self.position, cash=self.cash, pnl=self.pnl,
        )


# ---------------------------------------------------------------------------
# 4. Running strategies, metrics, CLI
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


def rsqp(pnl: np.ndarray) -> float:
    """Root semi-quadratic penalty sqrt(E[loss^2 * 1{loss > 0}]), loss = -P&L. Like an RMS of the
    P&L, but only losses count: ending above zero is never penalized. Used by Neagu et al. (2025)."""
    return float(np.sqrt(np.mean(np.maximum(-pnl, 0.0) ** 2)))


def certainty_equivalent(pnl: np.ndarray, risk_aversion: float) -> float:
    """CE = -log E[exp(-g * P&L)] / g: the sure amount you'd value the risky P&L at under
    exponential utility. ~ mean - (g/2) * variance for small risk; higher is better."""
    z = -risk_aversion * np.asarray(pnl)
    return float(-(z.max() + np.log(np.mean(np.exp(z - z.max())))) / risk_aversion)


def summarize(results: pd.DataFrame, n_splits: int = 1, risk_aversion: float | None = None) -> pd.DataFrame:
    """One row of metrics per strategy. With n_splits > 1 the episodes are cut into that many
    consecutive groups ("test sets"); rsqp is then the average over groups and rsqp_sd its spread.
    With a risk_aversion g, also the certainty equivalent ce (see certainty_equivalent)."""
    def cvar(x, q=0.05):  # average of the worst 5% outcomes ("expected shortfall")
        x = np.sort(x.to_numpy())
        return x[: max(1, int(len(x) * q))].mean()

    g = results.groupby("strategy", sort=False)
    split_rsqp = g["pnl"].apply(lambda x: [rsqp(s) for s in np.array_split(x.to_numpy(), n_splits)])
    out = pd.DataFrame({
        "mean_pnl": g["pnl"].mean(),
        "std_pnl": g["pnl"].std(),
        "cvar_5%": g["pnl"].apply(cvar),
        "worst": g["pnl"].min(),
        "rsqp": split_rsqp.apply(np.mean),
        "mean_cost": g["cost"].mean(),
        "mean_trades": g["n_trades"].mean(),
    })
    if results["n_violations"].any():   # steps per episode where a position limit was hit
        out["violations"] = g["n_violations"].mean()
    if n_splits > 1:
        out.insert(out.columns.get_loc("rsqp") + 1, "rsqp_sd", split_rsqp.apply(np.std))
    if risk_aversion:
        out.insert(out.columns.get_loc("mean_cost"), "ce", g["pnl"].apply(lambda x: certainty_equivalent(x, risk_aversion)))
    return out


def make_env(cfg) -> HedgingEnv:
    opt, mkt = cfg["option"], cfg["market"]
    return HedgingEnv(opt.get("type", "call"), opt.get("moneyness", 1.0), mkt.get("rate", 0.0),
                      mkt.get("cost_bps", 0.0), mkt.get("steps_per_year", 252), mkt.get("position_limits"),
                      opt.get("vol_markup", 0.0))


def _run_chunk(job):
    """Worker for --workers: one strategy on episodes [lo, hi). Rebuilds the episodes from the config
    (cached / seeded, so identical in every process)."""
    cfg, s, lo, hi = job
    env = make_env(cfg)
    strategy = STRATEGIES[s["name"]](**s.get("params", {}))
    label = s.get("label", s["name"])
    return [dict(strategy=label, **run_episode(env, strategy, ep)) for ep in load_episodes(cfg)[0][lo:hi]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a JSON config, e.g. configs/spy_vix.json")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes to run (strategy x chunk of episodes) in parallel; 1 = sequential")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    name = cfg.get("name", Path(args.config).stem)

    opt = cfg["option"]
    n_steps = opt["maturity_steps"]
    episodes, source = load_episodes(cfg)
    env = make_env(cfg)
    print(f"[{name}] {source}, {len(episodes)} episodes "
          f"of {n_steps} steps, short 1 {opt.get('type', 'call')} at moneyness {opt.get('moneyness', 1.0)}")

    rows, example = [], {}
    ex_idx = cfg.get("plot_episode", len(episodes) // 2)
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        cuts = np.linspace(0, len(episodes), args.workers + 1).astype(int)
        jobs = [(cfg, s, cuts[i], cuts[i + 1]) for s in cfg["strategies"] for i in range(args.workers) if cuts[i + 1] > cuts[i]]
        with ProcessPoolExecutor(args.workers) as pool:
            for chunk in pool.map(_run_chunk, jobs):     # map keeps job order = episode order per strategy
                rows += chunk
    for s in cfg["strategies"]:
        label = s.get("label", s["name"])
        strategy = STRATEGIES[s["name"]](**s.get("params", {}))
        todo = enumerate(episodes) if args.workers <= 1 else [(ex_idx, episodes[ex_idx])]
        for i, ep in todo:
            r = dict(strategy=label, **run_episode(env, strategy, ep))
            if args.workers <= 1:
                rows.append(r)
            if i == ex_idx:
                example[label] = pd.DataFrame(env.history)
    results = pd.DataFrame(rows)
    ev = cfg.get("eval", {})
    summary = summarize(results, ev.get("n_splits", 1), ev.get("risk_aversion"))

    print(f"mean premium received: {results.premium.mean():.3f}  (all P&L figures in $ per option on $100 of stock)\n")
    print(summary.round(3).to_string())

    out = Path(cfg.get("output_dir", "results")) / name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.json")
    results.to_csv(out / "episodes.csv", index=False)
    summary.to_csv(out / "summary.csv")
    pd.concat(example, names=["strategy"]).to_csv(out / "example_episode.csv")
    from plots import render
    render(out)                                   # metrics.png, example.png, comparison.csv
    print(f"\nsaved results to {out}/")


if __name__ == "__main__":
    main()
