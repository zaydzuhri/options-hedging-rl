# options-hedging-rl: hedging backtester

```bash
uv sync
uv run backtest.py --config configs/spy_vix.json
```
Outputs go to `results/<name>/`: `summary.csv`, `episodes.csv` (one row per strategy × episode), `example_episode.csv` (step-by-step trajectory), `pnl.png`, and a copy of the config.

## Files
| file | what it is |
|---|---|
| `backtest.py` | everything that isn't a strategy: data loading, episodes, `HedgingEnv`, metrics, plot, CLI |
| `strategy.py` | the interface: `Observation` (what a strategy sees) and `Strategy` (`reset()`, `act(obs) -> shares`), plus the `black_scholes` formula |
| `strategies/` | concrete strategies + the name → class registry in `__init__.py` (each file can be deleted on its own): |
| ↳ `baselines.py` | `no_hedge`, `delta_hedge`, `delta_band`, `ww_band` (Whalley–Wilmott no-trade band) |
| ↳ `dp.py` | `dp`: dynamic programming on a grid, the exactly optimal no-trade band for a price model (exponential utility) |
| ↳ `mpc.py` | `mpc`: model predictive control by single shooting over sampled scenarios; plans `open` / `affine` / `band`, objectives `mean_var` / `cara` / `rsqp` |
| ↳ `ilqg.py` | `ilqg`: iterative LQG trajectory optimization in delta-hedged coordinates; receding horizon or solve-once |
| ↳ `models.py` | price models the model-based strategies plan with: `gbm` (current vol) or `gjr_garch` (filters the variance from observed prices) |
| `configs/` | one JSON per experiment |

New to options? Read [`GLOSSARY.md`](GLOSSARY.md) first: every term, with a worked example.
The planning methods (DP, MPC, iLQG), how they work and what had to be fixed: [`METHODS.md`](METHODS.md).

## The problem in one paragraph
We **sell** one option (a call: the buyer can buy one share at the *strike* K at expiry, so we owe `max(S_T − K, 0)`). We get paid a *premium* upfront. If the stock rallies we lose a lot, so we **hedge** by holding some shares of the stock: when the stock goes up, the shares gain value, which offsets what we'll owe. The Black-Scholes **delta** says how many shares to hold right now so that small price moves cancel out. Delta changes as price and time move, so we have to keep trading, and each trade costs money. That trade-off (stay close to delta vs. avoid paying costs) is the control problem.

**Episode** = one option's life (e.g. 21 trading days ≈ 1 month). At each step the strategy outputs a target number of shares; at expiry we pay the payoff and sell off the shares.
**Final P&L** = premium + share gains − payoff − costs. We judge strategies by a risk-adjusted trade-off: high average P&L (`mean_pnl`) and low spread (`std_pnl`, `cvar_5%` = average of the worst 5% of episodes). `rsqp` = √E[loss² when losing] (loss = −P&L) only penalizes losses; lower is better. In the ideal Black-Scholes world a perfect hedge ends at exactly 0 every episode; on real data the centre moves, but tightness is still the main job of the hedge.

**Why the options are synthetic:** Yahoo has no historical option prices, only real stock (and VIX) prices. So at each start date we *pretend* to sell an option priced by Black-Scholes using a volatility estimate (`vol` in the config), then let the **real** price path play out. When the estimate is wrong (it always is, somewhat), delta hedging is imperfect, and that's where a learned policy could do better. With `"vol": {"type": "implied"}` the estimate is the one the market actually used for that stock on that day, so the premium is close to what you'd really have been paid (minus the bid/ask spread, which we don't model).

## Config reference
```jsonc
{
  "name": "spy_vix",                       // results folder name
  "data": {"source": "yahoo", "ticker": "SPY", "start": "2005-01-01", "end": "2025-12-31", "interval": "1d"},
  //   or {"source": "gbm", "n_steps": 50000, "sigma": 0.2, "mu": 0.05, "seed": 0}   (simulated Black-Scholes world)
  //   or {"source": "gjr_garch", "n_paths": 1310720, "mu": ..., "omega": ..., "alpha": ..., "gamma": ..., "beta": ..., "h0": ..., "seed": 0}
  //      independent simulated paths with moving volatility, one episode each (needs a constant vol);
  //      see configs/paper_gjr_garch.json, the setting of Neagu et al. 2025 (arXiv:2504.05521)
  "vol": {"type": "ticker", "ticker": "^VIX", "scale": 0.01},   // vol used to price the option + compute delta
  //   or {"type": "realized", "window": 21}   trailing realized vol
  //   or {"type": "constant", "value": 0.2}
  //   or {"type": "implied", "symbol": "AAPL", "max_stale_days": 7}   the market's real ~30-day at-the-money
  //      implied vol for that stock, from DoltHub (free, US stocks, 2019-02 onward; see configs/aapl_implied.json).
  //      Gaps up to max_stale_days are filled with the last value; options overlapping longer gaps are skipped.
  "option": {"type": "call", "moneyness": 1.0, "maturity_steps": 21},   // moneyness = strike / starting price
  "market": {"rate": 0.03, "cost_bps": 5, "steps_per_year": 252},    // cost_bps: 5 = 0.05% of each $ traded
  "episodes": {"stride": 5},               // start a new option every 5 steps (default: no overlap)
  "plot_episode": 500,                     // optional: which episode to draw in the plot
  "eval": {"n_splits": 10, "risk_aversion": 1.0},  // optional: cut episodes into 10 "test sets" (rsqp = their average,
                                           //   rsqp_sd = spread); risk_aversion g adds ce = -log E[exp(-g P&L)] / g
  "strategies": [
    {"name": "delta_hedge", "label": "delta_weekly", "params": {"rebalance_every": 5}}
  ]
}
```
Every episode's prices are rescaled to start at $100, so all P&L numbers are "$ per option on $100 of stock".

## Adding a strategy
```python
# strategies/my_idea.py
from strategy import Observation, Strategy

class HalfDelta(Strategy):
    def act(self, obs: Observation) -> float:
        return 0.5 * obs.delta          # target shares to hold until the next step
```
Then register it in `strategies/__init__.py` (`"half_delta": HalfDelta`) and use `{"name": "half_delta"}` in a config.

## Using the environment directly (e.g. for RL later)
```python
from backtest import HedgingEnv, load_episodes   # episodes, _ = load_episodes(cfg)
env = HedgingEnv(option_type="call", moneyness=1.0, rate=0.03, cost_bps=5)
obs = env.reset(episode)                 # Observation; obs.to_array() gives a flat vector
obs, reward, done, info = env.step(0.5)  # action = shares to hold; reward = change in P&L this step
```
The rewards in an episode add up exactly to its final P&L.
