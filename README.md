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
| `strategies/` | concrete strategies + the name → class registry in `__init__.py`: `baselines.py` (no hedge, delta, delta band), `mpc.py` (model predictive control, single shooting) |
| `configs/` | one JSON per experiment |

New to options? Read [`GLOSSARY.md`](GLOSSARY.md) first: every term, with a worked example.

## The problem in one paragraph
We **sell** one option (a call: the buyer can buy one share at the *strike* K at expiry, so we owe `max(S_T − K, 0)`). We get paid a *premium* upfront. If the stock rallies we lose a lot, so we **hedge** by holding some shares of the stock: when the stock goes up, the shares gain value, which offsets what we'll owe. The Black-Scholes **delta** says how many shares to hold right now so that small price moves cancel out. Delta changes as price and time move, so we have to keep trading, and each trade costs money. That trade-off (stay close to delta vs. avoid paying costs) is the control problem.

**Episode** = one option's life (e.g. 21 trading days ≈ 1 month). At each step the strategy outputs a target number of shares; at expiry we pay the payoff and sell off the shares.
**Final P&L** = premium + share gains − payoff − costs. A perfect hedge has P&L ≈ 0 in every episode, so we judge strategies by how *spread out* P&L is (`std_pnl`, `cvar_5%` = average of the worst 5% of episodes) as well as by the average.

**Why the options are synthetic:** Yahoo has no historical option prices, only real stock (and VIX) prices. So at each start date we *pretend* to sell an option priced by Black-Scholes using a volatility estimate (`vol` in the config), then let the **real** price path play out. When the estimate is wrong (it always is, somewhat), delta hedging is imperfect, and that's where a learned policy could do better.

## Config reference
```jsonc
{
  "name": "spy_vix",                       // results folder name
  "data": {"source": "yahoo", "ticker": "SPY", "start": "2005-01-01", "end": "2025-12-31", "interval": "1d"},
  //   or {"source": "gbm", "n_steps": 50000, "sigma": 0.2, "mu": 0.05, "seed": 0}   (simulated Black-Scholes world)
  "vol": {"type": "ticker", "ticker": "^VIX", "scale": 0.01},   // vol used to price the option + compute delta
  //   or {"type": "realized", "window": 21}   trailing realized vol
  //   or {"type": "constant", "value": 0.2}
  "option": {"type": "call", "moneyness": 1.0, "maturity_steps": 21},   // moneyness = strike / starting price
  "market": {"rate": 0.03, "cost_bps": 5, "steps_per_year": 252},    // cost_bps: 5 = 0.05% of each $ traded
  "episodes": {"stride": 5},               // start a new option every 5 steps (default: no overlap)
  "plot_episode": 500,                     // optional: which episode to draw in the plot
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
from backtest import HedgingEnv, load_market_data, make_episodes
env = HedgingEnv(option_type="call", moneyness=1.0, rate=0.03, cost_bps=5)
obs = env.reset(episode)                 # Observation; obs.to_array() gives a flat vector
obs, reward, done, info = env.step(0.5)  # action = shares to hold; reward = change in P&L this step
```
The rewards in an episode add up exactly to its final P&L.
