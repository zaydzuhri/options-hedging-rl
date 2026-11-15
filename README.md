# options-hedging-rl: hedging backtester

```bash
uv sync
uv run backtest.py --config configs/spy_vix.json
```
Outputs go to `results/<name>/`: `summary.csv`, `episodes.csv` (one row per strategy × episode), `example_episode.csv` (step-by-step trajectory), `pnl.png`, and a copy of the config.

## Files
| file | what it is |
|---|---|
| `backtest.py` | everything that isn't a strategy: Black-Scholes, data loading, episodes, `HedgingEnv`, metrics, plot, CLI |
| `strategy.py` | the interface: `Observation` (what a strategy sees) and `Strategy` (`reset()`, `act(obs) -> shares`) |
| `deep_hedge.py` | deep hedging: differentiable P&L, neural policies, training with CVaR / entropic risk |
| `dp_optimal.py` | exact dynamic programming in the Black-Scholes world: the optimal no-trade band, a ground truth |
| `strategies/` | concrete strategies + the name → class registry in `__init__.py` |
| `configs/` | one JSON per experiment |

New to options? Read [`GLOSSARY.md`](GLOSSARY.md) first: every term, with a worked example.

## The problem in one paragraph
We **sell** one option (a call: the buyer can buy one share at the *strike* K at expiry, so we owe `max(S_T − K, 0)`). We get paid a *premium* upfront. If the stock rallies we lose a lot, so we **hedge** by holding some shares of the stock: when the stock goes up, the shares gain value, which offsets what we'll owe. The Black-Scholes **delta** says how many shares to hold right now so that small price moves cancel out. Delta changes as price and time move, so we have to keep trading, and each trade costs money. That trade-off (stay close to delta vs. avoid paying costs) is the control problem.

**Episode** = one option's life (e.g. 21 trading days ≈ 1 month). At each step the strategy outputs a target number of shares; at expiry we pay the payoff and sell off the shares.
**Final P&L** = premium + share gains − payoff − costs. A perfect hedge has P&L ≈ 0 in every episode, so we judge strategies by how *spread out* P&L is (`std_pnl`, `cvar_5%` = average of the worst 5% of episodes) as well as by the average.

## Control formulation
One episode is a finite-horizon stochastic control problem with N steps (N = `maturity_steps`), Δt = 1/`steps_per_year`, cost rate λ = `cost_bps`/10⁴, growth factor G = e^{rΔt}.

| | |
|---|---|
| **state** x_t | (S_t, σ̂_t, q_t, c_t, t): spot, pricing vol, shares held, cash, time. The strike K and maturity are fixed per episode. |
| **control** u_t | shares to hold over (t, t+1], within the position limits: q_min ≤ u_t ≤ q_max |
| **disturbance** w_t | (S_{t+1}/S_t, σ̂_{t+1}): the market's next move, from data or a simulator |
| **dynamics** | q_{t+1} = u_t,  c_{t+1} = G·(c_t − (u_t − q_t)·S_t − λ·\|u_t − q_t\|·S_t),  (S, σ̂) evolve exogenously |
| **terminal** | P&L = c_N + q_N·S_N − λ·\|q_N\|·S_N − payoff(S_N), with c_0 = premium and q_0 = 0 |
| **objective** | minimize a risk measure of the final P&L: CVaR_α or entropic risk (1/a)·log E[e^{−a·P&L}] |
| **information** | u_t may only depend on what is known at t (`Observation`): no look-ahead |

**Why one decision at a time is not enough.** Without costs (λ = 0), each step can be solved on its own: hold delta, whatever you held before. With costs, today's position is tomorrow's starting point, so q_t is part of the state and a trade now changes what future trades cost. The optimal policy becomes a *no-trade band* around delta whose width depends on the whole remaining horizon. That coupling is what `whalley_wilmott` approximates, `dp_optimal` solves exactly in the Black-Scholes world, and `deep_hedge` learns.

**Assumptions.** We are a price taker: our trades don't move S or σ̂, which is what lets us replay historical paths as disturbances (and what makes the P&L differentiable in the actions). Options are synthetic (see below), costs are proportional, and the hedge is settled in shares at expiry.

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
  "market": {"rate": 0.03, "cost_bps": 5, "steps_per_year": 252,    // cost_bps: 5 = 0.05% of each $ traded
             "position_limits": [0, 1]},  // optional hard limits on shares held; requests outside are clipped
                                          //   and counted in the `violations` column
  "episodes": {"stride": 5},               // start a new option every 5 steps (default: no overlap)
  "split": {"test_start": "2019-01-01"},   // optional: evaluate only on data from here on (a row number for gbm);
                                           //   learned strategies train on what comes before
  "plot_episode": 500,                     // optional: which episode to draw in the plot
  "strategies": [
    {"name": "delta_hedge", "label": "delta_weekly", "params": {"rebalance_every": 5}}
  ]
}
```
Every episode's prices are rescaled to start at $100, so all P&L numbers are "$ per option on $100 of stock".

## Built-in strategies
| name | what it does | params |
|---|---|---|
| `no_hedge` | never trades | |
| `delta_hedge` | holds Black-Scholes delta | `rebalance_every` |
| `delta_band` | trades only when the position drifts more than a fixed `band` from delta | `band`, `to_edge` |
| `whalley_wilmott` | no-trade band around delta whose width comes from costs, gamma and risk aversion (Whalley & Wilmott, 1997); the classical benchmark with costs | `risk_aversion` |
| `dp_optimal` | the exact optimal band from `dp_optimal.py` (gbm data with constant vol only) | `model` + solver params |
| `deep_hedge` | a neural policy trained by `deep_hedge.py` (see below) | `model` + training params |

## Adding a strategy
```python
# strategies/my_idea.py
from strategy import Observation, Strategy

class HalfDelta(Strategy):
    def act(self, obs: Observation) -> float:
        return 0.5 * obs.delta          # target shares to hold until the next step
```
Then register it in `strategies/__init__.py` (`"half_delta": HalfDelta`) and use `{"name": "half_delta"}` in a config.

## Deep hedging
```bash
uv run deep_hedge.py --config configs/gbm_costs.json   # trains every "deep_hedge" strategy in the config (~20 s each, CPU)
uv run backtest.py   --config configs/gbm_costs.json   # evaluates them out of sample next to the baselines
```
Our trades don't move the market, so for a fixed price path the final P&L is a differentiable function of the positions. `deep_hedge.py` re-implements `HedgingEnv`'s accounting in PyTorch and backpropagates a risk measure of the P&L through whole episodes (the *pathwise* gradient; Buehler et al., 2019, "Deep hedging"). After training it checks that the backtester gives the same P&L for the trained policy (to ~1e-13).

A `deep_hedge` strategy entry holds both where the checkpoint goes and how to train it:
```jsonc
{"name": "deep_hedge", "label": "deep_band", "params": {
  "model": "models/gbm_costs/deep_band.pt",      // written by deep_hedge.py, read by backtest.py
  "arch": "band",                                 // "band": learn a no-trade band around delta (Imaki et al., 2021)
                                                  // "mlp":  position = delta + MLP(features, current position)
  "objective": {"type": "cvar", "alpha": 0.05},   // or {"type": "entropic", "risk_aversion": 1.0}
  "steps": 3000, "batch": 1024, "lr": 1e-3, "hidden": 32, "layers": 2, "seed": 0,
  "train_stride": 1, "val_frac": 0.2              // overlapping training episodes; last 20% of training data picks the checkpoint
}}
```
The config needs a `split`: training uses data before `test_start`, the backtest only data from it on. For `band` policies, `deep_hedge.py` also saves `results/<name>/band_<label>.png`, the learned band next to Whalley-Wilmott's.

Experiments: `gbm_costs.json` (Black-Scholes world with 10 bps costs, where Whalley-Wilmott should be close to optimal) and `spy_vix_deep.json` (SPY priced with VIX, trained on 2005-2018, tested on 2019-2025).

## Exact dynamic programming (the ground truth)
```bash
uv run dp_optimal.py --config configs/gbm_costs.json   # ~3 s; writes models/gbm_costs/dp.npz
```
In the Black-Scholes world with entropic risk, cash factors out of the value function, which then lives on a (time, spot, shares held) grid and can be solved backwards exactly. The solution is a no-trade band; the `dp_optimal` strategy plays it in the backtester. Its params: `model`, `risk_aversion` (default 1), grid sizes `n_spot` (801) and `n_pos` (401). The solver also checks that the optimal action from every position on the grid really is "clip to the band". Halving both grid steps moves the optimal risk by < 0.0015.

On `gbm_costs` (20,476 test episodes, entropic risk with a = 1, lower is better):

| strategy | entropic risk | gap to optimal (95% CI) |
|---|---|---|
| `dp_optimal` (exact value 0.3094) | 0.3136 | |
| `deep_band` | 0.3152 | +0.0016 [+0.0007, +0.0026] |
| `deep_mlp` | 0.3236 | +0.0100 [+0.0081, +0.0120] |
| `delta_daily` | 0.3607 | +0.0471 [+0.0426, +0.0517] |
| `whalley_wilmott` (a = 1) | 0.3630 | +0.0494 [+0.0438, +0.0552] |

The learned band recovers 97% of the gap between daily delta hedging and the optimum. Whalley-Wilmott's band, derived for continuous trading, is too wide for daily rebalancing and does worse than plain delta hedging.

## Using the environment directly (e.g. for RL later)
```python
from backtest import HedgingEnv, load_market_data, make_episodes
env = HedgingEnv(option_type="call", moneyness=1.0, rate=0.03, cost_bps=5)
obs = env.reset(episode)                 # Observation; obs.to_array() gives a flat vector
obs, reward, done, info = env.step(0.5)  # action = shares to hold; reward = change in P&L this step
```
The rewards in an episode add up exactly to its final P&L.
