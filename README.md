# options-hedging-rl: hedging backtester

```bash
uv sync
uv run backtest.py --config configs/spy_vix.json
```
Outputs go to `results/<name>/`: `summary.csv`, `episodes.csv` (one row per strategy × episode), `example_episode.csv` (step-by-step trajectory), a copy of the config, and two figures drawn by `plots.py`:
- `metrics.png`: final P&L across episodes as a box plot, then every metric per strategy (mean, std, CVaR, CE, RSQP,
  cost, trades) as a box showing how much it would move with other episodes, next to the baseline (numbers in `comparison.csv`)
- `example.png`: one episode, the price and each strategy's shares next to Black-Scholes delta

To re-draw finished runs without re-running them: `uv run plots.py results/<name> [results/<other> ...]`.

## Files
| file | what it is |
|---|---|
| `backtest.py` | everything that isn't a strategy: data loading, episodes, `HedgingEnv`, metrics, CLI |
| `plots.py` | the figures (and `comparison.csv`), drawn from a results folder |
| `strategy.py` | the interface: `Observation` (what a strategy sees) and `Strategy` (`reset()`, `act(obs) -> shares`), plus the `black_scholes` formula |
| `strategies/` | concrete strategies + the name → class registry in `__init__.py` (each file can be deleted on its own): |
| ↳ `baselines.py` | `no_hedge`, `delta_hedge`, `delta_band`, `ww_band` (Whalley–Wilmott no-trade band; also registered as `whalley_wilmott`) |
| ↳ `dp.py` | `dp`: dynamic programming on a grid, the exactly optimal no-trade band for a price model (exponential utility) |
| ↳ `optimal.py` | `dp_optimal`: plays the band that `dp_optimal.py` solved (Black-Scholes world, position limits) |
| ↳ `mpc.py` | `mpc`: model predictive control by single shooting over sampled scenarios; plans `open` / `affine` / `band`, objectives `mean_var` / `cara` / `rsqp` |
| ↳ `models.py` | price models the model-based strategies plan with: `gbm` (current vol) or `gjr_garch` (filters the variance from observed prices) |
| ↳ `learned.py` | `deep_hedge`: loads a policy trained by `deep_hedge.py` |
| `deep_hedge.py` | deep hedging: differentiable P&L, neural policies, training with CVaR / entropic risk |
| `dp_optimal.py` | exact dynamic programming in the Black-Scholes world: the optimal no-trade band, a ground truth |
| `gradient_estimators.py` | pathwise vs score-function (REINFORCE) gradients on the same problem |
| `garch.py` | GJR-GARCH(1,1) fitted to a training period, simulated with bootstrapped residuals, to train on more than one history |
| `make_site.py` | regenerates the project website's data (`docs/data.js`) and the poster figures (`docs/figures/`) |
| `docs/` | the project website (one page, GitHub Pages) |
| `writing/` | drafts of the course proposal and the poster text |
| `configs/` | one JSON per experiment |
| `tests/` | `uv run pytest` (~10 s, no network): pricing identities, accounting, limits, a published result, parity, DP checks |

New to options? Read [`GLOSSARY.md`](GLOSSARY.md) first: every term, with a worked example.
The planning methods (DP, MPC), how they work and what had to be fixed: [`METHODS.md`](METHODS.md).

## The problem in one paragraph
We **sell** one option (a call: the buyer can buy one share at the *strike* K at expiry, so we owe `max(S_T − K, 0)`). We get paid a *premium* upfront. If the stock rallies we lose a lot, so we **hedge** by holding some shares of the stock: when the stock goes up, the shares gain value, which offsets what we'll owe. The Black-Scholes **delta** says how many shares to hold right now so that small price moves cancel out. Delta changes as price and time move, so we have to keep trading, and each trade costs money. That trade-off (stay close to delta vs. avoid paying costs) is the control problem.

**Episode** = one option's life (e.g. 21 trading days ≈ 1 month). At each step the strategy outputs a target number of shares; at expiry we pay the payoff and sell off the shares.
**Final P&L** = premium + share gains − payoff − costs. We judge strategies by a risk-adjusted trade-off: high average P&L (`mean_pnl`) and low spread (`std_pnl`, `cvar_5%` = average of the worst 5% of episodes). `rsqp` = √E[loss² when losing] (loss = −P&L) only penalizes losses; lower is better. In the ideal Black-Scholes world a perfect hedge ends at exactly 0 every episode; on real data the centre moves, but tightness is still the main job of the hedge.

## Control formulation
One episode is a finite-horizon stochastic control problem with N steps (N = `maturity_steps`), Δt = 1/`steps_per_year`, cost rate λ = `cost_bps`/10⁴, growth factor G = e^{rΔt}.

| | |
|---|---|
| **state** x_t | (S_t, σ̂_t, q_t, c_t, t): spot, pricing vol, shares held, cash, time. The strike K and maturity are fixed per episode. |
| **control** u_t | shares to hold over (t, t+1], within the position limits if any: q_min ≤ u_t ≤ q_max |
| **disturbance** w_t | (S_{t+1}/S_t, σ̂_{t+1}): the market's next move, from data or a simulator |
| **dynamics** | q_{t+1} = u_t,  c_{t+1} = G·(c_t − (u_t − q_t)·S_t − λ·\|u_t − q_t\|·S_t),  (S, σ̂) evolve exogenously |
| **terminal** | P&L = c_N + q_N·S_N − λ·\|q_N\|·S_N − payoff(S_N), with c_0 = premium and q_0 = 0 |
| **objective** | minimize a risk measure of the final P&L: entropic risk (1/a)·log E[e^{−a·P&L}] = −CE, CVaR_α, RSQP, or mean − variance |
| **information** | u_t may only depend on what is known at t (`Observation`): no look-ahead |

**Why one decision at a time is not enough.** Without costs (λ = 0), each step can be solved on its own: hold delta, whatever you held before. With costs, today's position is tomorrow's starting point, so q_t is part of the state and a trade now changes what future trades cost. The optimal policy becomes a *no-trade band* around delta whose width depends on the whole remaining horizon. That coupling is what `ww_band` approximates, `dp` and `dp_optimal` solve exactly for a price model, `mpc` plans online, and `deep_hedge` learns.

**Assumptions.** We are a price taker: our trades don't move S or σ̂, which is what lets us replay historical paths as disturbances (and what makes the P&L differentiable in the actions). Options are synthetic (see below), costs are proportional, and the hedge is settled in shares at expiry.

**Why the options are synthetic:** Yahoo has no historical option prices, only real stock (and VIX) prices. So at each start date we *pretend* to sell an option priced by Black-Scholes using a volatility estimate (`vol` in the config), then let the **real** price path play out. When the estimate is wrong (it always is, somewhat), delta hedging is imperfect, and that's where a learned policy could do better. With `"vol": {"type": "implied"}` the estimate is the one the market actually used for that stock on that day, so the premium is close to what you'd really have been paid (minus the bid/ask spread; `option.vol_markup` can add one).

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
  //   optional "vol_markup": 0.02   charge the buyer the price at vol + 2 vol points, like a market maker's ask,
  //      instead of the fair price; or "leland": the markup that pays for daily delta-hedging costs on average
  //      (Leland 1985, about +2.4 vol points at 20 bps). Each episode, every strategy's P&L moves up by the same $,
  //      so rankings and paired differences don't change (except for the RSQP objective/metric, which only sees losses)
  "market": {"rate": 0.03, "cost_bps": 5, "steps_per_year": 252,    // cost_bps: 5 = 0.05% of each $ traded
             "position_limits": [0, 1]},  // optional hard limits on shares held; requests outside are clipped
                                          //   and counted in a `violations` column (shown when any happen)
  "episodes": {"stride": 5},               // start a new option every 5 steps (default: no overlap)
  "split": {"test_start": "2019-01-01"},   // optional: evaluate only on data from here on (a row number for gbm);
                                           //   learned strategies train on what comes before
  "plot_episode": 500,                     // optional: which episode to draw in example.png (default: the middle one)
  "eval": {"n_splits": 10, "risk_aversion": 1.0},  // optional: cut episodes into 10 "test sets" (rsqp = their average,
                                           //   rsqp_sd = spread); risk_aversion g adds ce = -log E[exp(-g P&L)] / g;
                                           //   "baseline": a strategy label to compare against in metrics.png
                                           //   (default: the ww_band strategy, else delta_hedge, else the first)
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
| `ww_band` (= `whalley_wilmott`) | no-trade band around delta whose width comes from costs, gamma and risk aversion (Whalley & Wilmott, 1997); the classical benchmark with costs | `risk_aversion` |
| `dp` | the optimal band for a price model (`gbm`, or `gjr_garch`), solved on a grid when the strategy is created; see [`METHODS.md`](METHODS.md) | `risk_aversion`, `model`, `vol_grid`, grid sizes |
| `dp_optimal` | the exact optimal band from `dp_optimal.py` (gbm data with constant vol only), inside the position limits | `model` + solver params |
| `mpc` | model predictive control: optimize a plan over sampled futures, play its first trade, re-plan every step; see [`METHODS.md`](METHODS.md) | `plan`, `objective`, `horizon`, `risk_aversion`, ... |
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
uv run deep_hedge.py --config configs/gbm_deep.json   # trains every "deep_hedge" strategy in the config (~20 s each, CPU)
uv run backtest.py   --config configs/gbm_deep.json   # evaluates them out of sample next to the baselines
```
Our trades don't move the market, so for a fixed price path the final P&L is a differentiable function of the positions. `deep_hedge.py` re-implements `HedgingEnv`'s accounting in PyTorch and backpropagates a risk measure of the P&L through whole episodes (the *pathwise* gradient; Buehler et al., 2019, "Deep hedging"). After training it checks that the backtester gives the same P&L for the trained policy (to ~1e-13).

A `deep_hedge` strategy entry holds both where the checkpoint goes and how to train it:
```jsonc
{"name": "deep_hedge", "label": "deep_band", "params": {
  "model": "models/gbm_deep/deep_band.pt",      // written by deep_hedge.py, read by backtest.py
  "arch": "band",                                 // "band": learn a no-trade band around delta (Imaki et al., 2021)
                                                  // "mlp":  position = delta + MLP(features, current position)
  "objective": {"type": "cvar", "alpha": 0.05},   // or {"type": "entropic", "risk_aversion": 1.0}
  "steps": 3000, "batch": 1024, "lr": 1e-3, "hidden": 32, "layers": 2, "seed": 0,
  "train_stride": 1, "val_frac": 0.2,             // overlapping training episodes; last 20% of training data picks the checkpoint
  "simulator": {"type": "gjr_garch", "n_paths": 100000}   // optional: train on simulated paths instead (see garch.py)
}}
```
The config needs a `split`: training uses data before `test_start`, the backtest only data from it on. For `band` policies, `deep_hedge.py` also saves `results/<name>/band_<label>.png`, the learned band next to Whalley-Wilmott's.

Experiments: `gbm_deep.json` (Black-Scholes world with 10 bps costs, where Whalley-Wilmott should be close to optimal) and `spy_vix_deep.json` (SPY priced with VIX, trained on 2005-2018, tested on 2019-2025).

### Results on real data (SPY, test 2019-2025, 348 episodes)
| strategy | CVaR 5% | std | mean P&L | CVaR gain vs daily delta (block-bootstrap 95% CI) |
|---|---|---|---|---|
| `delta_daily` | -1.81 | 0.71 | 0.31 | |
| `delta_band_0.1` | -1.63 | 0.70 | 0.34 | |
| `whalley_wilmott` (a = 1) | -1.72 | 0.73 | 0.34 | |
| `deep_band_cvar` | -1.26 | 0.63 | 0.28 | +0.54 [+0.02, +1.12] |
| `deep_mlp_cvar` | -0.92 | 0.53 | 0.25 | +0.89 [+0.02, +1.80] |
| `deep_band_cvar_garch` | -1.31 | 0.66 | 0.26 | +0.50 [+0.03, +0.99] |
| `deep_mlp_cvar_garch` | -1.16 | 0.60 | 0.22 | +0.64 [-0.03, +1.28] |

Read with care. The worst 5% of test episodes come from two events (Feb-Mar 2020 and Mar 2025), so the intervals are wide. The improvement is consistent across seeds (MLP CVaR between -0.86 and -1.02 for 4 seeds) and comes from one behavior: every learned policy holds ~0.1 share *less* than delta when the option is at or in the money, which loses a little in rallies (lower mean P&L) and a lot less in crashes. Policies trained only on simulated GJR-GARCH paths learn the same tilt, so it is not a memorized 2008. Training on the simulator did not beat training on the real history here (MLP: -0.25 CVaR, CI [-0.55, -0.02]).

## Exact dynamic programming (the ground truth)
```bash
uv run dp_optimal.py --config configs/gbm_deep.json   # ~3 s; writes models/gbm_deep/dp.npz
```
In the Black-Scholes world with entropic risk, cash factors out of the value function, which then lives on a (time, spot, shares held) grid and can be solved backwards exactly. The solution is a no-trade band; the `dp_optimal` strategy plays it in the backtester. Its params: `model`, `risk_aversion` (default 1), grid sizes `n_spot` (801) and `n_pos` (401). The solver also checks that the optimal action from every position on the grid really is "clip to the band". Halving both grid steps moves the optimal risk by < 0.0015.

On `gbm_deep` (20,476 test episodes, entropic risk with a = 1, lower is better):

| strategy | entropic risk | gap to optimal (95% CI) |
|---|---|---|
| `dp_optimal` (exact value 0.3094) | 0.3136 | |
| `deep_band` | 0.3152 | +0.0016 [+0.0007, +0.0026] |
| `deep_mlp` | 0.3236 | +0.0100 [+0.0081, +0.0120] |
| `delta_daily` | 0.3607 | +0.0471 [+0.0426, +0.0517] |
| `whalley_wilmott` (a = 1) | 0.3630 | +0.0494 [+0.0438, +0.0552] |
| `dp` (the other DP solver, `strategies/dp.py`) | 0.3136 | −0.0000 [−0.0001, +0.0001] |
| `mpc_band` (MPC, no training) | 0.3167 | +0.0030 [+0.0017, +0.0044] |

The last two rows come from `configs/gbm_deep_mpc.json` (same 20,476 test episodes; it loads the `gbm_deep` models,
so run `dp_optimal.py` and `deep_hedge.py` on `configs/gbm_deep.json` first; MPC takes ~25 min with `--workers 10`). The two DP solvers were written independently (different grids, expectations and band
extraction) and agree to the fourth decimal; MPC band closes 94% of daily delta's gap, the learned band 97%.

The learned band recovers 97% of the gap between daily delta hedging and the optimum. Whalley-Wilmott's band, derived for continuous trading, is too wide for daily rebalancing: at 10 bps it is no better than plain daily delta hedging (paired difference +0.0023, 95% CI [−0.0062, +0.0114]), at 2 and 5 bps it is worse, and only at 20 bps does it beat delta (`make_site.py` runs all five cost levels).

## Why not plain RL? Pathwise vs score-function gradients
```bash
uv run gradient_estimators.py --config configs/gbm_deep.json   # ~30 s
```
Policy-gradient methods (REINFORCE, PPO) estimate the gradient from which random actions happened to come with good outcomes: the *score function*, E[U · Σ_t ∇log π(u_t | x_t)]. Here the P&L is differentiable in the actions, so we can use the *pathwise* gradient E[∇U] instead. Both are unbiased estimates of the same ∇E[exp(−a·P&L)] for the same Gaussian policy, on the same episodes and noise (the score function even gets a leave-one-out baseline).

On `gbm_deep`, at the initial policy, the score function's relative error is ~23× the pathwise one with exploration noise s = 0.01 shares (~500× more episodes for the same accuracy) and ~8× with s = 0.05 (~60×). With the same Adam optimizer, batch and 1.5 M training episodes, pathwise training reaches validation risk 0.307; REINFORCE barely moves from the starting 0.321 (daily delta: 0.337). Output: `results/gbm_deep/gradient_estimators.png`.

## Project website and poster
`docs/index.html` is a one-page site with interactive figures (band explorer, closed-loop replay, SPY replay, gradient estimators) driven by precomputed results in `docs/data.js`. Open it directly in a browser, or publish it with GitHub Pages: a repo admin goes to Settings → Pages → Build and deployment → Source "Deploy from a branch" → branch `main` (or `deep-hedging`), folder `/docs`. A private repo needs GitHub Pro (included in the Student Developer Pack) for Pages; the site itself is public either way. The site then lives at https://zaydzuhri.github.io/options-hedging-rl/, which is what the poster's QR code (`docs/figures/qr.png`, `qr.svg`) points to. If the site moves, regenerate the code: `uv run --with segno python -c "import segno; q = segno.make('<url>', error='m'); q.save('docs/figures/qr.png', scale=24, border=2); q.save('docs/figures/qr.svg', scale=10, border=2)"`. To refresh it after changing the code: train the SPY models and run `gradient_estimators.py` (see above), then `uv run make_site.py` (~6 min). That also writes the poster figures `docs/figures/{bands,risk_vs_cost,spy_cvar,gradient_estimators}.png`.

## Tests: why you can trust the numbers
```bash
uv run pytest
```
| what | test |
|---|---|
| Black-Scholes: put-call parity, delta and gamma are the derivatives of the price, payoff at expiry | `test_pricing.py` |
| Accounting: rewards sum to the final P&L; unhedged P&L = premium with interest − payoff | `test_env.py` |
| **Reproduced result**: zero-cost daily delta hedging error has std ≈ √(π/4)·vega·σ/√N (Kamal & Derman, 1999), within 10% | `test_env.py` |
| Constraints: requests outside the position limits are clipped and counted; learned policies never ask for one | `test_env.py`, `test_deep_hedge.py` |
| Chronological split never mixes train and test | `test_env.py` |
| Limiting cases of Whalley-Wilmott: no costs → delta hedge; more risk aversion → narrower band | `test_strategies.py` |
| The differentiable P&L equals the backtester's (to 1e-10); its gradient equals finite differences | `test_deep_hedge.py` |
| DP: the solution is a band; no costs → zero width; more costs → wider; grid refinement converges; simulating the DP policy reproduces the DP value and beats delta | `test_dp_optimal.py` |

## Using the environment directly (e.g. for RL later)
```python
from backtest import HedgingEnv, load_episodes   # episodes, _ = load_episodes(cfg)
env = HedgingEnv(option_type="call", moneyness=1.0, rate=0.03, cost_bps=5)
obs = env.reset(episode)                 # Observation; obs.to_array() gives a flat vector
obs, reward, done, info = env.step(0.5)  # action = shares to hold; reward = change in P&L this step
```
The rewards in an episode add up exactly to its final P&L (minus the day-one markup, if `vol_markup` is set).
