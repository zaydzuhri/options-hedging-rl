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
| `strategy.py` | the interface: `Observation` (what a strategy sees) and `Strategy` (`reset()`, `act(obs) -> shares`), plus the `black_scholes` formula |
| `plots.py` | the figures (and `comparison.csv`), drawn from a results folder |
| `train.py` | trains the learned policies (deep hedging) in a config; `--gradients` runs the pathwise-vs-REINFORCE study |
| `make_site.py` | regenerates the project website's data (`docs/data.js`) and the poster figures (`docs/figures/`) |
| `strategies/` | concrete strategies + the name → class registry in `__init__.py` (each file can be deleted on its own): |
| ↳ `baselines.py` | `no_hedge`, `delta_hedge`, `delta_band`, `ww_band` (Whalley–Wilmott no-trade band) |
| ↳ `dp.py` | `dp`: dynamic programming on a grid, the exactly optimal no-trade band for a price model (exponential utility) |
| ↳ `mpc.py` | `mpc`: model predictive control by single shooting over sampled scenarios; plans `open` / `affine` / `band`, objectives `mean_var` / `cara` / `rsqp` / `cvar` |
| ↳ `learned.py` | `deep_hedge`: neural policies (a learned no-trade band, or an MLP) trained by `train.py` |
| ↳ `models.py` | price models the planners use (`gbm`, `gjr_garch`), and the one GJR-GARCH recursion every simulator shares (with a fit to data, for training) |
| `configs/` | one JSON per experiment; `*_bench.json` are the benchmarks below |
| `tests/` | `uv run pytest` (no network): pricing identities, accounting, limits, a published result, parity, DP checks |
| `docs/` | the project website (one page, GitHub Pages) and the poster figures |
| `writing/` | the course proposal (as approved) and the poster text |

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

**Why one decision at a time is not enough.** Without costs (λ = 0), each step can be solved on its own: hold delta, whatever you held before. With costs, today's position is tomorrow's starting point, so q_t is part of the state and a trade now changes what future trades cost. The optimal policy becomes a *no-trade band* around delta whose width depends on the whole remaining horizon. That coupling is what `ww_band` approximates with a formula, `dp` solves exactly for a price model, `mpc` plans online with a model, and `deep_hedge` learns offline from data.

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
| `ww_band` | no-trade band around delta whose width comes from costs, gamma and risk aversion (Whalley & Wilmott, 1997); the classical benchmark with costs | `risk_aversion` |
| `dp` | the optimal band for a price model (`gbm`, or `gjr_garch`), solved on a grid when the strategy is created; see [`METHODS.md`](METHODS.md) | `risk_aversion`, `h_range` (= the position limits), `model`, `vol_grid`, grid sizes |
| `mpc` | model predictive control: optimize a plan over sampled futures, play its first trade, re-plan every step; see [`METHODS.md`](METHODS.md) | `plan`, `objective`, `horizon`, `risk_aversion`, ... |
| `deep_hedge` | a neural policy trained by `train.py` (see below) | `model` + training params |

## Adding a strategy
```python
# strategies/my_idea.py
from strategy import Observation, Strategy

class HalfDelta(Strategy):
    def act(self, obs: Observation) -> float:
        return 0.5 * obs.delta          # target shares to hold until the next step
```
Then register it in `strategies/__init__.py` (`"half_delta": HalfDelta`) and use `{"name": "half_delta"}` in a config.

## Learned policies (deep hedging): `train.py`
```bash
uv run train.py    --config configs/gbm_bench.json   # trains every "deep_hedge" strategy in the config (~1 min each, CPU)
uv run backtest.py --config configs/gbm_bench.json   # evaluates them out of sample next to everything else
```
Our trades don't move the market, so for a fixed price path the final P&L is a differentiable function of the positions. `train.py` re-implements `HedgingEnv`'s accounting in PyTorch and backpropagates a risk measure of the P&L through whole episodes (the *pathwise* gradient; Buehler et al., 2019, "Deep hedging"). After training it checks that the backtester gives the same P&L for the trained policy (to ~1e-13).

A `deep_hedge` strategy entry holds both where the checkpoint goes and how to train it:
```jsonc
{"name": "deep_hedge", "label": "deep_band", "params": {
  "model": "models/gbm_bench/deep_band.pt",       // written by train.py, read by backtest.py
  "arch": "band",                                 // "band": learn a no-trade band around delta (Imaki et al., 2021)
                                                  // "mlp":  position = delta + MLP(features, current position)
  "objective": {"type": "entropic", "risk_aversion": 1.0},   // or {"type": "cvar", "alpha": 0.05}
  "steps": 3000, "batch": 1024, "lr": 1e-3, "hidden": 32, "layers": 2, "seed": 0,
  "train_stride": 1, "val_frac": 0.2,             // overlapping training episodes; the last 20% of training data picks the checkpoint
  "simulator": {"type": "gjr_garch", "n_paths": 100000}   // optional: train on paths from a GJR-GARCH fitted to the training years
}}
```
The config needs a `split`: training uses data before `test_start`, the backtest only data from it on.

## Benchmarks: every method under one protocol
Every method optimizes the same objective, the **entropic risk** (1/a)·log E[e^(−a·P&L)] = −CE: `dp` and `ww_band`
by construction, `mpc` with `objective: "cara"`, the learned policies with `objective: {"type": "entropic"}`.
Positions are limited to [0, 1] shares, everyone is scored out of sample on the same episodes, and differences
are paired (block) bootstraps. Each `*_bench.json` runs with `uv run train.py` then `uv run backtest.py --workers 10`
(MPC is the slow one: ~25 min for the 20,476 GBM episodes, ~1 min per real asset).

**Black-Scholes world (`gbm_bench.json`)**: σ = 20%, no drift, 10 bps, a = 1, 20,476 unseen episodes. Here the
model is right, so `dp` is the exact optimum and the ground truth.

| strategy | entropic risk | gap to the optimum (95% CI) | share of daily delta's gap closed | trades |
|---|---|---|---|---|
| `dp` (exact optimum) | 0.3136 |  | 100% | 14.6 |
| `mpc_band` (no training) | 0.3167 | +0.0031 [+0.0018, +0.0044] | 93% | 14.4 |
| `deep_band` (learned band) | 0.3152 | +0.0017 [+0.0008, +0.0026] | 96% | 14.8 |
| `delta_daily` | 0.3607 | +0.0472 [+0.0427, +0.0518] | 0% | 21.0 |
| `ww` (a = 1) | 0.3630 | +0.0495 [+0.0438, +0.0553] | -5% | 11.3 |

Gap = extra entropic risk over the optimum, paired bootstrap 95% CI. `deep_mlp` (an unconstrained network) reaches 0.3236 (79% of the gap closed). The DP's exact value is 0.3095; the simulated 0.3136 differs by sampling noise (95% CI 0.3056 to 0.3211). Across costs (`make_site.py`): MPC's gap is 0.0014, 0.0017, 0.0024, 0.0031, 0.0053 at 0, 2, 5, 10, 20 bps, the learned band's 0.0027, 0.0023, 0.0018, 0.0017, 0.0020; Whalley–Wilmott is worse than daily delta at 2–5 bps, ties it at 10 and beats it only at 20.

**Real data (`spy_bench.json`, `ko_bench.json`, `jnj_bench.json`, `aapl_bench.json`)**: train before 2019, test
2019–2025 (one-month calls started every 5 days, ~350 per asset), 10 bps, a = 0.3 (with a = 1 a single bad month
dominates the score). SPY is priced at VIX, the stocks at their trailing 21-day realized vol. The planners use a
GBM at the pricing vol with drift = the interest rate (they don't know the real dynamics); the learned policies
train on the real history before 2019, or on a GJR-GARCH fitted to it (`_garch`).

| CE gain over daily delta | SPY | KO | JNJ | AAPL |
|---|---|---|---|---|
| `ww` | +0.010 [−0.062, +0.086] | +0.080 [−0.082, +0.244] | +0.201 [−0.007, +0.482] | +0.020 [−0.207, +0.293] |
| `dp` (GBM model) | +0.047 [−0.006, +0.102] | **+0.136** [+0.003, +0.255] | **+0.166** [+0.035, +0.323] | −0.023 [−0.110, +0.051] |
| `mpc_band` | +0.041 [−0.008, +0.093] | **+0.110** [+0.007, +0.205] | **+0.164** [+0.027, +0.332] | −0.034 [−0.136, +0.058] |
| `deep_band` | +0.007 [−0.085, +0.112] | +0.095 [−0.111, +0.274] | +0.028 [−0.235, +0.288] | **+0.256** [+0.026, +0.563] |
| `deep_mlp` | −0.055 [−0.208, +0.109] | +0.126 [−0.108, +0.327] | +0.330 [−0.066, +0.862] | **+0.311** [+0.080, +0.572] |
| `deep_band_garch` | +0.064 [−0.010, +0.154] | +0.084 [−0.089, +0.236] | +0.132 [−0.057, +0.314] | +0.132 [−0.052, +0.360] |

Certainty-equivalent gain over daily delta ($ per $100 of stock, higher is better), 348 test options per stock, paired block bootstrap 95% CI (blocks of 10 overlapping episodes); bold = the interval excludes 0. The planners are the steadiest: same sign and similar size everywhere, the narrowest intervals, significant on KO and JNJ, level with delta on AAPL. The learned policies swing from the best (AAPL) to the worst (SPY): they learn how many shares to hold from their training years, and on AAPL, which kept rising, they held more stock than delta (see METHODS.md, F).

## Why not plain RL? Pathwise vs score-function gradients
```bash
uv run train.py --config configs/gbm_bench.json --gradients   # ~1 min
```
Policy-gradient methods (REINFORCE, PPO) estimate the gradient from which random actions happened to come with good outcomes: the *score function*, E[U · Σ_t ∇log π(u_t | x_t)]. Here the P&L is differentiable in the actions, so we can use the *pathwise* gradient E[∇U] instead. Both are unbiased estimates of the same ∇E[exp(−a·P&L)] for the same Gaussian policy, on the same episodes and noise (the score function even gets a leave-one-out baseline).

On `gbm_bench`, at the initial policy and with a batch of 1024 episodes, the score function's relative error is ~22× the pathwise one with exploration noise 0.01 shares (~480× more episodes for the same accuracy) and ~9× with 0.05 (~80×). With the same Adam optimizer, batch and 1.5 M training episodes, pathwise training reaches validation risk 0.308; REINFORCE barely moves from the starting 0.321. Output: `results/gbm_bench/gradient_estimators.png`.

## Project website and poster
`docs/index.html` is a one-page site with interactive figures (band explorer, closed-loop replay, real-data replays, gradient estimators) driven by precomputed results in `docs/data.js`. Open it directly in a browser, or publish it with GitHub Pages: a repo admin goes to Settings → Pages → Build and deployment → Source "Deploy from a branch" → branch `main`, folder `/docs`. A private repo needs GitHub Pro (included in the Student Developer Pack) for Pages; the site itself is public either way. The site then lives at https://zaydzuhri.github.io/options-hedging-rl/, which is what the poster's QR code (`docs/figures/qr.png`, `qr.svg`) points to. If the site moves, regenerate the code: `uv run --with segno python -c "import segno; q = segno.make('<url>', error='m'); q.save('docs/figures/qr.png', scale=24, border=2); q.save('docs/figures/qr.svg', scale=10, border=2)"`.

To refresh it: `uv run train.py --config configs/gbm_bench.json --gradients`, then `uv run make_site.py`. It trains whatever learned policy is missing, evaluates everything through the backtester (cached in `results/site/cache/`, so only what changed is recomputed; the first run takes ~2.5 h because of MPC at five cost levels), and writes `docs/data.js` and the poster figures `docs/figures/{bands,risk_vs_cost,real_data,gradient_estimators}.png`. The poster text is `writing/poster.md`.

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
| DP: the solution is a band (brute force from every position); no costs → zero width; more costs → wider; grid refinement converges; simulating the DP policy reproduces the DP value (also with interest) and beats delta | `test_dp.py` |
| GJR-GARCH: fitting data simulated from known parameters recovers them | `test_garch.py` |

## Using the environment directly (e.g. for RL later)
```python
from backtest import HedgingEnv, load_episodes   # episodes, _ = load_episodes(cfg)
env = HedgingEnv(option_type="call", moneyness=1.0, rate=0.03, cost_bps=5)
obs = env.reset(episode)                 # Observation; obs.to_array() gives a flat vector
obs, reward, done, info = env.step(0.5)  # action = shares to hold; reward = change in P&L this step
```
The rewards in an episode add up exactly to its final P&L (minus the day-one markup, if `vol_markup` is set).
