# Planning methods for hedging with costs (non-learned)

Two model-based methods live in `strategies/`: **dynamic programming** (`dp.py`) and **model predictive
control** (`mpc.py`). Both plan with a *model* of the price
(`strategies/models.py`: GBM or GJR-GARCH) and are evaluated on whatever data the config gives them
(the same model, or real prices). Read [`GLOSSARY.md`](GLOSSARY.md) first for the option terms.

## The control problem they all solve

- **State:** time t, price S (and, under GARCH, the current variance v), shares held h, cash.
- **Action:** u = shares to hold until the next step. Trading |u − h| shares costs c·|u − h|·S.
- **Score:** how good the *final* P&L distribution is. Three scores are used:
  - **CE (certainty equivalent)** = −log E[exp(−g·P&L)] / g. *Technical:* the sure amount that an
    exponential-utility investor with risk aversion g values the random P&L at. *Intuition:* "the
    average, minus a penalty for spread", ≈ mean − (g/2)·variance when risks are small. Higher is better.
  - **mean-variance**: −mean + (λ/2)·variance. Same idea, simpler formula.
  - **RSQP** = √E[loss² when losing] (the RL paper's measure). Lower is better.
- **Why costs make it a real control problem:** without costs, the best action is "hold the
  minimum-variance hedge" at every step independently. With costs, today's position is tomorrow's starting
  point, so every trade must weigh the risk it removes against what it costs, now and later.

**The known answer's shape.** With proportional costs the optimal policy is a **no-trade band** around
(roughly) delta: if your shares are inside the band, do nothing; if outside, trade only back to the
nearest edge. The band is wide when gamma is large (delta moves a lot, so chasing it is expensive).
Whalley & Wilmott (1997) give a formula for small costs: `ww_band` in `baselines.py`.

---

## 1. Dynamic programming (`dp.py`)

*Technical.* Backward induction on a grid. With exponential utility, the wealth you already have drops
out of the problem, so the value only depends on (t, log S [, v], h). Money is counted in expiry dollars
(G = e^{r·dt}): a cost paid at step t weighs G^{N−t}, and holding h' shares earns h'·(S' − G·S)·G^{N−t−1}:

    psi_N(S, h)  = g · (payoff(S) + c|h|S)                                         pay the payoff, sell the shares
    G_t(S, h')   = log E[ exp(−g·h'·G^{N−t−1}·(S' − G·S) + psi_{t+1}(S', h')) ]   hold h' over one step
    psi_t(S, h)  = min_{h'}  g·c·S·G^{N−t}·|h' − h|  +  G_t(S, h')                trade from h to h'
    risk         = −G^N · premium + psi_0(S_0, 0) / g                               (= −CE)

G is convex in h', which proves the band: `lo = argmin G(h') + a·h'`, `hi = argmin G(h') − a·h'`
(a = g·c·S·G^{N−t}), and the optimal action is `clip(h, lo, hi)`; `solve_gbm(check=True)` confirms it by
brute force from every grid position. Under GBM the log-price is a **Markov chain on the grid**: each cell
gets the exact normal probability of its interval, so the expectation is a matrix product. (The first version
used Gauss–Hermite quadrature with linear interpolation; its value jumped by ±0.008 between grid sizes,
because of where the nodes fell between grid points. The Markov chain, from the teammates' solver, converges
smoothly: 0.3169 → 0.3109 → 0.3095 as both grid steps halve.) Under GJR-GARCH (a 2-D state) it is still
quadrature. The tables are solved once (0.4 s for GBM, ~25 s for GARCH), then each step is a lookup.

*Intuition.* Start at expiry, where you know exactly what every (price, shares) state is worth, and walk
backward one day at a time: "if I'm here tomorrow, I know how good that is; so today, which holding
gives the best tomorrow on average, after paying to get there?"

*Strengths/limits.* **Exactly optimal for its model** (up to the grid), so it is the benchmark everything
else is measured against. Before the two branches were merged, two DP solvers were written independently
(different grids, expectations and band extraction); they agreed to the fourth decimal on 20,476 episodes, and
the one kept combines the best of both. Needs a small state (3–4 dimensions) and a time-consistent objective
(exponential utility is; terminal variance is not).

*Pitfall found:* under GJR-GARCH, exponential utility with g = 1 is **infinite**: after a crash the
variance explodes, a two-month loss behaves like a product of two normals, and E[exp(g·loss)] diverges.
The DP then "optimizes" an infinite number and returns nonsense (hedging 0.25 shares instead of 0.54).
For the paper's monthly GARCH setting use g ≲ 0.3.

## 2. Model predictive control (`mpc.py`)

*Technical.* Every step: sample M price scenarios from the model, optimize a *plan* for the rest of the
option's life by single shooting (roll the plan forward on all scenarios, score the final P&L, get the
gradient by a backward adjoint pass, L-BFGS), apply only the first action, repeat next step.
Plans (`plan`):

- `open`: one number per future step, the same in every scenario. **Can't react**, so it overestimates
  future risk; poor (it plans as if it could never rebalance).
- `affine`: u_k = a_k + b_k·(S_k − S_0), a feedback rule ("tube MPC").
- `band`: today's trade u_0 is a free number; future steps follow `clip(h, delta + a − w, delta + a + w)`,
  i.e. the optimal policy's shape. `shared_band: true` uses one (a, w) for all future steps.

Objectives (`objective`): `mean_var`, `cara` (= CE), `rsqp`, `cvar`.

*Intuition.* "Imagine many possible futures, find the plan that does best across them, take today's step,
look at what actually happened, imagine again."

*Pitfalls found:*
- **Band plan with a band at step 0 too stalls**: all scenarios start from the same position, so when that
  position is inside the band the gradient for today's trade is exactly zero and the optimizer stops
  (it stopped trading). Fix: today's trade is a free variable; bands only for future steps.
- **Overfitting the scenarios**: a band per future step (≈ 40 numbers) fitted to 256 scenarios makes jittery
  decisions. Fix: one shared band (3 numbers) and 1024 scenarios.

## Also tried: iLQG (removed)

Iterative LQG (Todorov & Li 2005) in delta-hedged coordinates worked, but stayed behind MPC and DP, more so
at high cost (CE gap to DP −0.022 / −0.084 at 20 / 50 bps, vs −0.005 / −0.005 for the MPC band plan), and
its local nature showed: solving once and following the feedback law was the worst method tested. The code,
its write-up and its results are in commit `ab0995a` (`strategies/ilqg.py`, this file's section 3).

---

## Results so far (all with trading costs)

Each setting below has a results folder made by the normal runner (`results/<config name>/`: `metrics.png`,
`example.png`, `summary.csv`, `comparison.csv`, `episodes.csv`): `gbm_costs`, `gbm_costs_50bps`,
`paper_gjr_garch_costs`, `spy_vix_costs`, `ko_costs`, `jnj_costs`, `aapl_realized_costs`, and the benchmarks
`gbm_bench`, `spy_bench`, `ko_bench`, `jnj_bench`, `aapl_bench`. For speed, use `uv run backtest.py --config ... --workers 10`
(the planners take 0.3–2 s per episode).

Paired comparisons: every strategy runs on the *same* episodes, so differences are measured episode by
episode (much less noisy than comparing two averages). Error bars are bootstrap standard errors; on real
data with overlapping episodes they are **block** bootstraps (resampling runs of 8 neighbouring episodes).

**A. Simulated GBM world (`configs/gbm_costs.json`, `configs/gbm_costs_50bps.json`).** The model is right,
so DP is the true optimum. 2000 one-month ATM calls, daily rebalancing, score = CE with g = 1 (higher is
better). Tuning was done on seed 1; these numbers are on seed 2 (never tuned on).

| method | CE gap to DP, 20 bps | CE gap to DP, 50 bps |
|---|---|---|
| dp (optimal) | 0 | 0 |
| mpc, band plan (`mpc_band`) | −0.005 ± 0.003 | −0.005 ± 0.003 |
| mpc, affine plan | −0.015 ± 0.004 | −0.023 ± 0.006 |
| Whalley–Wilmott band | −0.077 ± 0.011 | −0.174 ± 0.016 |
| delta daily | −0.121 ± 0.014 | −0.435 ± 0.028 |

**B. The RL paper's GJR-GARCH world with 1% costs (`configs/paper_gjr_garch_costs.json`).** Score = RSQP
(lower is better), 10 test sets × 1024 paths. Strategies plan with the true GARCH model.

| method | RSQP | vs the paper's delta baseline |
|---|---|---|
| mpc, band plan, RSQP objective | **1.505** | −0.34 (−19%) |
| mpc, affine plan, RSQP objective | 1.572 | −0.28 |
| dp (exponential utility g = 0.3) | 1.653 | −0.20 |
| delta monthly (paper's baseline) | 1.849 | — |
| Whalley–Wilmott band | 2.310 | +0.46 |
| dp (g = 0.1) | 2.659 | +0.81 |

DP's two rows show why the risk aversion matters. The model has positive drift, so the less risk-averse
DP (g = 0.1) holds *extra* shares to earn it: the highest mean P&L (+1.66) but the worst RSQP. It even sits at
the edge of its share grid (1.1 shares, `h_range`), so its real optimum bets even harder. The CE that DP
maximizes and the RSQP the paper scores are different goals; MPC wins on RSQP because it optimizes RSQP
directly.

**C/D. Real data, the planners use GBM at the implied vol (they don't know the true dynamics).**
(Sections A–D were run with the first DP solver, which ignored interest and used quadrature. Re-running `dp` with
the merged solver gives the same CE on `gbm_costs`, `gbm_costs_50bps` and `spy_vix_costs`, and a CE 0.015 lower
on `ko_costs` and 0.027 lower on `jnj_costs`; the conclusions below are unchanged.)
- **SPY 2005–2025, VIX premium, 20 bps (`configs/spy_vix_costs.json`, 1053 overlapping episodes, g = 0.3):**
  CE vs the WW band: mpc affine +0.071 ± 0.020, dp +0.064 ± 0.020, mpc band +0.051 ± 0.022,
  delta −0.128 ± 0.035. The planners' gains survive on real prices.
- **AAPL 2019–2026, implied-vol premium, 5 and 20 bps (`configs/aapl_costs.json`, 372 episodes):** no
  significant differences between any of the planners and WW (error bars 2–4x larger than on SPY). Also:
  CE with g = 1 is dominated by the single worst month there (fat tails), so use g ≈ 0.3 on real data.
- **Long single-stock histories, 1990–2025, realized vol** (premium and planners use the trailing 21-day
  realized vol, since implied-vol data starts in 2019; 1805 episodes each; every setting copied from the SPY
  config, nothing tuned). Differences vs the WW band, block bootstrap (z = difference / SE):

  | stock | mpc_band CE | mpc_band RSQP | dp CE | mpc_affine CE | delta CE |
  |---|---|---|---|---|---|
  | KO (`ko_costs.json`) | **+0.061 (z 3.3)** | **−0.063 (z −3.1)** | +0.087 (z 5.0) | +0.077 (z 3.9) | −0.100 |
  | JNJ (`jnj_costs.json`) | +0.033 (z 1.5) | −0.029 (z −1.2) | +0.063 (z 3.2) | +0.038 (z 1.8) | −0.145 |
  | AAPL (`aapl_realized_costs.json`) | −1.64 (z −1.6) | −0.007 (z −0.1) | −0.61 | −1.00 | −1.62 |
  | AAPL without Sept 2000 | +0.106 (z 1.9) | **−0.086 (z −2.7)** | +0.134 (z 2.5) | +0.096 (z 1.7) | +0.001 |

  AAPL's full-sample CE is **one day**: on 2000-09-29 AAPL fell 52% overnight. The 3 episodes containing it
  are every strategy's worst, and which strategy loses most there is luck (how many shares it happened to hold
  at the gap). For mpc_band that one episode is 34% of E[exp(−g·P&L)]. Without the 5 episodes around it,
  AAPL looks like the others. vs **delta**, mpc_band's mean P&L is +0.21 (z ≈ 8–9) on KO and JNJ: delta's daily
  trading costs dominate.

  On real data **DP is consistently a little better than mpc_band**: mean +0.018 to +0.025 (z 2.4–3.0) on
  SPY, KO, JNJ and AAPL-without-2000, while on simulated GBM they are equal (mpc_affine sits in between).
  Diagnosed on KO (300 episodes): mpc_band holds fewer shares than delta, on average 0.061 fewer vs 0.051
  (affine) and 0.038 (dp), most of it early in the option's life. The planners assume the stock grows at the
  interest rate (3%/yr); KO actually returned ~12%/yr over 1990–2025, so each share-month held earned ~$0.76.
  That drift explains about two thirds of the mean-P&L gaps (predicted −0.018 vs dp, −0.008 vs affine;
  observed −0.028, −0.012). Tiny trades are not the cause (about one trade under 0.01 shares per episode for
  all three). So on real data this is mostly "who held more stock in a rising market", not hedging skill.

**E. The benchmark shared with the learned methods (`configs/gbm_bench.json`).** Black-Scholes world, 10 bps,
positions limited to [0, 1], 20,476 unseen episodes, entropic risk with a = 1 (= −CE, lower is better), the
objective every method optimizes. Gap to the exact optimum, paired bootstrap 95% CI:

| strategy | entropic risk | gap to the optimum (95% CI) | share of daily delta's gap closed | trades |
|---|---|---|---|---|
| `dp` (exact optimum) | 0.3136 |  | 100% | 14.6 |
| `mpc_band` (no training) | 0.3167 | +0.0031 [+0.0018, +0.0044] | 93% | 14.4 |
| `deep_band` (learned band) | 0.3152 | +0.0017 [+0.0008, +0.0026] | 96% | 14.8 |
| `delta_daily` | 0.3607 | +0.0472 [+0.0427, +0.0518] | 0% | 21.0 |
| `ww` (a = 1) | 0.3630 | +0.0495 [+0.0438, +0.0553] | -5% | 11.3 |

The learned band is slightly but significantly closer to optimal than MPC here; MPC needs no training data and
re-plans online, the learned band needs training but is a single forward pass per decision. `make_site.py`
repeats this at 0, 2, 5, 10 and 20 bps (the website's band explorer and the poster's Fig. 2).

**F. Real data under one protocol (`configs/{spy,ko,jnj,aapl}_bench.json`).** Train (or tune) before 2019, test
2019–2025: a one-month ATM call started every 5 days, 348 per stock, 10 bps, positions in [0, 1], everyone
optimizing entropic risk with a = 0.3. SPY is priced at VIX, the stocks at trailing 21-day realized vol. The
planners use GBM at the pricing vol with drift = the interest rate; the learned policies train on the stock's
own history before 2019 (`deep_band`, `deep_mlp`) or on a GJR-GARCH fitted to it (`deep_band_garch`).
Certainty-equivalent gain over daily delta ($ per $100 of stock, higher is better), paired block bootstrap
(blocks of 10 overlapping episodes), 95% CI; bold = the CI excludes 0:

| CE gain over daily delta | SPY | KO | JNJ | AAPL |
|---|---|---|---|---|
| `ww` | +0.010 [−0.062, +0.086] | +0.080 [−0.082, +0.244] | +0.201 [−0.007, +0.482] | +0.020 [−0.207, +0.293] |
| `dp` (GBM model) | +0.047 [−0.006, +0.102] | **+0.136** [+0.003, +0.255] | **+0.166** [+0.035, +0.323] | −0.023 [−0.110, +0.051] |
| `mpc_band` | +0.041 [−0.008, +0.093] | **+0.110** [+0.007, +0.205] | **+0.164** [+0.027, +0.332] | −0.034 [−0.136, +0.058] |
| `deep_band` | +0.007 [−0.085, +0.112] | +0.095 [−0.111, +0.274] | +0.028 [−0.235, +0.288] | **+0.256** [+0.026, +0.563] |
| `deep_mlp` | −0.055 [−0.208, +0.109] | +0.126 [−0.108, +0.327] | +0.330 [−0.066, +0.862] | **+0.311** [+0.080, +0.572] |
| `deep_band_garch` | +0.064 [−0.010, +0.154] | +0.084 [−0.089, +0.236] | +0.132 [−0.057, +0.314] | +0.132 [−0.052, +0.360] |

The planners are the steadiest: the same sign and similar size on every stock, the narrowest intervals, and
DP and MPC agree with each other (both significant on KO and JNJ, both level with delta on AAPL). The
learned policies swing from the best (AAPL, significant) to the worst (SPY): they learn how many shares to
hold from their training years. On AAPL they hold more stock than delta (+0.045 band, +0.085 MLP on average,
vs −0.018 for DP), and AAPL kept rising (19%/yr before 2019, 33%/yr after), so they earned the trend; on SPY
they hold 0.11 shares less than delta, which helped in the 2020 crash but cost in the rally. That is the same
drift effect as the band-vs-affine diagnosis above, now learned from data: on real prices, a good part of
every comparison is "who held more stock in a rising market".

**Takeaways.** (1) Where the model is right, MPC with the band plan is essentially optimal and DP is the
exact reference; both beat the classical Whalley–Wilmott band, by more as costs rise. (2) In the RL
paper's world with costs, MPC optimizing the paper's own risk measure beats their delta baseline by 19%.
(3) On real data (SPY, KO, JNJ; AAPL apart from one 52% crash day) every planner beats WW in the same direction,
significantly on SPY and KO; there, DP is slightly but consistently the best planner.
(4) On the shared benchmark, MPC band (no training) and the learned no-trade band both come close to the exact
optimum, the learned band slightly closer; Whalley–Wilmott does not beat daily delta at 10 bps. (5) On real
data under one protocol, the planners give the most consistent gains over daily delta; the learned policies are
the best on some stocks and the worst on others, mostly through the trend they learned.
