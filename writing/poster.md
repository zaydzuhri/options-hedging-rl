# Poster draft (≈ 650 words; figures in docs/figures/)

**Title:** Plan or learn the no-trade band?
**Main result (one sentence):** With 10 bps trading costs, model predictive control closes 93% of the gap between daily delta hedging and the exact optimum with no training at all, and a learned no-trade band 96%; the published Whalley-Wilmott band closes none of it.

*(QR code to the project website, top right)*

## Problem
We sell a one-month at-the-money call and hedge it by trading the stock once a day. The Black-Scholes delta tells us how many shares cancel small moves, but delta changes daily and every trade pays a proportional cost. Without costs, holding delta is optimal one day at a time. With costs, today's position is tomorrow's starting point, so the best policy is a **no-trade band** around delta whose width depends on the whole remaining horizon.

## Model *(system diagram: disturbance → state → control → dynamics, with the 0 ≤ u ≤ 1 constraint)*
- State: spot, pricing vol, shares held, cash, days left.
- Control: shares to hold until tomorrow, between 0 and 1.
- Disturbance: the market's next move, from a simulator or replayed history.
- Objective, the same for every method: entropic risk (1/a)·log E[exp(−a·P&L)] = −certainty equivalent.
- Assumption: we are a price taker, so historical paths are valid disturbances, and the P&L is differentiable in our actions.

## Methods: a formula, a model, or data
1. **Whalley-Wilmott (1997):** the published band for small costs and continuous trading. Needs only a formula.
2. **Dynamic programming:** with entropic risk, cash drops out of the value function; we solve the (day, spot, shares) grid backwards. Exact when the model is right: our ground truth.
3. **Model predictive control (planning):** every day, simulate 1,024 futures from the model, optimize today's trade plus a no-trade band for the remaining days (three numbers) by gradient descent through the simulated P&L, trade, and re-plan tomorrow. Needs a model; no training; ~10 ms per decision.
4. **Deep hedging (learning):** a network outputs the band's edges; trained offline by backpropagating the risk of the P&L through whole episodes. Needs data or a simulator; one forward pass per decision.

## Results
**Fig. 1 (bands.png). Where to stop trading: MPC and the learned band follow the optimal band** (MPC's edges are jagged: each point is a fresh plan over random futures). Whalley-Wilmott's band, derived for continuous trading, is far too wide when you can only trade once a day.

**Fig. 2 (risk_vs_cost.png). Across costs from 0 to 20 bps, on 20,476 unseen paths, MPC and the learned band stay close to the optimum** (MPC slightly closer at 0-2 bps, the learned band at 5 bps and above). At 10 bps, MPC is 0.0031 from optimal (95% CI 0.0018 to 0.0044) and the learned band 0.0017 (0.0008 to 0.0026). Whalley-Wilmott is worse than daily delta at 2-5 bps, ties it at 10 bps, and beats it only at 20 bps.

**Fig. 3 (real_data.png). On four real stocks (SPY, KO, JNJ, AAPL; trained or tuned before 2019, tested 2019-2025), the planners gave the steadiest gains over daily delta:** positive on SPY, KO and JNJ (significant on KO and JNJ), with the narrowest intervals, and level with delta on AAPL, though they plan with a plain Black-Scholes model. The learned policies swung from the best (AAPL, significant) to the worst (SPY): they learned how many shares to hold from their training years, so on AAPL, which kept rising, they held more stock than delta and earned the trend.

**Fig. 4 (gradient_estimators.png). Why not REINFORCE or PPO: with the same episodes, the score-function gradient is 9-22× noisier, so it needs roughly 80-480× more episodes for the same accuracy.** With the same optimizer, it barely improves on the starting policy. MPC and deep hedging both use the pathwise gradient.

## How we checked the numbers
- We reproduce the Kamal & Derman (1999) discrete-hedging error: 0.43 observed vs 0.45 predicted.
- The DP converges as the grid is refined (0.3169 → 0.3109 → 0.3095), a brute-force check confirms the band is optimal from every grid position, and simulating the DP policy gives its exact value within the confidence interval.
- The training P&L equals the backtester's to 1e-13, and its gradient equals finite differences.
- Limiting cases hold: no costs → every band collapses to delta; more costs → wider band.
- Automated tests (`uv run pytest`).

## Limitations
- Options are synthetic (priced at VIX or realized volatility); real quotes would add a smile and option spreads.
- On real data the planners use a simple model; the tails of the test period are a few events, so intervals are wide.
- The exact optimum is only available in the Black-Scholes world.
- We hedge with the stock only.
