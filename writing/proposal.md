# Project proposal: Learning the no-trade band

**IFT 6162, Fall 2026.** Team: *(three names)*. Status: topic approved. Repository: github.com/zaydzuhri/options-hedging-rl

## System
A derivatives desk sells an option and hedges it by trading the underlying stock. Each trade pays a proportional cost (bid-ask spread), so the desk must trade off tracking the Black-Scholes delta against paying costs. Starting points: Whalley & Wilmott (1997), the asymptotically optimal no-trade band under proportional costs; Buehler et al. (2019), *Deep hedging*; Kamal & Derman (1999) for the discrete-hedging error we reproduce as a check. Data: daily SPY and VIX from Yahoo Finance, 2005-2025 (train before 2019, test 2019-2025), plus a Black-Scholes simulator where the optimum is computable.

## Control formulation
One short at-the-money call on one share, 21 daily steps.
- **State** x_t = (S_t, σ̂_t, q_t, c_t, t): spot, pricing volatility (VIX), shares held, cash, time.
- **Control** u_t: shares held until t+1, with the constraint 0 ≤ u_t ≤ 1.
- **Disturbance** w_t: next price move and volatility (simulated, or replayed history).
- **Dynamics** q_{t+1} = u_t; c_{t+1} = e^{rΔt}(c_t − (u_t − q_t)S_t − λ|u_t − q_t|S_t).
- **Objective**: minimize a risk measure of terminal P&L (premium + share gains − payoff − costs): entropic risk in the Black-Scholes world, CVaR 5% on real data.

**Why per-step decisions are not enough:** without costs, holding delta is optimal step by step; with costs, today's position is tomorrow's starting point, so the optimal policy is a no-trade band whose width depends on the remaining horizon.

**Assumption:** we are a price taker, so historical paths are valid disturbances for any policy (a justified action-response simulator). Options are synthetic (priced at VIX), since free historical option quotes are not available.

## Baselines and learning
- Baselines: daily delta hedging, a fixed band, and the published Whalley-Wilmott band.
- Ground truth: exact dynamic programming on a (t, spot, position) grid in the Black-Scholes world. Entropic risk lets cash factor out of the value function.
- Learning: a neural policy trained by backpropagating the risk of the P&L through whole episodes (pathwise gradients). There are two architectures: a learned no-trade band (Imaki et al., 2021) and an unconstrained MLP. We also train on a GJR-GARCH simulator fitted to the training years. We compare pathwise with score-function (REINFORCE) gradients on the same problem.

## Verification
Tests reproduce the Kamal-Derman hedging error within 10%, check that the training P&L equals the backtester's P&L (to 1e-13), check DP grid convergence, and check that the simulated DP policy matches the DP value. All evaluations are out of sample, with paired or block-bootstrap confidence intervals.

## Preliminary results
- **Black-Scholes world, 10 bps costs (20,476 unseen paths):** the learned band is within 0.5% of the DP optimum (it closes 97% of the gap from daily delta). Whalley-Wilmott is no better than daily delta, because its band assumes continuous trading.
- **SPY 2019-2025:** learned policies improve CVaR 5% from -1.81 (daily delta) to between -0.92 and -1.31. The tail is two events, so the confidence intervals are wide.
- **Gradients:** the score-function gradient needs roughly 60-500× more episodes than the pathwise gradient for the same accuracy.

## Status
The instructor approved the topic (November 2026). This file is kept as a record of the proposal.
