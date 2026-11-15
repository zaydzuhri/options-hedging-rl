# Poster draft (≈ 600 words; figures in docs/figures/)

**Title:** Learning the no-trade band
**Main result (one sentence):** With 10 bps trading costs, a neural no-trade band trained by differentiating through the hedging P&L comes within 0.5% of the exact dynamic-programming optimum, while the published Whalley-Wilmott band is no better than plain daily delta hedging.

*(QR code to the project website, top right)*

## Problem
We sell a one-month at-the-money call and hedge it by trading the stock once a day. The Black-Scholes delta tells us how many shares cancel small moves, but delta changes daily and every trade pays a proportional cost. Without costs, holding delta is optimal one day at a time. With costs, today's position is tomorrow's starting point, so the best policy is a **no-trade band** around delta whose width depends on the whole remaining horizon.

## Model *(system diagram: disturbance → state → control → dynamics, with the 0 ≤ u ≤ 1 constraint)*
- State: spot, pricing vol, shares held, cash, days left.
- Control: shares to hold until tomorrow, between 0 and 1.
- Disturbance: the market's next move, from a simulator or replayed SPY history.
- Objective: entropic risk of final P&L (Black-Scholes world) or CVaR 5% (SPY).
- Assumption: we are a price taker, so historical paths are valid disturbances, and the P&L is differentiable in our actions.

## Methods
1. **Published baseline:** the Whalley-Wilmott (1997) band, plus daily delta and a fixed band.
2. **Exact optimum:** with entropic risk, cash factors out of the value function. We solve the remaining (day, spot, shares) grid backwards and recover the optimal band.
3. **Learning:** a network outputs the band edges given moneyness, time and volatility. We train it by backpropagating the risk of the P&L through all 21 trades (pathwise gradient).

## Results
**Fig. 1 (bands.png). The learned band is close to the optimal one; Whalley-Wilmott's is far too wide.** Whalley-Wilmott assumes continuous trading, but between daily trades delta moves by about the width of its band.

**Fig. 2 (risk_vs_cost.png). Across costs from 0 to 20 bps, the learned policy tracks the optimum on 20,476 unseen paths.** At 10 bps it is 0.0016 from optimal (95% CI 0.0007 to 0.0026), closing 97% of the gap from daily delta. Whalley-Wilmott is worse than delta at 2-5 bps, ties it at 10 bps, and beats it only at 20 bps.

**Fig. 3 (spy_cvar.png). On SPY 2019-2025, never seen in training, learned policies improve the worst-5% outcome from -1.81 to as good as -0.92 per $100.** They hold about 0.1 share less than delta near the money. That loses a little in rallies and much less in crashes. Policies trained on a GJR-GARCH simulator learn the same tilt.

**Fig. 4 (gradient_estimators.png). Why not REINFORCE or PPO: with the same episodes, the score-function gradient is 8-23× noisier, which means 60-500× more episodes for the same accuracy.** With the same optimizer, it barely improves on the starting policy.

## How we checked the numbers
- We reproduce the Kamal & Derman (1999) discrete-hedging error: 0.43 observed vs 0.45 predicted.
- DP grid refinement converges, and simulating the DP policy gives its exact value within the confidence interval.
- The training P&L equals the backtester's to 1e-13, and its gradient equals finite differences.
- Limiting cases hold: no costs → every band collapses to delta; more costs → wider band.
- 29 automated tests.

## Limitations
- Options are synthetic (priced at VIX); real quotes would add a smile and option spreads.
- The SPY tail is two events, so its intervals are wide.
- The exact optimum is only available in the Black-Scholes world.
- We hedge with the stock only.
