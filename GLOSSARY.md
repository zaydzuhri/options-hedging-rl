# Glossary: every term used in this project

Each term has a **Technical** meaning (the precise definition), an **Intuition** (what it means in plain terms), and, where relevant, **In the code** (the variable or function it corresponds to).

Read it top to bottom; later terms build on earlier ones. Section 8 is a fully worked 3-step example that ties everything together, with numbers you can reproduce from the code.

---

## 1. P&L

### P&L (Profit and Loss)
- **Technical:** P&L stands for *Profit and Loss*. It is the change in the value of everything you own (your "positions") over some period. Positive = you made money, negative = you lost money. For us:
  ```
  P&L = (what we own now) − (what we owe now) − (what we started with)
  ```
  We start with nothing (we receive the premium and immediately owe an option of exactly that value, so the net is 0). That means P&L is simply our current net worth from this trade:
  ```
  P&L = cash + (shares held × share price) − (value of the option we sold)
  ```
- **Intuition:** "If I closed everything down right now, how much richer or poorer would I be than before I started?" It's the running score of the trade.
- **Two flavours:**
  - **Final P&L:** the P&L at expiry, when the option is settled and the shares are sold. That's real, locked-in money. This is what the backtest reports per episode.
  - **Mark-to-market P&L** (see §6): the P&L *during* the episode, using today's estimated option value. It's a "paper" number that can still change.
- **In the code:** `obs.pnl` (running), `env.pnl` at the end of an episode (final), the `pnl` column in `episodes.csv`. The env's `reward` is the *change* in P&L over one step.

### What we want P&L to look like: high on average AND predictable
- **Technical:** We maximize a **risk-adjusted objective** over the distribution of final P&L across episodes:
  ```
  maximize   E[P&L] − λ · Risk[P&L]        (Risk = std, variance, or CVaR; λ = risk aversion)
  ```
  The mean is good; the spread is bad; λ says how many dollars of average profit you'd give up to remove one unit of risk.
- **Intuition:** A trader selling options is like an insurance company: collect the fee, don't gamble on where the stock goes. A strategy that sometimes makes +10 and sometimes −10 is a bad hedge even if it averages +0.5. A strategy that always lands at +0.2 is a great hedge, and one that always lands at +0.3 is better still.
- **Where "near zero" comes from:** in the ideal Black-Scholes world (option priced at the true vol, free continuous trading), a perfect hedge ends at *exactly* 0 every time. So "P&L ≈ 0 in every episode" describes a perfect hedge *in that world*. Zero itself is not the target; *tightness* is. In real data the centre of the distribution moves away from 0 for reasons the hedger mostly doesn't control (premium vs. realized vol, see "Volatility risk premium"), and costs pull it down.

---

## 2. The stock side

### Underlying (asset)
- **Technical:** The asset an option's value depends on (here: a stock or ETF like SPY or AAPL).
- **Intuition:** The "thing" the option is a bet on. The option is *derived* from it, which is why options are called **derivatives**.
- **In the code:** `data.ticker` in the config.

### Ticker
- **Technical:** The short code a security trades under on an exchange. `SPY` = an ETF tracking the S&P 500 index; `^VIX` = the VIX index (Yahoo marks indices with `^`).
- **Intuition:** Its name on the stock market.

### ETF / SPY
- **Technical:** Exchange-Traded Fund: a fund that holds a basket of assets and trades like a single stock. SPY holds the 500 stocks in the S&P 500.
- **Intuition:** Buying one share of SPY ≈ buying a tiny slice of the whole US large-cap market.

### Spot (price)
- **Technical:** The current market price of the underlying, usually written `S` or `S_t` (price at time t).
- **Intuition:** "What does one share cost right now?"
- **In the code:** `obs.spot`, `episode.spots`.

### Adjusted close
- **Technical:** The day's closing price, adjusted backwards in time for stock splits and dividends so that the series reflects true investment returns.
- **Intuition:** If a stock splits 2-for-1, the raw price halves overnight even though nobody lost money. Adjusting removes those fake jumps.
- **In the code:** `download_close(...)` uses `auto_adjust=True`.

### Position / long / short
- **Technical:** Your *position* is how many units of something you hold. **Long** = positive quantity (you own it; you gain when the price rises). **Short** = negative quantity (you owe it; you gain when the price falls).
- **Short selling a stock:** you borrow shares from someone, sell them now, and later buy them back to return them. If the price fell in between, you bought back cheaper → profit.
- **Intuition:** Long = "I want it to go up." Short = "I want it to go down."
- **In the code:** `obs.position` = shares held. It can be negative (e.g. when hedging a put).

### Return / log return
- **Technical:** Simple return from t to t+1 = `S_{t+1}/S_t − 1`. Log return = `ln(S_{t+1}/S_t)`. For small moves they are almost equal (1% ≈ 0.995% in log terms). Log returns add up over time, which makes the maths cleaner.
- **Intuition:** The percentage the price moved.
- **In the code:** `np.log(df["spot"]).diff()` in `load_market_data` (used for realized volatility).

### Drift (μ, "mu")
- **Technical:** The average growth rate of the price per year in a model like GBM (see §4).
- **Intuition:** The direction the stock tends to go on average, ignoring the noise. Stocks drift up a few % per year.
- **In the code:** `data.mu` for the `gbm` source.

### Basis point (bp, bps)
- **Technical:** One hundredth of a percent: 1 bp = 0.01% = 0.0001.
- **Intuition:** A unit for small percentages. "5 bps cost" = you lose 0.05% of every dollar you trade, i.e. 5 cents per $100.
- **In the code:** `market.cost_bps`; the env converts it with `cost_bps / 1e4`.

---

## 3. Options

### Option
- **Technical:** A contract giving its **buyer** the *right, but not the obligation*, to buy or sell the underlying at a fixed price (the strike) on or before a fixed date (the expiry). The **seller** (also called the *writer*) has the *obligation* to fulfil it if the buyer chooses to use it ("exercises").
- **Intuition:** An insurance policy or a lottery ticket on a stock's price. The buyer pays a fee upfront; the seller collects the fee and takes on the risk.

### Call option
- **Technical:** The right to **buy** the underlying at the strike `K`. At expiry it's worth `max(S_T − K, 0)`.
- **Intuition:** A bet that the price goes **up**. If the stock ends at 110 and K = 100, the buyer can buy at 100 something worth 110 → the option is worth 10. If the stock ends at 90, nobody would buy at 100 → worth 0.

### Put option
- **Technical:** The right to **sell** the underlying at the strike `K`. At expiry it's worth `max(K − S_T, 0)`.
- **Intuition:** A bet that the price goes **down**, or insurance against a crash. If the stock ends at 90 and K = 100, the buyer can sell at 100 something worth 90 → worth 10.
- **In the code:** `option.type` = `"call"` or `"put"`.

### European option
- **Technical:** An option that can only be exercised **at** expiry (vs. *American* options, exercisable any time before). Black-Scholes prices European options.
- **Intuition:** Simplest kind: only the final price matters for the payoff. All options in this project are European.

### Strike (K)
- **Technical:** The fixed price written into the contract at which the buyer can buy (call) or sell (put).
- **Intuition:** The line the stock must cross for the option to pay out.
- **In the code:** `obs.strike`, set as `moneyness × starting price`.

### Expiry / maturity (T)
- **Technical:** The date the option ends and is settled. *Maturity* also refers to the length of the option's life (e.g. "1-month maturity").
- **Intuition:** The deadline.
- **In the code:** `option.maturity_steps` (e.g. 21 trading days ≈ 1 month).

### Time to maturity (τ, "tau")
- **Technical:** Time remaining until expiry, in **years**: `τ = steps_left / steps_per_year`.
- **Intuition:** "How long until the deadline?" Matters because more time = more chance for a big move = more valuable option.
- **In the code:** `obs.tau`, `obs.steps_left`.

### Premium
- **Technical:** The price paid for the option at the start, by the buyer to the seller.
- **Intuition:** The insurance fee. We (the seller) receive it at step 0 and keep it no matter what.
- **In the code:** `env.premium`, the `premium` column in `episodes.csv`.

### Payoff
- **Technical:** What the option is worth at expiry: `max(S_T − K, 0)` for a call, `max(K − S_T, 0)` for a put. As the seller, we pay this.
- **Intuition:** The insurance claim we have to pay out at the end.
- **In the code:** `black_scholes(...)` returns it when `tau <= 0`.

### Intrinsic value / time value
- **Technical:** *Intrinsic value* = the payoff if the option expired right now. *Time value* = option price − intrinsic value.
- **Intuition:** An option is worth more than its intrinsic value because "things could still change before the deadline." That extra worth decays to 0 as expiry approaches.

### Moneyness; ITM / ATM / OTM
- **Technical:** How the strike compares with the current price. In this code, `moneyness = K / S_0`.
  - **ATM (at-the-money):** `K ≈ S` (moneyness 1.0).
  - **ITM (in-the-money):** the option would pay out right now (call: `S > K`; put: `S < K`).
  - **OTM (out-of-the-money):** it wouldn't pay out right now (call: `S < K`; put: `S > K`).
- **Intuition:** ATM options are coin flips and the most sensitive to price moves. Deep OTM options are cheap "lottery tickets" that usually expire worthless but occasionally pay big.
- **In the code:** `option.moneyness`. Example: a put with 0.95 = strike 5% below the starting price (OTM).

### Writing / selling an option; "the book"; exposure
- **Technical:** *Writing* = selling an option you didn't own, creating a short option position. The *book* = all positions a trader is responsible for. *Exposure* = how much the book's value changes with some market variable.
- **Intuition:** In this project, the book is: short 1 option + some shares + a cash account. Our exposure to the stock price is what hedging tries to cancel.

---

## 4. Pricing

### Black-Scholes (BS) model
- **Technical:** A 1973 formula giving the fair price of a European option, assuming: (1) the stock follows GBM with constant volatility, (2) you can trade continuously with no costs, (3) a constant risk-free interest rate. For a call:
  ```
  price = S·N(d1) − K·e^(−r·τ)·N(d2)
  d1 = [ln(S/K) + (r + σ²/2)·τ] / (σ·√τ),   d2 = d1 − σ·√τ
  ```
- **Intuition:** The key insight is that under those assumptions, you can perfectly **replicate** the option by continuously holding the right number of shares (delta). If you can replicate it exactly, its fair price must be the cost of replicating it. Our project lives in the gap where those assumptions *fail* (vol changes, trading is discrete and costs money).
- **In the code:** `black_scholes(spot, strike, tau, vol, rate, kind)` in `strategy.py` returns `(price, delta, gamma)`. `spot` can be a numpy array, to price many simulated scenarios at once (the MPC strategy does this).

### N(·): the standard normal CDF
- **Technical:** `N(x)` = probability that a standard normal random variable is ≤ x. Ranges from 0 to 1.
- **Intuition:** In Black-Scholes it shows up as a "probability-like" weight; for example, `N(d2)` is roughly the (risk-neutral) probability the call finishes in-the-money.
- **In the code:** `ndtr(...)` from `scipy.special`.

### Risk-free rate (r)
- **Technical:** The interest rate on a riskless loan (in practice, short-term US government bonds), annualized.
- **Intuition:** What cash earns sitting in a bank. Matters because if you hold cash you earn it, and if you borrow cash (negative cash) you pay it.
- **In the code:** `market.rate`, `obs.rate`; the env does `cash *= exp(rate * dt)` every step.

### Discounting
- **Technical:** Converting a future amount into today's money: `X today = X at time T × e^(−r·τ)`.
- **Intuition:** $100 next year is worth a bit less than $100 today, because today's $100 could earn interest meanwhile.

### Volatility (σ, "sigma")
- **Technical:** The standard deviation of the underlying's log returns, **annualized**. Example: 20% vol means one year's return has a standard deviation of roughly 20%.
- **Intuition:** How wildly the price swings. The single most important input to an option's price: more volatility = bigger possible moves = more valuable option (and more dangerous to have sold).

### Annualizing (√252)
- **Technical:** If daily returns have std `σ_daily`, then the annual std is `σ_daily × √252` (252 ≈ trading days per year), because variance adds over independent days.
- **Intuition:** Puts all volatilities on the same "per year" scale. 1% daily moves ≈ 16% annual vol.
- **In the code:** `market.steps_per_year`, the `math.sqrt(spy)` in `load_market_data`.

### Realized (historical) volatility
- **Technical:** Volatility measured from past price data: the std of the last N log returns, annualized.
- **Intuition:** "How bumpy has it actually been lately?" Backward-looking.
- **In the code:** `vol.type = "realized"` with `window` (e.g. 21 days).

### Implied volatility (IV)
- **Technical:** The volatility value that, when plugged into Black-Scholes, reproduces the option's actual market price. It's "implied" by the price, not measured from the past.
- **Intuition:** The market's forecast of future volatility, expressed through option prices. Forward-looking.

### VIX
- **Technical:** An index computed from S&P 500 option prices, measuring the market's implied volatility for the next 30 days, quoted in percent (VIX = 15 means 15% annualized).
- **Intuition:** The "fear gauge". Low (≈12–15) in calm markets, spikes (40–80) in crashes.
- **In the code:** `vol.type = "ticker"`, `ticker = "^VIX"`, with `scale = 0.01` to turn 15 into 0.15.

### Volatility risk premium
- **Technical:** The empirical fact that implied volatility tends to be higher than the volatility that is subsequently realized.
- **Intuition:** People overpay for insurance (options), so option sellers who hedge well earn a small profit on average. This is why delta-hedged SPY results in `spy_vix` have a **positive** mean P&L.

### GBM (Geometric Brownian Motion)
- **Technical:** A random price model where log returns over each step are independent, normally distributed with mean `(μ − σ²/2)·dt` and std `σ·√dt`. It is the model Black-Scholes assumes.
- **Intuition:** "Random walk in percentage terms with constant volatility." Real prices are not like this (vol changes, crashes happen), but it's the ideal world in which delta hedging is provably optimal. Useful as a sanity check.
- **In the code:** `data.source = "gbm"`, function `simulate_gbm`.

### Put-call parity
- **Technical:** For European options with the same strike and expiry: `Call − Put = S − K·e^(−r·τ)`.
- **Intuition:** A call and a put differ by a fixed amount of stock and cash. So once delta-hedged, short calls and short puts have nearly identical hedging P&L.

---

## 5. The Greeks (sensitivities)

"Greeks" are derivatives (in the calculus sense) of the option price with respect to its inputs. They're named after Greek letters.

### Delta (Δ)
- **Technical:** `Δ = ∂(option price)/∂S`: how much the option's price changes per $1 move in the stock. Call delta is between 0 and 1 (`N(d1)` in BS); put delta is between −1 and 0 (`N(d1) − 1`).
- **Intuition:** If delta = 0.5, the option moves 50 cents when the stock moves $1, so it behaves like half a share. Deep ITM call → delta ≈ 1 (acts like one share). Deep OTM → delta ≈ 0 (barely reacts).
- **Why it's the hedge:** We're short the option, so our P&L moves by `−Δ` per $1 stock move. Holding `Δ` shares adds `+Δ` per $1. Total ≈ 0. That's being **delta-neutral**.
- **In the code:** `obs.delta`. The `delta_hedge` strategy literally returns `obs.delta`.

### Gamma (Γ)
- **Technical:** `Γ = ∂Δ/∂S = ∂²(price)/∂S²`: how fast delta changes as the stock moves.
- **Intuition:** Gamma measures how quickly your hedge goes stale. High gamma (ATM, close to expiry) → delta jumps around → you must rebalance often → more trading cost and more hedging error. As the seller we are **short gamma**: every rebalance forces us to buy after the price has risen and sell after it has fallen (buy high, sell low). Those small losses are what the premium pays for.
- **In the code:** `obs.gamma`.

### Theta (Θ) and Vega (ν), not in the code yet
- **Theta:** how much the option's value drops per unit of time passing, all else equal. As the seller, time passing is good for us: we earn theta. Roughly, **theta earned vs. gamma losses** is the core trade-off of a hedged short option. If the stock moves *more* than the vol we priced with, gamma losses win; if *less*, theta wins.
- **Vega:** sensitivity of the price to volatility. Tells you how much a wrong vol estimate costs.

---

## 6. Hedging

### Hedge
- **Technical:** A position taken to offset the risk of another position.
- **Intuition:** Buying an umbrella shop's stock *and* a sunscreen shop's stock so the weather doesn't matter to you. Here: holding shares to offset the short option.

### Hedge position / action
- **Technical:** The number of shares of the underlying held from now until the next step.
- **Intuition:** The one dial the strategy controls.
- **In the code:** the float returned by `Strategy.act(obs)`. It's a **target** position; the env computes the trade needed to reach it.

### Delta hedging
- **Technical:** At each rebalancing time, set hedge position = option's delta.
- **Intuition:** "Hold however many shares makes small price moves cancel out." Perfect in Black-Scholes' world with continuous trading; imperfect in reality.

### Rebalancing; discrete hedging error
- **Technical:** *Rebalancing* = trading to bring the hedge back to its target. We can only rebalance at discrete times (daily, weekly), so between rebalances delta drifts and the hedge is wrong. The resulting P&L noise is the *discrete hedging error*.
- **Intuition:** It's like steering a car by only looking at the road once per second. More often = smoother, but each correction costs money.
- **In the code:** `delta_hedge` with `params.rebalance_every`.

### Transaction costs
- **Technical:** Money lost when trading: the bid-ask spread, commissions, and price impact. Modelled here as **proportional**: `cost = cost_rate × |shares traded| × price`.
- **Intuition:** The fee for adjusting the hedge. It's what makes "rebalance constantly" a bad idea and turns this into a real control problem.
- **In the code:** `market.cost_bps`; `cost` in `HedgingEnv.step`; `mean_cost` in the summary.

### Bid-ask spread
- **Technical:** The difference between the highest price a buyer will pay (bid) and the lowest price a seller will accept (ask). Buying at the ask and selling at the bid loses the spread.
- **Intuition:** The main real-world "trading fee", even when brokers charge no commission.

### No-trade band
- **Technical:** A rule: only rebalance when `|delta − position| > band`; otherwise leave the position as it is. Under proportional costs, the theoretically optimal policy has this shape.
- **Intuition:** "Don't bother fixing small errors; fixing costs more than they do."
- **In the code:** `delta_band` strategy, `params.band` and `params.to_edge`.

### Unwind / liquidate
- **Technical:** Closing a position by trading it back to zero.
- **Intuition:** At expiry, we sell (or buy back) all hedge shares so we end holding only cash.
- **In the code:** the `if done:` block in `HedgingEnv.step`. It also pays transaction cost.

### Cash account
- **Technical:** All the money flows of the strategy: premium in, share purchases out, share sales in, costs out, plus interest. Can be **negative**, which means we've borrowed to buy shares (and pay interest on it).
- **Intuition:** Our bank balance for this trade.
- **In the code:** `obs.cash`, `env.cash`.

### Mark-to-market (MTM); mark-to-model
- **Technical:** Valuing positions at current market prices instead of what was paid. When there's no market price (our synthetic option), we use a model price (Black-Scholes), which is technically *mark-to-model*.
- **Intuition:** "What would this be worth if I had to value it today?"
- **In the code:** `obs.option_price` is the model value; `obs.pnl` is the MTM P&L.

### Hedging error
- **Technical:** The final P&L of a hedged short option. In the ideal Black-Scholes world it's 0, so any deviation is "error". Some methods minimize mean squared hedging error `E[P&L²] = Var[P&L] + (E[P&L])²`, which literally targets 0 and treats profit as error too. That's the *replication* view. We use the risk-adjusted view instead (see §1), where profit counts as good.
- **Intuition:** How much the hedge failed to cancel the option, in dollars.

---

## 7. Backtesting and metrics

### Backtest
- **Technical:** Running a strategy on historical data as if it had been trading at the time, using only information available at each moment.
- **Intuition:** "What would have happened if we'd used this strategy over 2005–2025?"

### Look-ahead bias
- **Technical:** Accidentally using information from the future in a backtest (e.g. using tomorrow's price to decide today). It makes results look unrealistically good.
- **Intuition:** Cheating by peeking. The `Observation` only contains present/past info, and realized vol uses a *trailing* window, to avoid this.

### Synthetic option
- **Technical:** An option that never traded; we compute its price with Black-Scholes from the real stock price and a vol estimate, because Yahoo doesn't store historical option prices.
- **Intuition:** "Imagine we sold an option on this date at a fair-looking price; would our hedge have held up on what *actually* happened next?"

### Episode
- **Technical:** One option's full life: sell at step 0, hedge at steps 0..N−1, settle at step N.
- **Intuition:** One "game". The backtest plays hundreds of them on different date ranges.
- **In the code:** the `Episode` dataclass, built by `make_episodes`.

### Step
- **Technical:** One time increment (one trading day for `"1d"` data). The strategy acts once per step.
- **In the code:** `obs.step`, `HedgingEnv.step(...)`.

### Stride; overlapping episodes
- **Technical:** Number of steps between consecutive episode start dates. If stride < maturity, episodes overlap in time.
- **Intuition:** Overlap gives more episodes, but neighbouring episodes share most of their price path, so they're not independent samples (results look more precise than they are).
- **In the code:** `episodes.stride`.

### Rescaling to $100
- **Technical:** Each episode's prices are divided by its starting price and multiplied by 100.
- **Intuition:** SPY was $120 in 2005 and $600 in 2025; P&L in raw dollars wouldn't be comparable. After rescaling, every P&L number means "$ per option on $100 of stock", which also reads as "% of the stock price".

### Mean P&L (`mean_pnl`)
- **Technical:** Average final P&L across episodes.
- **Intuition:** Does the strategy make or lose money on average?

### Standard deviation of P&L (`std_pnl`)
- **Technical:** The spread of final P&L across episodes.
- **Intuition:** The main measure of hedging quality: lower = more predictable = better hedge.

### CVaR / Expected Shortfall (`cvar_5%`)
- **Technical:** *Conditional Value-at-Risk* at 5% = the average P&L over the worst 5% of episodes.
- **Intuition:** "When things go badly, how badly on average?" Focuses on the tail (rare bad outcomes), which std can underweight. A common objective in deep hedging research.

### Worst (`worst`)
- **Technical:** The single lowest final P&L across episodes.
- **Intuition:** The worst case seen. Noisy, but scary numbers matter.

### Trades / turnover (`mean_trades`, `turnover`)
- **Technical:** `n_trades` = how many steps had a nonzero trade. `turnover` = total shares traded (sum of |trade|).
- **Intuition:** How busy the strategy is; this drives transaction costs.

---

## 8. Worked example: one tiny episode, by hand

Setup: we sell an ATM call (K = 100) with **3 days** to expiry. We price it with σ = 20% vol, r = 0, and pay 10 bps costs. We delta-hedge daily. The stock path is `100 → 102 → 99 → 103`. (Those are *big* daily moves, about 47% annualized vol, far more than the 20% we priced with. Watch what that does.)

Black-Scholes values along the path:

| step | spot | option value | delta |
|---|---|---|---|
| 0 | 100 | 0.87 | 0.50 |
| 1 | 102 | 2.12 | 0.87 |
| 2 | 99 | 0.15 | 0.21 |
| 3 (expiry) | 103 | 3.00 (= payoff: 103 − 100) | — |

Each step's P&L change (= the env's `reward`) is:
```
reward = (gain on shares) − (increase in the value of the option we owe) − (trading cost)
```

| step | what we do | gain on shares | option value change | cost | reward |
|---|---|---|---|---|---|
| 0 → 1 | buy 0.50 shares at 100 | 0.50 × (+2) = **+1.01** | 0.87 → 2.12: we owe **1.25** more | 0.05 | **−0.29** |
| 1 → 2 | buy 0.36 more (now 0.87) at 102 | 0.87 × (−3) = **−2.61** | 2.12 → 0.15: we owe **1.97** less | 0.04 | **−0.67** |
| 2 → 3 | sell 0.65 (now 0.21) at 99, then unwind at 103 | 0.21 × (+4) = **+0.86** | 0.15 → 3.00: we owe **2.85** more | 0.09 | **−2.08** |
| **total** | | | | **0.17** | **−3.04** |

(Rounded to 2 decimals, so a row can be off by 0.01.)

Cross-check with the "final" formula:
```
final P&L = premium − payoff + share gains − costs
          = 0.87 − 3.00 + (1.01 − 2.61 + 0.86) − 0.17
          = −3.04
```

**What happened, intuitively:**
1. The stock rose → delta went up → we **bought** more shares at 102 (high).
2. Then it fell → delta went down → we **sold** shares at 99 (low).
3. Then it rose again to 103 → the option finished ITM and we held few shares.

That's the **short gamma** effect: delta hedging a short option means buying after rises and selling after falls. The premium (0.87) was priced for 20% vol, which would cover those whipsaw losses *on average* if the stock really moved at 20% vol. The stock actually moved at ~47% vol, so the premium was far too small, and we lost 3.04.

Note: not hedging at all would have lost only `0.87 − 3.00 = −2.13` here. Hedging does **not** win every episode. It makes the *distribution* of outcomes across many episodes narrower. That's why we judge strategies on hundreds of episodes, not one.

**Reproduce it:**
```bash
uv run python -c "
import numpy as np
from backtest import HedgingEnv, Episode
from strategies import STRATEGIES
ep = Episode('example', np.array([100., 102., 99., 103.]), np.full(4, 0.2))
env = HedgingEnv('call', moneyness=1.0, rate=0.0, cost_bps=10)
strat = STRATEGIES['delta_hedge']()
obs, done = env.reset(ep), False
while not done:
    obs, reward, done, _ = env.step(strat.act(obs))
    print(f'step {obs.step}: reward {reward:+.3f}, pnl {obs.pnl:+.3f}')
"
```
