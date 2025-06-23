# Bitcoin-Market-Sentiment-Hyperliquid-Trader

Exploratory data science project that investigates how the Crypto Fear & Greed Index affects real‑world trader behaviour and profitability on Hyperliquid. It blends rigorous EDA, risk‑adjusted metrics, predictive modelling, lag analysis, trader clustering, and a simple sentiment‑driven back‑test.

Run this code
```jupyter lab trader_sentiment_analysis_full.ipynb```

## Analysis Workflow (Notebook Outline)

1) Data quality & preprocessing — date alignment, missing‑value heat‑map, outlier flagging.
2) Risk‑adjusted metrics — mean, stdev, Sharpe‑like score per sentiment bucket.
3) Win‑rate vs payoff profile — win‑rate, average win/loss, payoff ratio visualised.
4) Predictive modelling — logistic regression (sentiment + features ⇒ profitable trade), ROC‑AUC & coefficients.
5) Lagged causality — Granger tests & cross‑correlation plots (sentiment ➜ next‑day PnL).
6) Trader segmentation — K‑means clustering on wallet‑level statistics, silhouette score, cluster insights.
7) Back‑test — rule‑based strategy: long on Extreme Fear ≤ 25, flat on Extreme Greed ≥ 75, equity curve, CAGR & max‑DD.
8) Feature Engineering – Lagged & Rolling Sentiment Metrics

## Key Findings

1) Greed days deliver higher mean PnL but double volatility; risk‑adjusted returns favour moderated exposure.
2) Sentiment alone is a weak win/loss predictor (AUC ≈ 0.55); adding size & intraday timing improves edge.
3) Five contrarian wallets outperform on Fear days—ideal for copy‑trade research.
4) No strong lead‑lag effect within ±3 days; sentiment acts concurrently rather than predictively.
5) Simple Extreme‑Fear long rule beats flat baseline (positive CAGR) but suffers ⟂10–15 % drawdowns.

## Hidden Patterns

| Cluster        | Traits                                | Δ PnL (Greed‑Fear) | Action |
| -------------- | ------------------------------------- | ------------------ | ------ |
| Momentum Bulls | High leverage,greed‑heavy, boom‑bust. | + $420             |   Scale down size during Extreme Greed.    | 
| Steady Makers  | Low variance, sentiment‑neutral.      | + $35              |   Prime copy‑trade base.                   |
| Contrarians    | Profit more in Fear, often short‑bias.| – $280             |    Mirror their trades for reversal alpha. |

## Core Relationships

| **Finding**                                         | **Evidence**                                                         | **Why it matters**                                              |
| --------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------- |
| Greed days boost average PnL—but double volatility. | Mean PnL ↑ +76 %; st-dev ↑ +105 %; Welch *t*-test *p* < 10⁻⁸         | Higher raw gains but thinner risk-adjusted edge → tighten stops |
| Sentiment moves magnitude, not win-rate.            | Logistic AUC ≈ 0.51 with sentiment only; 0.58 with size + hour       | Adjust **sizing**, not directional bias                         |
| Volume surges on Fear days.                         | 4× more trades; +60 % book depth                                     | Ideal liquidity windows for market-making & scalps              |
| Engineered features predict volatility spikes.      | `value_z7` corr 0.42 with next-day PnL st-dev; `price_vol7` stronger | Plug into VAR / volatility-targeting modules                    |
