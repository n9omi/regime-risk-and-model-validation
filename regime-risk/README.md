# Regimes and Risk: Markov Switching, Volatility Forecasts and VaR/ES Backtests

> **The question:** which regime is the energy market in, which volatility model forecasts best, and can we trust the VaR and ES built on top of them?

A daily risk workflow in Python: pull EIA spot prices, clean them, detect calm and turbulent regimes, forecast tomorrow's volatility four ways, forecast VaR and ES five ways, and backtest everything out of sample with the tests regulators and model validators use.

---

## Table of contents

- [Overview](#overview)
- [How to use and run](#how-to-use-and-run)
- [Data](#data)
- [Workflow steps in plain language](#workflow-steps-in-plain-language)
- [Outputs](#outputs)
- [Notation and key definitions](#notation-and-key-definitions)
- [Models compared](#models-compared)
- [Validation and regulatory context](#validation-and-regulatory-context)
- [How to explain this project](#how-to-explain-this-project)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

Four return series are modelled: three EIA benchmarks, each treated as a $1mm long "desk", and an energy book of constant dollar positions.

| Series | What it is | Position |
|---|---|---|
| WTI | Crude oil spot, Cushing, Oklahoma (`DCOILWTICO`) | $1mm long (desk) · +$4mm in the book |
| Brent | Crude oil spot, Europe (`DCOILBRENTEU`) | $1mm long (desk) · −$2mm in the book (a WTI–Brent spread) |
| Henry Hub | Natural gas spot, Louisiana (`DHHNGSP`) | $1mm long (desk) · +$2mm in the book |
| Book | All three positions together, return = P&L / $8mm gross | modelled on its own P&L series |

| Setting | Value | Where to change it (`config.py`) |
|---|---|---|
| Sample | 2007-01-02 onward (full daily coverage for all three series) | `START_DATE` |
| First out-of-sample forecast | 2010-01-04 (after ~3 years of training data) | `BACKTEST_START`, `MIN_TRAIN` |
| VaR / ES confidence | 99% VaR, 97.5% ES, 1-day | `VAR_CONF`, `ES_CONF` |
| Historical simulation window | 250 trading days | `HS_WINDOW` |
| EWMA decay | λ = 0.94 (RiskMetrics) | `EWMA_LAMBDA` |
| Re-estimation | GARCH / HAR monthly (21 days); regimes every 126 days | `REFIT_EVERY`, `MS_REFIT_EVERY` |
| Regimes | 2 used downstream; 2 and 3 compared by BIC | `MS_STATES` |

---

## How to use and run

```bash
bash setup.sh            # once, from the repo root
bash run.sh risk         # this workflow only (about 4 minutes)
```

Or one step at a time, in order:

```bash
.venv/bin/python regime-risk/01_data.py
.venv/bin/python regime-risk/02_regimes.py
.venv/bin/python regime-risk/03_volatility.py
.venv/bin/python regime-risk/04_var_es_backtest.py
.venv/bin/python regime-risk/05_report.py
```

**Tips**
- **Scripts hand off through files** in `data/processed/*.pkl`, so any single step can be re-run after one full run.
- **Change assumptions only in `config.py`.**
- **Force a fresh download:** delete `data/raw/`. Otherwise data refreshes at most once a day.
- **If FRED is unreachable** the scripts fall back to a GitHub mirror of the same EIA series and record the source in `01_data_pull_log.csv`.

---

## Data

| Series | Source | Frequency | Quirks handled |
|---|---|---|---|
| WTI, Brent, Henry Hub spot | EIA via FRED; GitHub mirror as fallback | Daily (trading days) | Holidays differ between the US and Europe; the April 2020 negative WTI print; gas cold-snap spikes |

**Data-quality rules** (every action is logged to `outputs/tables/01_*.csv`)

| Check | Rule | Action |
|---|---|---|
| Non-positive price | Price ≤ 0 (WTI spot, 2020-04-20) | Set to missing: a log return is undefined |
| Short gap | 1–2 missing days | Carry the last price forward |
| Longer gap | Any series still missing | Drop the date; the next return spans the gap |
| Stale price | 3+ days of exactly 0% change | Flag |
| Extreme move | Larger than 6 × the rolling 1-year SD | **Flag, not delete.** Crashes and spikes are what risk models must survive |
| Thin coverage | Fewer than 200 common trading days in a year | Flag |

---

## Workflow steps in plain language

### Step 1: Get and clean the data (`01_data.py`)
1. Download the three daily series (FRED first, mirror second), cache the raw files untouched.
2. Put them on one calendar, apply the data-quality rules, and log every change.
3. Turn prices into daily log returns (%), and build the book's daily P&L from constant dollar positions.
4. Summarise volatility, skew and fat tails; chart prices and book P&L.

### Step 2: Find the regimes (`02_regimes.py`)
1. Fit a Markov-switching model to each series: every day is drawn from a "calm" or a "turbulent" normal distribution, and the market switches between them with fixed probabilities.
2. Estimate by EM: the **Hamilton filter** runs forward through the data, the **Kim smoother** runs backward, and the parameters are updated until the likelihood stops improving.
3. Compare two and three regimes by BIC.
4. **Check the estimator:** simulate data from the fitted model, re-estimate, and confirm the parameters and hidden states come back.
5. Re-estimate every six months on data up to that date and run the filter forward, giving a **real-time** regime probability with no look-ahead.
6. Record today's regime, and list the longest turbulent episodes.

### Step 3: Forecast volatility four ways (`03_volatility.py`)
1. Every day from 2010, forecast the next day's variance with EWMA, GARCH(1,1)-t, HAR-RV and the regime mixture, using only the past.
2. Score each forecast with QLIKE and MSE against the squared return, and test the gaps with Diebold–Mariano.
3. Check calibration with Mincer–Zarnowitz regressions, and check the GARCH estimator by simulate-and-refit.

### Step 4: Forecast and backtest VaR and ES (`04_var_es_backtest.py`)
1. Every day, forecast 99% VaR, 97.5% VaR and 97.5% ES with five models (see [Models compared](#models-compared)).
2. Mark every **exception**: a day the loss beat VaR.
3. Test each model: right number of exceptions (Kupiec), no clustering (Christoffersen), both (conditional coverage), Basel zone, ES adequacy (Z2, McNeil–Frey).
4. Split exception rates by the regime the model knew about at forecast time.
5. Convert today's forecasts to dollars for each desk and the book.

### Step 5: Write the report (`05_report.py`)
Collects every table and chart into `outputs/regime_risk_summary.md` and `reports/regime_risk_summary.pdf`. Sentences with numbers are generated from the tables.

---

## Outputs

| File | What it shows |
|---|---|
| `outputs/regime_risk_summary.md` | **Start here.** The full report (PDF in `../reports/`) |
| `tables/01_data_pull_log.csv`, `01_data_quality_fixes.csv`, `01_flagged_moves.csv` | Provenance and data quality |
| `tables/01_summary_stats.csv`, `01_correlations.csv` | Return statistics |
| `tables/02_ms_parameters.csv`, `02_model_selection.csv` | Regime parameters; 2 vs 3 regimes by BIC |
| `tables/02_simulation_recovery.csv` | Simulate-and-refit check of the regime estimator |
| `tables/02_current_regime.csv`, `02_turbulent_episodes.csv` | Today's regime; turbulent episodes |
| `tables/03_forecast_losses.csv`, `03_diebold_mariano.csv`, `03_mincer_zarnowitz.csv` | Volatility forecast scores and tests |
| `tables/03_full_sample_params.csv`, `03_garch_params_path.csv`, `03_har_params_path.csv` | Parameters, full sample and through time |
| `tables/03_garch_recovery.csv` | Simulate-and-refit check of the GARCH estimator |
| `tables/04_backtest_results.csv` | Every test statistic, p-value, zone and verdict |
| `tables/04_var_es_today.csv` | Next-day VaR and ES in dollars |
| `tables/04_exceptions_by_regime.csv` | Exception rates in calm vs turbulent regimes |
| `tables/04_book_daily_forecasts.csv` | Every daily forecast for the book vs realised return |
| `figures/*.png` | Charts, numbered by the step that made them |

---

## Notation and key definitions

| Term | Plain-language definition |
|---|---|
| **Regime** $s_t$ | The market's hidden "state" on day $t$: calm (low volatility) or turbulent (high volatility). |
| **Transition probability** $P_{ij}$ | The chance of moving from regime $i$ today to regime $j$ tomorrow. A calm regime with $P_{11} = 0.99$ lasts on average $1/(1-0.99) = 100$ days. |
| **Filtered probability** $\Pr(s_t \mid r_1,\dots,r_t)$ | The regime probability using data up to today: what a trader knows in real time. |
| **Smoothed probability** $\Pr(s_t \mid r_1,\dots,r_T)$ | The regime probability using the whole sample: the best reading with hindsight. Never used for forecasting. |
| **EM algorithm** | Alternates between "guess the regimes given the parameters" and "fit the parameters given the regimes" until nothing changes. |
| **BIC** | Likelihood penalised for the number of parameters; lower is better. |
| **Conditional variance** $h_t$ | The forecast of day $t$'s variance made at the previous close. |
| **GARCH(1,1)** | $h_t = \omega + \alpha e_{t-1}^2 + \beta h_{t-1}$: today's variance is a weighted mix of a long-run level, yesterday's shock and yesterday's variance. **Persistence** $\alpha+\beta$ near 1 means shocks fade slowly. |
| **Student-t errors** ($\nu$) | Fat-tailed shocks; lower $\nu$ = fatter tails. $\nu$ near 4 (Henry Hub, the book) means extreme days are far more common than a normal distribution allows. |
| **HAR-RV** | Forecasts variance from its average over the last day, week and month: a simple way to capture long memory. |
| **QLIKE** | $\log h_t + r_t^2/h_t$. A loss function that ranks variance forecasts correctly even when the "truth" is only the noisy squared return. |
| **Diebold–Mariano test** | Is one forecast's average loss significantly lower than another's? Uses a heteroskedasticity- and autocorrelation-robust variance. |
| **Mincer–Zarnowitz regression** | Regress the outcome on the forecast; an unbiased forecast has intercept 0 and slope 1. |
| **Value-at-Risk (VaR)** | The loss that should be exceeded on only 1% of days (99% VaR). |
| **Expected Shortfall (ES)** | The average loss on the days worse than VaR at 97.5%: "when it's bad, how bad?" FRTB's measure. |
| **Exception** | A day with a loss larger than that day's VaR forecast. Expected: 1% of days. |
| **Kupiec POF test** | A likelihood-ratio test of whether the exception rate equals 1%. |
| **Christoffersen test** | Tests whether an exception today makes one tomorrow more likely (clustering). |
| **Conditional coverage** | Kupiec and Christoffersen together; a model **passes** here when neither Kupiec nor this test rejects at 5%. |
| **Basel traffic light** | Exceptions in the last 250 days of 99% VaR: green 0–4, yellow 5–9, red 10+. |
| **Acerbi–Szekely Z2** | An ES backtest; 0 if ES is right, negative if losses beyond VaR are bigger than ES says. |
| **McNeil–Frey test** | On exception days, is the loss beyond ES bigger than zero on average? Bootstrap p-value. |

---

## Models compared

**Volatility (step 3)**

| Model | How it works | Strength | Weakness |
|---|---|---|---|
| **EWMA** | Exponentially weighted average of squared returns, λ = 0.94 | No estimation; reacts in days | No mean reversion to a long-run level |
| **GARCH(1,1)-t** | Maximum likelihood with fat-tailed errors, re-fitted monthly | Mean reversion + fat tails; best QLIKE here | Parameters drift; symmetric response to up and down moves |
| **HAR-RV** | OLS on daily, weekly and monthly average squared returns | Long memory; transparent | Built for intraday realised variance; with daily closes the proxy is noisy |
| **Markov-switching** | Mixture variance weighted by predicted regime probabilities | Jumps between volatility levels | Only two volatility levels; real-time probabilities are noisy |

**VaR and ES (step 4)**

| Model | How it works | Strength | Weakness |
|---|---|---|---|
| **Historical simulation** | 2nd-worst of the last 250 returns | No distribution assumption | Regime-blind: reacts slowly, then drops a crisis abruptly after 250 days |
| **Filtered HS** (Hull & White, 1998) | Rescale past returns to today's EWMA volatility, then HS | Real tail shape at today's volatility | More moving parts |
| **GARCH-t** | Student-t quantile scaled by the GARCH forecast | Fat tails + volatility clustering | Tail shape fixed between re-fits |
| **HAR-RV normal** | Normal quantile scaled by the HAR forecast | Simple | Normal tails are too thin: included to show the cost |
| **Markov-switching** | Quantile of the predicted two-regime normal mixture | Captures regime jumps | Gaussian within each regime; noisy filter |

---

## Validation and regulatory context

| Framework | What it asks | Where it shows up here |
|---|---|---|
| **SR 11-7** (Fed/OCC model risk guidance) | Conceptual soundness, benchmarking, outcomes analysis, ongoing monitoring | Simulate-and-refit (02, 03), Diebold–Mariano vs a benchmark (03), backtests (04), traffic light (04); summarised in section 5 of the report |
| **Basel backtesting framework** (BCBS, 1996) | 250 days of 99% VaR exceptions in green / yellow / red zones | `04_backtest_results.csv`, traffic-light chart |
| **FRTB** (BCBS, 2019, MAR30–33) | 97.5% ES, backtesting at 99% and 97.5% | ES at 97.5%, Z2 and McNeil–Frey |

---

## How to explain this project

> "I built an out-of-sample risk workflow on daily energy benchmarks. First I fit a Markov-switching model to find calm and turbulent regimes, and I verify the EM estimator by simulating from the fitted model and recovering it. Then I forecast volatility with EWMA, GARCH-t, HAR and the regime mixture, and rank them with QLIKE and Diebold–Mariano: GARCH-t wins on every series. Finally I forecast VaR and ES five ways and backtest them with Kupiec, Christoffersen, the Basel traffic light and Z2. The most useful finding is the regime split: plain historical simulation looks fine on average but is beaten on 3% of turbulent days, three times the target, while filtering past returns by today's volatility fixes most of that."

**Follow-up questions to prepare for:**
- **Why QLIKE rather than MSE?** Both rank forecasts correctly with a noisy proxy (Patton, 2011), but MSE is dominated by a few huge days; QLIKE penalises under-prediction more, which is the dangerous error for risk.
- **Why can a model pass Kupiec and still fail?** The right number of exceptions can arrive in clusters; Christoffersen catches that. Historical simulation on the book is the example here.
- **Why does GARCH-t forecast variance best but fail VaR on crude?** The variance forecast is good, but the fitted Student-t tail is not heavy enough for crude's worst days: a good second moment is necessary for a good VaR, not sufficient.
- **Why does BIC prefer three regimes?** With Gaussian regimes, an extra state absorbs fat tails. I kept two for interpretability and documented it.
- **Smoothed vs filtered probabilities: which would you trade on?** Only filtered: the smoothed series uses future data.

---

## Limitations

- **Spot, not futures.** Henry Hub spot spikes during cold snaps (e.g. January 2024) are far larger than front-month futures moves; a tradable book would be marked on futures.
- **Daily closes only.** Realised variance is proxied by the squared return, which handicaps HAR-RV.
- **Gaussian regimes.** A Student-t Markov-switching model or MS-GARCH would fit the tails better.
- **Univariate book model.** The book is modelled on its own P&L series; correlations between legs change only implicitly.
- **Linear, constant-dollar positions.** No options, no liquidity horizons.

---

## References

**Regimes and volatility**
- Hamilton, J. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
- Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of Econometrics*, 60, 1–22.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31, 307–327.
- Bollerslev, T. (1987). A conditionally heteroskedastic time series model for speculative prices and rates of return. *Review of Economics and Statistics*, 69(3), 542–547.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246–256.
- Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- J.P. Morgan / Reuters (1996). *RiskMetrics: Technical Document* (4th ed.).

**VaR and ES backtesting**
- Kupiec, P. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73–84.
- Christoffersen, P. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841–862.
- Acerbi, C. & Szekely, B. (2014). Backtesting expected shortfall. *Risk*, December.
- McNeil, A. & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. *Journal of Empirical Finance*, 7, 271–300.
- Hull, J. & White, A. (1998). Incorporating volatility updating into the historical simulation method for value-at-risk. *Journal of Risk*, 1(1), 5–19.

**Regulation and guidance**
- BCBS (1996). Supervisory framework for the use of "backtesting" in conjunction with the internal models approach to market risk capital requirements.
- BCBS (2019). Minimum capital requirements for market risk (FRTB), Basel Framework MAR30–33.
- Board of Governors of the Federal Reserve System & OCC (2011). SR 11-7: Supervisory Guidance on Model Risk Management.

**Data**
- U.S. Energy Information Administration, spot prices, via Federal Reserve Bank of St. Louis [FRED](https://fred.stlouisfed.org/): `DCOILWTICO`, `DCOILBRENTEU`, `DHHNGSP`.
