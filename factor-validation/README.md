# Factor Validation: Published Commodity Strategies, Purged CV and Overfitting Checks

> **The question:** do published commodity factor strategies survive honest out-of-sample testing, or is the reported Sharpe ratio the best of many lucky tries?

A research-validation workflow in Python: build a 46-year monthly commodity panel, rebuild three published strategies with their papers' settings, measure how they did before and after publication, then test whether choosing parameters after seeing the results would have fooled you.

---

## Table of contents

- [Overview](#overview)
- [How to use and run](#how-to-use-and-run)
- [Data](#data)
- [Workflow steps in plain language](#workflow-steps-in-plain-language)
- [Outputs](#outputs)
- [Notation and key definitions](#notation-and-key-definitions)
- [Strategies and validation methods](#strategies-and-validation-methods)
- [How to explain this project](#how-to-explain-this-project)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

| Strategy | Paper | Signal | Portfolio |
|---|---|---|---|
| **Time-series momentum (TSMOM)** | Moskowitz, Ooi & Pedersen (2012, *JFE*) | Sign of the past 12-month return (skip 1) | Long rising, short falling; each position sized to 40% annual vol |
| **Cross-sectional momentum (XSMOM)** | Miffre & Rallis (2007, *JBF*) | Past 12-month return, ranked across commodities | Long the top third, short the bottom third |
| **Value** | Asness, Moskowitz & Pedersen (2013, *JF*) | log(average price 4.5–5.5 years ago / price now) | Long the cheapest third, short the dearest third |

| Setting | Value | Where to change it (`config.py`) |
|---|---|---|
| Universe | 26 commodities with a liquid futures analogue | `UNIVERSE` |
| Sample | 1980-01 onward; a month trades once ≥ 10 commodities have prices | `FACTOR_START`, `MIN_ASSETS` |
| Trading cost | 10 bp per unit of turnover | `COST_BPS` |
| Research grid | 46 variants: look-backs 1–12m, skip 1–2m, vol-scaling, terciles/quartiles, 3–5y value | `GRID` |
| CPCV | 10 groups, 2 held out → 45 splits, 9 paths; purge 1 month, embargo 12 months | `CPCV_*`, `PURGE`, `EMBARGO` |
| PBO | 16 blocks → 12,870 half/half splits | `PBO_BLOCKS` |

---

## How to use and run

```bash
bash setup.sh             # once, from the repo root
bash run.sh factors       # this workflow only (under a minute)
```

Or one step at a time: `.venv/bin/python factor-validation/01_data.py`, then `02_replicate.py`, `03_purged_cv.py`, `04_overfitting.py`, `05_report.py`.

**Tips**
- **Regime link:** if the regime-risk workflow has been run first, step 2 also reports each strategy's performance in calm vs turbulent energy regimes (`run.sh` runs it first).
- **The Pink Sheet file** refreshes at most weekly. If the World Bank link has moved, the GitHub mirror is used; you can also download `CMO-Historical-Data-Monthly.xlsx` from the [Pink Sheet page](https://www.worldbank.org/en/research/commodity-markets) and save it in `data/raw/`.

---

## Data

| Source | Content | Frequency | Quirks handled |
|---|---|---|---|
| World Bank Commodity Price Data ("Pink Sheet"), `Monthly Prices` sheet | Nominal USD prices for ~70 commodities since 1960 | Monthly **averages** | Pre-1980 administered prices (flat for months) excluded; stale runs and 40%+ monthly moves flagged |

**Why every signal skips a month.** The Pink Sheet reports the *average* price within each month. Averaging a random walk creates positive autocorrelation of about 0.25 between consecutive monthly changes (Working, 1960), and none at lag 2. The data shows exactly that: median lag-1 autocorrelation 0.26, lag-2 0.01. A signal that used last month's return would harvest this artefact, so every signal ends one month before the position is taken.

---

## Workflow steps in plain language

### Step 1: Build the panel (`01_data.py`)
1. Download the Pink Sheet, keep the 26 commodities in `UNIVERSE`, and start in 1980.
2. Turn prices into monthly log returns; flag stale prices and extreme moves.
3. Measure lag-1 and lag-2 autocorrelation to confirm the averaging artefact.

### Step 2: Replicate the papers (`02_replicate.py`)
1. Build each strategy's monthly positions with the paper's settings. Positions are set at month end and earn next month's return.
2. Subtract trading costs from turnover; report gross and net performance with Newey–West t-statistics.
3. Split history into three windows (McLean & Pontiff, 2016): inside the paper's sample, after the sample but before publication, and after publication.
4. Combine value and momentum 50/50, and split performance by the WTI regime from the regime-risk workflow.

### Step 3: Choose parameters honestly (`03_purged_cv.py`)
1. Build all 46 variants in the research grid.
2. **Combinatorial purged cross-validation:** split the months into 10 blocks. For every way of holding out 2 blocks (45 ways), pick the variant with the best Sharpe on the other 8, and record how it did on the 2 held out. Drop 1 month before and 12 months after each held-out block, so training months never see prices from the test block through their look-back windows.
3. Stitch the held-out pieces into 9 complete out-of-sample histories, and compare their Sharpe ratios with the in-sample best.
4. Repeat without purging (to see what leakage does), with an anchored walk-forward, and within each strategy family.

### Step 4: Measure overfitting (`04_overfitting.py`)
1. **PBO:** split the months into 16 blocks; for each of the 12,870 ways to pick 8 as "in-sample", find the in-sample winner and its rank in the other 8. PBO is the share of splits where the winner ranks below the median.
2. **Deflated Sharpe ratio:** the probability that the true Sharpe beats the best Sharpe you'd expect from N worthless variants, allowing for skew and fat tails.
3. Compare Newey–West t-statistics with the t > 3 hurdle proposed by Harvey, Liu & Zhu (2016).

### Step 5: Write the report (`05_report.py`)
Collects every table and chart into `outputs/factor_validation_summary.md` and `reports/factor_validation_summary.pdf`.

---

## Outputs

| File | What it shows |
|---|---|
| `outputs/factor_validation_summary.md` | **Start here.** The full report (PDF in `../reports/`) |
| `tables/01_universe.csv`, `01_data_quality_flags.csv`, `01_data_pull_log.csv` | Universe, volatility, autocorrelation, flags, provenance |
| `tables/02_published_performance.csv` | Gross and net performance of each strategy and the combination |
| `tables/02_decay_windows.csv` | In-sample, post-sample and post-publication performance |
| `tables/02_strategy_monthly_returns.csv`, `02_strategy_correlations.csv` | Monthly returns and correlations |
| `tables/02_performance_by_regime.csv` | Performance in calm vs turbulent energy regimes |
| `tables/03_grid_performance.csv` | All 46 variants |
| `tables/03_cv_results.csv`, `03_cpcv_path_sharpes.csv`, `03_walk_forward_picks.csv` | Out-of-sample selection results |
| `tables/04_pbo.csv`, `04_deflated_sharpe.csv` | Overfitting diagnostics |
| `figures/*.png` | Charts, numbered by step |

---

## Notation and key definitions

| Term | Plain-language definition |
|---|---|
| $r_{i,t}$ | Log return of commodity $i$ in month $t$. |
| $\text{MOM}_{i,t}(L, s)$ | Sum of returns over months $t-s-L+1$ to $t-s$: an $L$-month look-back that skips the last $s$ months. "12-1" means $L=12$, $s=1$. |
| **Vol-scaling** | Sizing each position to a target volatility ($w = 0.40/\sigma_i$), so quiet and wild markets contribute equal risk. |
| **Turnover** | Sum of absolute changes in weights each month; costs are charged on it. |
| **Sharpe ratio** $SR$ | Average return over its standard deviation, annualised ×√12. |
| **Newey–West t** | The t-statistic of the average return, robust to autocorrelation. |
| **Publication decay** | The fall in performance after a strategy is published (McLean & Pontiff find ~58% for US stock anomalies). |
| **Purging** | Removing training observations whose information overlaps the test period. |
| **Embargo** | Also removing training observations just *after* each test block, because their look-back windows reach into it. |
| **CPCV** | Cross-validation that holds out every combination of $k$ of $N$ blocks, yielding many complete out-of-sample paths instead of one. |
| **PBO** | Probability that the variant that looks best in-sample ranks below median out of sample. 0% = selection is reliable; ≥ 50% = selection is no better than chance. |
| **PSR** | Probability that the true Sharpe exceeds a benchmark, given sample length, skew and kurtosis. |
| **Luck bar** $SR_0$ | The Sharpe the best of $N$ zero-skill variants would be expected to show, given how much their Sharpes vary. |
| **Deflated Sharpe ratio (DSR)** | PSR measured against $SR_0$: the probability the strategy beats luck after charging for every variant tried. A common pass bar is 0.95. |
| **t > 3 hurdle** | Harvey, Liu & Zhu's suggested significance bar once hundreds of factors have been tried. |

---

## Strategies and validation methods

| Method | What it guards against | Result here |
|---|---|---|
| Skip-month signals | Averaging artefact in monthly data | Lag-1 autocorrelation 0.26 neutralised |
| No-look-ahead test (`tests/`) | Signals using future prices | Weights before month t are unchanged when prices after t are altered |
| Placebo (200 shuffles) | Structural bias in the portfolio code | Market-neutral placebos centre near zero; value's real Sharpe sits far in the tail (p ≈ 0.005). TSMOM's placebo keeps its net tilt and earns 0.38 vs the real 0.44: its edge here is mostly timing the whole complex |
| McLean–Pontiff windows | Publication decay | Value and TSMOM hold up after publication |
| Purged CPCV | Selecting parameters with hindsight | Out-of-sample within 0.11 of in-sample in every family |
| PBO | The in-sample winner being a fluke | 37% for TSMOM, 25% XSMOM, 19% value |
| Deflated Sharpe | Multiple testing | Value survives (DSR 1.000); TSMOM 0.85 falls short of 0.95 |

---

## How to explain this project

> "I rebuilt three published commodity strategies on 46 years of World Bank prices and asked whether they'd survive a skeptical validator. The first thing I found was a data artefact: monthly average prices create 0.26 lag-1 autocorrelation, so every signal skips a month. Value survives everything: Sharpe 0.95 net, t of 5.5, and it held up after publication. Time-series momentum is positive but modest, and when I try 20 versions of it, there's a 37% chance the best-looking one is just lucky. Cross-sectional momentum doesn't replicate on spot data. To choose parameters honestly I used combinatorial purged cross-validation with an embargo, then the probability of backtest overfitting and the deflated Sharpe ratio, which charges a strategy for every variant tried."

**Follow-up questions to prepare for:**
- **Why purge and embargo?** A training month right after a test block has a look-back window that includes test-period prices; without the embargo the model is partly trained on the test set.
- **Why did purging change so little here?** The only thing "fitted" is the choice among a few rules, and rankings are stable. For fitted models on overlapping labels (e.g. gradient boosting on 12-month forward returns) purging matters much more.
- **Why is the deflated Sharpe across all 46 variants so harsh?** The formula treats the spread of Sharpes as luck; here it is partly genuine (value really is better than XS momentum), so SR0 is inflated. Deflating within a family is the fairer test.
- **Why doesn't XS momentum work here when the paper found it?** Different data: spot averages vs futures returns, no roll yield, a different universe and period. It is a spot-price analogue, not a replication of the paper's exact portfolio.
- **Is value's Sharpe of 0.95 tradable?** Probably not in full: spot prices mean-revert more than futures returns, because the futures curve already prices expected reversion.

---

## Limitations

- **Spot, monthly-average prices** instead of futures excess returns; no roll yield, so no carry strategy.
- **Flat costs** of 10 bp per unit of turnover; real costs vary by market and era.
- **N is a lower bound.** The literature tried far more variants than this grid.
- **Few turbulent months** in the regime split (18): descriptive only.

---

## References

**Strategies**
- Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*, 104(2), 228–250.
- Miffre, J. & Rallis, G. (2007). Momentum strategies in commodity futures markets. *Journal of Banking & Finance*, 31(6), 1863–1886.
- Asness, C., Moskowitz, T. & Pedersen, L. H. (2013). Value and momentum everywhere. *Journal of Finance*, 68(3), 929–985.
- McLean, R. D. & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*, 71(1), 5–32.

**Validation**
- Bailey, D. & López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk*, 15(2), 3–44.
- Bailey, D. & López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management*, 40(5), 94–107.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39–69.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 7 (purged CV), 11 (backtesting dangers), 12 (CPCV).
- Harvey, C., Liu, Y. & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- Newey, W. & West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Working, H. (1960). Note on the correlation of first differences of averages in a random chain. *Econometrica*, 28(4), 916–918.

**Data**
- World Bank (2026). Commodity Price Data (The Pink Sheet): monthly historical prices.
