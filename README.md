# Copula-GARCH Portfolio Optimization Across Market Regimes

Do asymmetric copulas actually build better portfolios than a simple correlation benchmark? This project tests that on 24 years of US industry returns and finds the answer is: only sometimes, and the exceptions are more interesting than the rule.

**Framework:** EGARCH(1,1) skew-t margins → copula-implied dependence → Global Minimum Variance optimization → rolling Diebold-Mariano tests against a CCC-GARCH benchmark.

---

## Data & Setup

- **Assets:** 10 industry value-weighted portfolios from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_10_ind_port.html)
- **Sample:** January 2000 – December 2023 (daily); out-of-sample results run from late 2003
- **Benchmark:** Constant Conditional Correlation (CCC) GARCH — the static Pearson correlation of GARCH residuals
- **Loss function:** squared out-of-sample portfolio return (variance loss)
- **Constraints:** short selling permitted (weight bounds ±1); a long-only variant is run separately for comparison

---

## Method

**1. Marginal filtering.** Each asset is fitted with an EGARCH(1,1) model with skew-t innovations, chosen for asymmetric volatility response and heavy tails. Fits run in parallel across assets. The model yields a one-step-ahead volatility forecast and standardized residuals, which are mapped to uniform margins by the probability integral transform.

**2. Dependence modeling.** Copulas are fitted to the uniform margins: Gaussian, Student-t, Clayton, Gumbel, their 180°-rotated counterparts, and a vine copula selected from a Clayton/Gumbel/t/Gaussian family set. Rotation is done by fitting to 1−u and inverting the simulated draws back.

**3. Covariance construction.** Each fitted copula generates 10,000 joint scenarios. The rank correlation of those scenarios becomes R, and the covariance matrix is assembled as Σ = D R D with GARCH volatility forecasts on the diagonal of D. Negative eigenvalues are repaired before optimization.

**4. Optimization.** Global Minimum Variance weights, minimizing wᵀΣw subject to weights summing to one.

**5. Dual rolling-window evaluation.**
- *Outer loop:* 1,000-day training window, 126-day evaluation horizon, rolled forward 126 days at a time with full re-estimation of every margin and copula.
- *Inner loop:* a 504-day rolling Diebold-Mariano test over the realized loss series. A single aggregate DM test can hide local predictability — strong crisis performance cancels against mild underperformance in a long bull market. The rolling version maps *when* each model wins.

---

## Findings

**Symmetric beats exotic, most of the time.** The Normal copula outperformed the CCC benchmark fairly consistently across the sample.

**Asymmetric copulas are regime-dependent.** Gumbel — which loads on upper-tail dependence — significantly outperformed symmetric models during the 2020 V-shaped recovery, when assets rallied together. In the calm 2006 to mid-2008 bull market, both Gumbel and Clayton *underperformed* CCC: with few extreme joint events to fit, the extra parameters bought estimation noise instead of insight.

**The short-selling paradox.** During the 2008 crash, standard Gumbel beat rotated Gumbel — despite rotated Gumbel being the model that correctly captures joint crashes. When the rotated copula identifies near-perfect downside correlation, an unconstrained mean-variance optimizer reads it as an arbitrage opportunity and takes large offsetting long/short positions. Realized correlations then deviate from the estimate and the leverage amplifies variance. Standard Gumbel's blindness to left-tail dependence flattened the correlation matrix and acted as an accidental regularizer.

The effect reverses under long-only constraints: with leverage unavailable, rotated Gumbel performs better, as theory predicts. The failure is a property of the optimizer, not the copula.

**Model failure as a diagnostic.** Because each copula family specializes — Clayton for left-tail crashes, Gumbel for right-tail rallies, Gaussian for elliptical behavior — tracking which model is currently winning the rolling DM test is itself a read on the prevailing regime.

---

## Repository

```
├── Copula_framwork.ipynb            # Main framework: fitting, optimization, backtest, DM tests
├── COPULA_report_educational.ipynb  # Copula theory walkthrough with worked bivariate examples
├── archive/                         # Earlier iterations
├── README.md
└── LICENSE                          # MIT
```

**Stack:** `arch` (EGARCH), `copulae` (Gaussian/t/Clayton/Gumbel), `pyvinecopulib` (vine), `PyPortfolioOpt` (GMV), `scipy`, `pandas`, `numpy`, `joblib`.

**Running it.** Training is gated behind `RUN_TRAINING = False` — a full backtest takes hours. Weights are written incrementally to CSV with a resume check, so an interrupted run picks up where it stopped.

---

## Limitations

Stated plainly, because they bound what the results mean.

1. **The copula is compressed into a rank-correlation matrix.** Simulating from the copula and taking the Spearman correlation of the draws collapses the dependence structure into a single matrix before it reaches the optimizer. Tail dependence — the property that distinguishes Clayton from Gumbel — does not survive that step, which limits how much the copula choice can move the portfolio. A direct scenario-based optimization over simulated joint returns would preserve it.
2. **Spearman's rho is used where a Pearson correlation is expected.** Σ = D R D assumes linear correlation. The standard elliptical conversion ρ = 2·sin(πρₛ/6) is not applied.
3. **Reproducibility gap.** The committed version of the optimization function has several copula families commented out, so a fresh run generates only a subset of the weight files the loss calculation expects. The reported results come from accumulated runs.
4. **Per-asset GARCH rescaling.** `arch_model(..., rescale=True)` applies an independent scaling factor per asset, so forecast variances are not on a common scale before entering D.
5. **Silent fallbacks.** A failed copula fit assigns the benchmark's weights, which registers as a tie rather than a failure.
6. **Loss is variance only.** Portfolios are compared on squared out-of-sample return. Tail risk measures such as VaR or expected shortfall — the natural metric for a tail-dependence study — are not computed.

---

## References

- Patton, A. J. (2012). *Copula Methods for Forecasting Multivariate Time Series.*
- Joe, H. (2015). *Dependence Modeling with Copulas.*
