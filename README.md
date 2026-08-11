# Multivariate Copula Modeling & Portfolio Optimization

A quantitative framework for modeling joint asset return distributions, non-linear dependencies, and tail risk using multivariate Copula models (e.g., Clayton, Gumbel, Student-$t$) paired with time-series dynamics.

## Overview

Standard portfolio optimization models frequently rely on assumptions of multivariate normality, underestimating joint tail risk during market crises. This repository provides an end-to-end Python framework to model complex dependence structures and optimize portfolios under non-Gaussian conditions.

## Key Features

* **GARCH Volatility Filtering:** Extracts standardized, uncorrelated residuals from raw asset return series.
* **Flexible Dependence Structure:** Implements elliptical (Student-$t$, Gaussian) and Archimedean (Clayton, Gumbel) copulas to capture asymmetric tail dependencies.
* **Rolling-Window Backtesting:** Simulates forward-looking risk metrics ($VaR$, $ES$) and portfolio allocations across historical volatility regimes.
* **Risk-Adjusted Optimization:** Evaluates performance against standard mean-variance and benchmark tracking models.

## Repository Structure

```text
├── Copula_framwork.ipynb    # Main framework and analysis notebook
├── README.md                # Project documentation
└── LICENSE                  # MIT License
