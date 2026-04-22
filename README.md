# Fractional Differentiation in Financial Time Series

> Semester project (ETH Zürich, D-MATH)  
> Advisor: Dr. Gabriele Visentin

## Overview

This project investigates fractional differentiation (FD) as a preprocessing technique for financial time series in machine learning.

The goal is to assess whether FD can enforce stationarity while preserving long-range dependence, as advocated in López de Prado (2018), and to evaluate its impact on predictive performance.

The analysis combines theoretical insights, empirical experiments, and controlled simulations.

---

## Scope of the Project

The project is structured in three main parts:

### 1. Reproduction of existing methodology

We reproduce the FD + ADF pipeline proposed in the literature:
- fractional differentiation via fixed-window approximation
- selection of the order \( d^* \) using the Augmented Dickey–Fuller test
- application to financial time series

This part is mainly implemented in:
- `visualization_stationarity.ipynb`
- `training.ipynb`

---

### 2. Empirical evaluation on real data (S&P 500)

We conduct a detailed study on S&P 500 log-prices:
- application of fractional differentiation with ADF-based selection
- analysis of stationarity and dependence structure
- rolling train–test evaluation
- comparison of models:
  - neural networks
  - linear regression

We also introduce a **trading-based metric**:
- positions based on predicted log-returns
- evaluation via cumulative returns

These analyses are implemented in:
- `S&P500_training.ipynb`
- `S&P500_second_notebook.ipynb`

---

### 3. Controlled simulations

To better understand the behavior of FD, we perform simulations of stochastic processes:

- ARIMA processes (e.g. ARIMA(1,1,1))
- Black–Scholes log-price dynamics

These experiments allow us to:
- isolate the effect of fractional differentiation
- test the reliability of ADF-based order selection
- study stationarity in a controlled setting

Implemented in:
- `arima_simulation.ipynb`

---

## Main Findings

The project highlights several limitations of the FD + ADF methodology:

- **ADF-based selection does not guarantee stationarity**  
  Even when the unit-root hypothesis is rejected, the resulting series may remain non-stationary.

- **Residual non-stationarity induces distribution shift**  
  Features obtained via FD can exhibit significant train–test drift, especially in rolling evaluations.

- **No clear predictive advantage**  
  Fractional differentiation does not consistently outperform simpler preprocessing methods. Linear models remain competitive with neural networks.

---

## Methodology

- Fractional differentiation via truncated binomial expansion
- ADF-based order selection
- Time series models: ARIMA / ARFIMA framework
- Evaluation:
  - RMSE / MAE
  - trading-based performance
  - distributional stability analysis

---

## Repository Structure

- `notebooks/`
  - `visualization_stationarity.ipynb` – stationarity analysis and ADF behavior
  - `training.ipynb` – model training and evaluation
  - `S&P500_training.ipynb` – empirical analysis on S&P 500
  - `S&P500_second_notebook.ipynb` – extended analysis on S&P 500
  - `arima_simulation.ipynb` – controlled ARIMA experiments

- `src/`
  - `time_series_utils.py` - implementation of fractional differentiation, preprocessing and evaluation utilities

---

## Key Insight

Fractional differentiation provides a trade-off between memory preservation and stationarity. However, in practice, standard selection procedures may fail to achieve true stationarity, leading to unstable learning conditions in downstream models.

---

## References

- M. L. De Prado, *Advances in Financial Machine Learning*, Wiley, 2018.
- R. Walasek & J. Gajda, “Fractional differentiation and its use in machine learning”, IJESAM, 2021.
- P. Brockwell & R. Davis, *Time Series: Theory and Methods*, Springer, 2009.
