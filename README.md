# Fractional Differentiation for Time-Series Forecasting

*Stationarity Analysis and ML Forecasting on S&P500, DAX, and Nikkei225*

This repository contains the code and experiments developed to analyse the effect of **fractional differentiation** on financial time series and to evaluate how it influences the performance of machine-learning forecasting models.
The project is organized around two Jupyter notebooks:

* **`visualization_stationarity.ipynb`**
* **`training.ipynb`**

and relies on daily Low, High, Close, Open price data of the **S&P500**, **DAX**, and **Nikkei 225** indices from **1 June 2010 to 30 June 2020**.

---

## 1. Overview

Financial time series typically exhibit non-stationary behaviour. Since many supervised learning methods require stationary inputs, transformations such as differencing are essential.
Standard *integer differencing* often removes too much information, whereas **fractional differencing** can achieve stationarity while retaining long-term memory.
This project examines this idea both through statistical analysis and through forecasting experiments.

---

## 2. Dataset

The dataset consists of daily closing prices for the S&P500, DAX, and Nikkei 225.
For each index:

* the file `*_data.csv` contains the raw prices;
* the file `*_updated.pkl` contains the processed dataset, including:

  * the log-returns,
  * the integer-differenced features,
  * the fractionally-differenced features.


---

## 3. Notebook 1 — Visualization and Stationarity Analysis

**File:** `visualization_stationarity.ipynb`

This notebook examines the structure and statistical properties of each time series before any modelling is carried out.

### Main components:

* **Data loading and preprocessing**.
* **Exploratory visualizations** of prices, returns, and both integer- and fractionally-differenced series.
* **Stationarity assessment using the Augmented Dickey–Fuller (ADF) test**, which statistically evaluates whether a unit root is present. The test is applied before and after each transformation to determine whether the series has become stationary.
* **Construction of fractional differences**, highlighting that fractional differencing typically produces a stationary series while preserving more information than traditional differencing.

This analysis motivates the use of fractional differentiation in predictive models.

---

## 4. Notebook 2 — Forecasting Models

**File:** `training.ipynb`

This notebook trains and compares several forecasting models using two different setups: **one-step-ahead prediction** and a **sliding-window approach**.

### Forecasting setups

* **One-step-ahead prediction**:
  Predicts the log-return of the next day using only the features available on the current day.

* **Sliding-window prediction**:
  Uses the most recent `window_size` observations as input to forecast the next log-return.

---

## 5. References

* M. L. De Prado, *Advances in Financial Machine Learning*, Wiley, 2018.
* R. Walasek & J. Gajda, “Fractional differentiation and its use in machine learning”, IJESAM, 2021.
* P. Brockwell & R. Davis, *Time Series: Theory and Methods*, Springer, 2009.

---

Se vuoi, posso perfezionare ulteriormente il tono, aggiungere esempi di comandi d’esecuzione, includere grafici di esempio, o adattarlo completamente allo stile della tua tesi.
