# Geothermal Energy Optimization & Reservoir Mapping via Regularized Polynomial Regression

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning project applying **Regularized Polynomial Regression** to solve two multi-stage geothermal energy challenges: optimizing surface steam turbine performance and mapping 3D subterranean thermal reservoirs.

---

## 🌟 Project Overview

This project investigates non-linear regression techniques to balance **model complexity** and **generalization** on simulated real-world sensor logs:
- Systematically benchmarks **Ordinary Least Squares (OLS)**, **Ridge ($L_2$)**, **Lasso ($L_1$)**, and **ElasticNet ($L_1 + L_2$)**.
- Explores polynomial feature spaces up to **Degree 10** (Phase 1) and **Degree 20** (Phase 2).
- Employs a **1% Parsimony Tolerance Rule** (Occam's Razor) to select simpler, more robust models over marginal performance gains.

---

## 🔬 Problems Addressed

### Phase 1: Steam Turbine Optimization (`var1`)
* **Objective:** Predict surface facility **Net Power Score** ($y$) to optimize energy generation prior to production well drilling.
* **Input Features ($x_1$–$x_6$):** Percentage adjustments for high-pressure steam valve, condenser coolant flow, pump hydraulic pressure, blade pitch angle, non-condensable gas exhaust, and steam inlet pressure.
* **Dataset Size:** 1,001 training records, 1,001 test records.

### Phase 2: Subterranean Thermal Mapping (`var2`)
* **Objective:** Predict **Thermal Anomaly Score** ($y$) across a 3D subsurface grid to identify high-yield geothermal extraction coordinates.
* **Input Features ($x_1$–$x_3$):** Spatial offsets in meters from central basecamp:
  - $x_1$: East-West offset
  - $x_2$: North-South offset
  - $x_3$: Vertical depth offset
* **Dataset Size:** 1,001 training records, 1,001 test records.

---

## ⚙️ Methodology & Model Selection

1. **Leak-Free Pipelines:**
   Every candidate model is wrapped in a `scikit-learn` Pipeline.

2. **Model Grid (10 candidates evaluated per degree):**
   - **OLS Linear Regression**
   - **Ridge Regression:** $\alpha \in \{0.1, 1.0, 10.0\}$
   - **Lasso Regression:** $\alpha \in \{0.001, 0.01, 0.1\}$
   - **ElasticNet:** $\alpha \in \{0.001, 0.01\}$ with $l_1\text{-ratio} \in \{0.5, 0.7\}$

3. **5-Fold Cross-Validation:**
   Evaluated using Mean Squared Error (MSE) and Coefficient of Determination ($R^2$).

4. **1% Parsimony Selection Principle:**
   Instead of choosing an over-parameterized model for an insignificant gain, any candidate with:
   $$\text{CV MSE} \le \text{Global Best MSE} \times 1.01$$
   is considered eligible. The one with the **lowest polynomial degree** and **lowest structural complexity** is chosen.

---

## 📊 Experimental Results

| Phase | Selected Model | Polynomial Degree | CV MSE | CV $R^2$ | Train MSE | Train $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 (`var1`)** | **Lasso ($\alpha=0.01$)** | **5** | **0.3606** | **0.9655** | **0.3340** | **0.9680** |
| **Phase 2 (`var2`)** | **Ridge ($\alpha=1.0$)** | **10** | **0.2673** | **0.9936** | **0.2443** | **0.9942** |

> **Key Parsimony Finding in Phase 2:**
> Degree 11 Ridge yielded the minimal CV MSE (0.2647), but Degree 10 Ridge achieved **0.2673** (within 1% tolerance). Choosing Degree 10 significantly reduces polynomial terms while retaining **>99.3% explained variance**.

---

## 📈 Visualizations & Diagnostics

All figures are generated using the datasets in the current repository:

* **CV MSE & $R^2$ vs Degree Curves:** Illustrates the bias-variance tradeoff as polynomial degree scales.
* **Parsimony Selection:** Highlights the 1% tolerance threshold comparing Degree 10 vs. Degree 11.
* **Actual vs. Predicted Parity Plots:** Confirms linear alignment along the 45° ideal fit line without severe outliers.
* **Residual Analysis:**
  - *Residuals vs. Fitted:* Confirms homoscedasticity and absence of systematic patterns.
  - *Residual Distribution:* Verified to be zero-centered and normally distributed.
