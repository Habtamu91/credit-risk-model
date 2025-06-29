# Credit Risk Probability Model for Alternative Data

This project is part of the 10 Academy Week 5 Challenge, where we simulate building a machine learning product for **Bati Bank** to predict credit risk using alternative behavioral transaction data.

---

## Business Context

Bati Bank is introducing a Buy-Now-Pay-Later service in partnership with an eCommerce platform. As a data scientist, your goal is to develop a **credit scoring system** using historical transaction data to:
- Identify high-risk customers
- Predict probability of default
- Estimate optimal loan terms

---

## Objective

Build an end-to-end machine learning system that:
- Uses RFM (Recency, Frequency, Monetary) to engineer a proxy credit risk label (`is_high_risk`)
- Trains and evaluates classification models (Logistic Regression, Gradient Boosting, etc.)
- Deploys predictions using FastAPI in a Docker container

---

---

##  Credit Scoring Business Understanding

### 1. How does the Basel II Accord influence the need for an interpretable model?

Basel II emphasizes risk-sensitive capital allocation. Banks must clearly understand and **document their risk models**. This makes **model interpretability** crucial — models like **Logistic Regression** with Weight of Evidence (WoE) are favored in regulated environments, as they're easier to explain to auditors and regulators.

---

### 2. Why is creating a proxy variable necessary?

Our dataset does not have a direct **loan default label**. We simulate one by creating a **proxy target (`is_high_risk`)** based on behavioral patterns (RFM metrics). However, using proxies introduces **bias and risk** — it may not perfectly reflect real-world defaults, potentially leading to unfair or inaccurate credit decisions.

---

### 3. What are the trade-offs between simple vs complex models?

| Simple Model (Logistic Regression) | Complex Model (Gradient Boosting)        |
|------------------------------------|-------------------------------------------|
| ✅ Interpretable                    | ✅ Higher accuracy                          |
| ✅ Regulator-friendly               | ❌ Hard to explain                          |
| ❌ May underperform on noisy data   | ✅ Captures nonlinear patterns              |
| ✅ WoE & IV features compatible     | ❌ Requires careful monitoring & tuning     |

In banking, often a **hybrid approach** is taken — complex models for risk scoring, but simple interpretable models for compliance.

---

## ✅ Tasks Completed

- ✅ EDA in `notebooks/1.0-eda.ipynb`
- ✅ Feature engineering in `src/data_processing.py`
- ✅ Proxy target generation using RFM + KMeans
- ✅ Data saved in `data/processed/processed_data.csv`
- ✅ Model training (Task 5 — in progress)
- ✅ Docker setup with FastAPI (Task 6 — next)

---
