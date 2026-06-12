# 🏦 Loan Default Risk Intelligence Platform

A production-ready Streamlit application that predicts the probability of loan default for individual applicants, powered by **LightGBM** and explained with **SHAP**.

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

> Deploy instructions below ↓

---

## 📸 Features

| Feature | Description |
|---|---|
| 🎯 Real-time prediction | Instant default probability with risk tier (Low / Medium / High) |
| 📊 Risk gauge | Visual probability bar with threshold markers |
| 🔍 SHAP explainability | Per-applicant feature contributions — why did the model decide this? |
| 🧮 Feature engineering | Auto-computes financial ratios, age, employment tenure |
| 📋 Model card | Algorithm details, hyperparameters, feature dictionary |

---

## 🗂️ Project Structure

```
loan-default-risk/
├── app.py                  # Streamlit application
├── lgbm_model.pkl          # Trained LightGBM model
├── imputer.pkl             # Fitted median imputer
├── requirements.txt        # Python dependencies
├── notebooks/
│   └── loan_default_risk.ipynb   # Full EDA + training notebook
└── README.md
```

---

## ⚙️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/loan-default-risk.git
cd loan-default-risk

# 2. Create and activate a conda environment
conda create -n loan_risk python=3.10 -y
conda activate loan_risk

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy!**

The app will be live at `https://<your-app>.streamlit.app` in ~2 minutes.

---

## 📊 Model Details

| Item | Value |
|---|---|
| Algorithm | LightGBM Classifier |
| Dataset | [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk) |
| Training samples | ~307,000 applications |
| Imbalance handling | `is_unbalance=True` |
| ROC-AUC (test) | ~0.76 |
| Explainability | SHAP TreeExplainer |

### Key Engineered Features

- `CREDIT_INCOME_RATIO` — how much credit relative to income
- `ANNUITY_INCOME_RATIO` — monthly repayment burden
- `EXT_SOURCE_MEAN / MIN` — aggregated external bureau scores
- `EMPLOYED_YEARS` — employment stability signal

---

## 🧠 Skills Demonstrated

- Binary classification on highly imbalanced data
- Domain-driven feature engineering for credit risk
- Hyperparameter tuning with LightGBM
- Model interpretability with SHAP
- Streamlit app development + deployment
- Python packaging for reproducibility

---

## 👨‍💻 Author

**Vikas** — Generative AI & ML Portfolio  
[GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-profile>)

---

*Built as part of an ML portfolio focusing on practical, deployable AI solutions.*
