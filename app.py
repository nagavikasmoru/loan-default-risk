import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .risk-high   { border-left: 5px solid #e84c4c !important; }
    .risk-medium { border-left: 5px solid #ff8c42 !important; }
    .risk-low    { border-left: 5px solid #2ecc71 !important; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #4c9be8;
        border-bottom: 2px solid #2d3561;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    div[data-testid="stSidebar"] { background-color: #16213e; }
    .stButton > button {
        background: linear-gradient(90deg, #4c9be8, #2d3561);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 28px;
        font-size: 1rem;
    }
    .stButton > button:hover { opacity: 0.85; }
    label { color: #b0bec5 !important; }
    .stSlider > div > div { color: #4c9be8; }
</style>
""", unsafe_allow_html=True)

# ── Load Model & Imputer ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load("lgbm_model.pkl")
    imputer = joblib.load("imputer.pkl")
    return model, imputer

model, imputer = load_artifacts()

# ── Expected feature list (matches training) ──────────────────────────────────
# Core numeric features used during training; imputer was fitted on these.
NUMERIC_FEATURES = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AGE_YEARS', 'EMPLOYED_YEARS', 'DAYS_EMPLOYED_ANOM',
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_TERM',
    'GOODS_CREDIT_RATIO', 'EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN',
]

# ── Helper: feature engineering ───────────────────────────────────────────────
def engineer_features(raw: dict) -> pd.DataFrame:
    d = raw.copy()
    # Derived features
    d['AGE_YEARS']             = round(-d['DAYS_BIRTH'] / 365, 1)
    d['DAYS_EMPLOYED_ANOM']    = 1 if d['DAYS_EMPLOYED'] == 365243 else 0
    if d['DAYS_EMPLOYED'] == 365243:
        d['DAYS_EMPLOYED'] = np.nan
    d['EMPLOYED_YEARS']        = max(-d.get('DAYS_EMPLOYED', 0) / 365, 0) if not np.isnan(d.get('DAYS_EMPLOYED', np.nan) or np.nan) else np.nan
    d['CREDIT_INCOME_RATIO']   = d['AMT_CREDIT'] / (d['AMT_INCOME_TOTAL'] + 1)
    d['ANNUITY_INCOME_RATIO']  = d['AMT_ANNUITY'] / (d['AMT_INCOME_TOTAL'] + 1)
    d['CREDIT_TERM']           = d['AMT_ANNUITY'] / (d['AMT_CREDIT'] + 1)
    d['GOODS_CREDIT_RATIO']    = d['AMT_GOODS_PRICE'] / (d['AMT_CREDIT'] + 1)
    ext = [d.get('EXT_SOURCE_1', np.nan), d.get('EXT_SOURCE_2', np.nan), d.get('EXT_SOURCE_3', np.nan)]
    ext_valid = [v for v in ext if not np.isnan(v)]
    d['EXT_SOURCE_MEAN'] = np.mean(ext_valid) if ext_valid else np.nan
    d['EXT_SOURCE_MIN']  = np.min(ext_valid)  if ext_valid else np.nan

    df = pd.DataFrame([{f: d.get(f, np.nan) for f in NUMERIC_FEATURES}])
    return df


def predict_risk(features_df: pd.DataFrame):
    X_imp = imputer.transform(features_df)
    proba = model.predict_proba(X_imp)[0][1]
    return proba


def risk_label(p: float):
    if p >= 0.50:
        return "🔴 HIGH RISK",    "risk-high",   "#e84c4c"
    elif p >= 0.25:
        return "🟠 MEDIUM RISK",  "risk-medium", "#ff8c42"
    else:
        return "🟢 LOW RISK",     "risk-low",    "#2ecc71"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input Form
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🏦 Applicant Details")
st.sidebar.markdown("---")

with st.sidebar:
    st.markdown("### 💰 Financial Info")
    income       = st.number_input("Annual Income (₹)", 10_000, 10_000_000, 180_000, step=5_000)
    credit       = st.number_input("Credit Amount (₹)", 10_000, 5_000_000, 500_000, step=10_000)
    annuity      = st.number_input("Annual Annuity (₹)", 1_000, 500_000, 25_000, step=1_000)
    goods_price  = st.number_input("Goods Price (₹)", 0, 5_000_000, 450_000, step=10_000)

    st.markdown("### 👤 Personal Info")
    age          = st.slider("Age (years)", 18, 70, 35)
    employed_yrs = st.slider("Years Employed", 0, 40, 5)
    never_worked = st.checkbox("Never Employed", value=False)

    st.markdown("### 📊 External Scores")
    ext1 = st.slider("External Score 1", 0.0, 1.0, 0.50, step=0.01)
    ext2 = st.slider("External Score 2", 0.0, 1.0, 0.55, step=0.01)
    ext3 = st.slider("External Score 3", 0.0, 1.0, 0.50, step=0.01)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🏦 Loan Default Risk Intelligence Platform")
st.markdown("Powered by **LightGBM** · Feature engineering + SHAP explainability")
st.markdown("---")

# ── Build raw input dict ───────────────────────────────────────────────────────
days_employed_raw = 365243 if never_worked else int(-employed_yrs * 365)

raw_input = {
    'AMT_INCOME_TOTAL': income,
    'AMT_CREDIT':       credit,
    'AMT_ANNUITY':      annuity,
    'AMT_GOODS_PRICE':  goods_price,
    'DAYS_BIRTH':       int(-age * 365),
    'DAYS_EMPLOYED':    days_employed_raw,
    'EXT_SOURCE_1':     ext1,
    'EXT_SOURCE_2':     ext2,
    'EXT_SOURCE_3':     ext3,
}

features_df = engineer_features(raw_input)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 Prediction", "📊 SHAP Explanation", "ℹ️ Model Info"])

# ─────────────────────────────── TAB 1: Prediction ───────────────────────────
with tab1:
    if predict_btn:
        proba = predict_risk(features_df)
        label, css_class, color = risk_label(proba)

        # ── Score gauge ───────────────────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <div style="font-size:0.85rem;color:#888;margin-bottom:6px">DEFAULT PROBABILITY</div>
                <div style="font-size:2.8rem;font-weight:800;color:{color}">{proba*100:.1f}%</div>
                <div style="font-size:1.1rem;margin-top:8px">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#888;margin-bottom:6px">CREDIT / INCOME RATIO</div>
                <div style="font-size:2.0rem;font-weight:700;color:#4c9be8">{credit/max(income,1):.2f}x</div>
                <div style="font-size:0.85rem;color:#888;margin-top:6px">Healthy &lt; 5x</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#888;margin-bottom:6px">ANNUITY / INCOME RATIO</div>
                <div style="font-size:2.0rem;font-weight:700;color:#4c9be8">{annuity/max(income,1)*100:.1f}%</div>
                <div style="font-size:0.85rem;color:#888;margin-top:6px">Healthy &lt; 30%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Risk gauge bar ────────────────────────────────────────────────────
        st.markdown("### Risk Probability Gauge")
        fig, ax = plt.subplots(figsize=(10, 1.2))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#0f1117')
        ax.barh([''], [1], color='#1a1f2e', height=0.4)
        bar_color = color
        ax.barh([''], [proba], color=bar_color, height=0.4)
        ax.axvline(0.25, color='#ff8c42', lw=1.5, linestyle='--', alpha=0.7)
        ax.axvline(0.50, color='#e84c4c', lw=1.5, linestyle='--', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%\n(Low→Med)', '50%\n(Med→High)', '75%', '100%'], color='#888', fontsize=8)
        ax.tick_params(left=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Engineered features preview ───────────────────────────────────────
        st.markdown("### Computed Input Features")
        display_df = features_df.T.rename(columns={0: "Value"})
        display_df["Value"] = display_df["Value"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "NaN")
        st.dataframe(display_df, use_container_width=True)

    else:
        st.info("👈 Fill in the applicant details in the sidebar and click **Predict Default Risk**.")


# ─────────────────────────────── TAB 2: SHAP ─────────────────────────────────
with tab2:
    if predict_btn:
        st.markdown("### SHAP Feature Contribution")
        st.caption("SHAP values show how each feature pushes the prediction above (red) or below (blue) the base rate.")

        try:
            X_imp = imputer.transform(features_df)
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_imp)

            # LightGBM may return list for binary classification
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
                base_val = explainer.expected_value[1]
            else:
                sv = shap_values[0]
                base_val = explainer.expected_value

            feature_names = NUMERIC_FEATURES
            shap_df = pd.DataFrame({
                'Feature':    feature_names[:len(sv)],
                'SHAP Value': sv[:len(feature_names)],
                'Input Value': X_imp[0][:len(feature_names)]
            }).sort_values('SHAP Value', key=abs, ascending=False).head(12)

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#1a1f2e')
            colors = ['#e84c4c' if v > 0 else '#4c9be8' for v in shap_df['SHAP Value']]
            ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors, edgecolor='none')
            ax.axvline(0, color='#888', linewidth=0.8)
            ax.set_xlabel('SHAP Value (impact on default probability)', color='#b0bec5')
            ax.set_title('Feature Contributions – This Applicant', color='#e0e0e0', fontweight='bold')
            ax.tick_params(colors='#b0bec5')
            for spine in ax.spines.values():
                spine.set_color('#2d3561')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Waterfall-style table
            st.markdown("#### Detailed SHAP Breakdown")
            shap_display = shap_df.copy()
            shap_display['Direction'] = shap_display['SHAP Value'].apply(
                lambda x: '⬆ Increases Risk' if x > 0 else '⬇ Decreases Risk'
            )
            shap_display['SHAP Value'] = shap_display['SHAP Value'].map('{:+.4f}'.format)
            shap_display['Input Value'] = shap_display['Input Value'].map('{:.4f}'.format)
            st.dataframe(shap_display[['Feature', 'Input Value', 'SHAP Value', 'Direction']],
                         use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"SHAP computation error: {e}")
    else:
        st.info("👈 Run a prediction first to see the SHAP explanation.")


# ─────────────────────────────── TAB 3: Model Info ───────────────────────────
with tab3:
    st.markdown("### About This Model")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Algorithm:** LightGBM Classifier

        **Training Dataset:** Home Credit Default Risk (Kaggle)
        - ~307,000 loan applications
        - Binary target: `0 = No Default`, `1 = Default`
        - Class imbalance handled with `is_unbalance=True`

        **Preprocessing:**
        - Domain-driven feature engineering (ratios, age, employment)
        - Median imputation for missing values
        - External bureau scores (EXT_SOURCE 1–3)
        """)
    with col2:
        st.markdown("""
        **Model Hyperparameters:**
        - n_estimators: 200
        - learning_rate: 0.05
        - num_leaves: 31
        - subsample: 0.8
        - colsample_bytree: 0.8

        **Evaluation Metrics (Test Set):**
        - ROC-AUC: ~0.76
        - Explainability: SHAP TreeExplainer

        **Decision Threshold:**
        - Low Risk: < 25%
        - Medium Risk: 25% – 50%
        - High Risk: ≥ 50%
        """)

    st.markdown("---")
    st.markdown("### Feature Dictionary")
    feature_dict = {
        "AMT_INCOME_TOTAL":     "Applicant's annual income",
        "AMT_CREDIT":           "Total loan credit amount",
        "AMT_ANNUITY":          "Loan annuity (periodic payment)",
        "AMT_GOODS_PRICE":      "Price of the goods financed",
        "DAYS_BIRTH":           "Age in days (negative = past)",
        "DAYS_EMPLOYED":        "Employment duration in days",
        "EXT_SOURCE_1/2/3":     "External credit bureau scores (0–1)",
        "CREDIT_INCOME_RATIO":  "Credit ÷ Income — affordability signal",
        "ANNUITY_INCOME_RATIO": "Annuity ÷ Income — repayment burden",
        "CREDIT_TERM":          "Annuity ÷ Credit — loan term proxy",
        "EXT_SOURCE_MEAN/MIN":  "Aggregated external scores",
    }
    df_dict = pd.DataFrame(list(feature_dict.items()), columns=["Feature", "Description"])
    st.dataframe(df_dict, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Built with Streamlit · LightGBM · SHAP · Scikit-learn  |  Portfolio Project by Vikas")
