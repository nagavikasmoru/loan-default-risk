import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
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

st.markdown("""
<style>
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
    div[data-testid="stSidebar"] { background-color: #16213e; }
    .stButton > button {
        background: linear-gradient(90deg, #4c9be8, #2d3561);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 10px 28px; font-size: 1rem;
    }
    label { color: #b0bec5 !important; }
</style>
""", unsafe_allow_html=True)

# ── All 237 features the imputer expects ─────────────────────────────────────
ALL_FEATURES = ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'EMERGENCYSTATE_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'AGE_YEARS', 'DAYS_EMPLOYED_ANOM', 'EMPLOYED_YEARS', 'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_TERM', 'GOODS_CREDIT_RATIO', 'EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN', 'CODE_GENDER_M', 'CODE_GENDER_XNA', 'NAME_TYPE_SUITE_Family', 'NAME_TYPE_SUITE_Group of people', 'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Other_B', 'NAME_TYPE_SUITE_Spouse, partner', 'NAME_TYPE_SUITE_Unaccompanied', 'NAME_INCOME_TYPE_Commercial associate', 'NAME_INCOME_TYPE_Maternity leave', 'NAME_INCOME_TYPE_Pensioner', 'NAME_INCOME_TYPE_State servant', 'NAME_INCOME_TYPE_Student', 'NAME_INCOME_TYPE_Unemployed', 'NAME_INCOME_TYPE_Working', 'NAME_EDUCATION_TYPE_Higher education', 'NAME_EDUCATION_TYPE_Incomplete higher', 'NAME_EDUCATION_TYPE_Lower secondary', 'NAME_EDUCATION_TYPE_Secondary / secondary special', 'NAME_FAMILY_STATUS_Married', 'NAME_FAMILY_STATUS_Separated', 'NAME_FAMILY_STATUS_Single / not married', 'NAME_FAMILY_STATUS_Unknown', 'NAME_FAMILY_STATUS_Widow', 'NAME_HOUSING_TYPE_House / apartment', 'NAME_HOUSING_TYPE_Municipal apartment', 'NAME_HOUSING_TYPE_Office apartment', 'NAME_HOUSING_TYPE_Rented apartment', 'NAME_HOUSING_TYPE_With parents', 'OCCUPATION_TYPE_Cleaning staff', 'OCCUPATION_TYPE_Cooking staff', 'OCCUPATION_TYPE_Core staff', 'OCCUPATION_TYPE_Drivers', 'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff', 'OCCUPATION_TYPE_IT staff', 'OCCUPATION_TYPE_Laborers', 'OCCUPATION_TYPE_Low-skill Laborers', 'OCCUPATION_TYPE_Managers', 'OCCUPATION_TYPE_Medicine staff', 'OCCUPATION_TYPE_Private service staff', 'OCCUPATION_TYPE_Realty agents', 'OCCUPATION_TYPE_Sales staff', 'OCCUPATION_TYPE_Secretaries', 'OCCUPATION_TYPE_Security staff', 'OCCUPATION_TYPE_Waiters/barmen staff', 'WEEKDAY_APPR_PROCESS_START_MONDAY', 'WEEKDAY_APPR_PROCESS_START_SATURDAY', 'WEEKDAY_APPR_PROCESS_START_SUNDAY', 'WEEKDAY_APPR_PROCESS_START_THURSDAY', 'WEEKDAY_APPR_PROCESS_START_TUESDAY', 'WEEKDAY_APPR_PROCESS_START_WEDNESDAY', 'ORGANIZATION_TYPE_Agriculture', 'ORGANIZATION_TYPE_Bank', 'ORGANIZATION_TYPE_Business Entity Type 1', 'ORGANIZATION_TYPE_Business Entity Type 2', 'ORGANIZATION_TYPE_Business Entity Type 3', 'ORGANIZATION_TYPE_Cleaning', 'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Culture', 'ORGANIZATION_TYPE_Electricity', 'ORGANIZATION_TYPE_Emergency', 'ORGANIZATION_TYPE_Government', 'ORGANIZATION_TYPE_Hotel', 'ORGANIZATION_TYPE_Housing', 'ORGANIZATION_TYPE_Industry: type 1', 'ORGANIZATION_TYPE_Industry: type 10', 'ORGANIZATION_TYPE_Industry: type 11', 'ORGANIZATION_TYPE_Industry: type 12', 'ORGANIZATION_TYPE_Industry: type 13', 'ORGANIZATION_TYPE_Industry: type 2', 'ORGANIZATION_TYPE_Industry: type 3', 'ORGANIZATION_TYPE_Industry: type 4', 'ORGANIZATION_TYPE_Industry: type 5', 'ORGANIZATION_TYPE_Industry: type 6', 'ORGANIZATION_TYPE_Industry: type 7', 'ORGANIZATION_TYPE_Industry: type 8', 'ORGANIZATION_TYPE_Industry: type 9', 'ORGANIZATION_TYPE_Insurance', 'ORGANIZATION_TYPE_Kindergarten', 'ORGANIZATION_TYPE_Legal Services', 'ORGANIZATION_TYPE_Medicine', 'ORGANIZATION_TYPE_Military', 'ORGANIZATION_TYPE_Mobile', 'ORGANIZATION_TYPE_Other', 'ORGANIZATION_TYPE_Police', 'ORGANIZATION_TYPE_Postal', 'ORGANIZATION_TYPE_Realtor', 'ORGANIZATION_TYPE_Religion', 'ORGANIZATION_TYPE_Restaurant', 'ORGANIZATION_TYPE_School', 'ORGANIZATION_TYPE_Security', 'ORGANIZATION_TYPE_Security Ministries', 'ORGANIZATION_TYPE_Self-employed', 'ORGANIZATION_TYPE_Services', 'ORGANIZATION_TYPE_Telecom', 'ORGANIZATION_TYPE_Trade: type 1', 'ORGANIZATION_TYPE_Trade: type 2', 'ORGANIZATION_TYPE_Trade: type 3', 'ORGANIZATION_TYPE_Trade: type 4', 'ORGANIZATION_TYPE_Trade: type 5', 'ORGANIZATION_TYPE_Trade: type 6', 'ORGANIZATION_TYPE_Trade: type 7', 'ORGANIZATION_TYPE_Transport: type 1', 'ORGANIZATION_TYPE_Transport: type 2', 'ORGANIZATION_TYPE_Transport: type 3', 'ORGANIZATION_TYPE_Transport: type 4', 'ORGANIZATION_TYPE_University', 'ORGANIZATION_TYPE_XNA', 'FONDKAPREMONT_MODE_org spec account', 'FONDKAPREMONT_MODE_reg oper account', 'FONDKAPREMONT_MODE_reg oper spec account', 'HOUSETYPE_MODE_specific housing', 'HOUSETYPE_MODE_terraced house', 'WALLSMATERIAL_MODE_Mixed', 'WALLSMATERIAL_MODE_Monolithic', 'WALLSMATERIAL_MODE_Others', 'WALLSMATERIAL_MODE_Panel', 'WALLSMATERIAL_MODE_Stone, brick', 'WALLSMATERIAL_MODE_Wooden']

@st.cache_resource
def load_artifacts():
    model = joblib.load("lgbm_model.pkl")
    with open("imputer_medians.json") as f:
        medians = json.load(f)
    return model, medians

model, imputer_medians = load_artifacts()

def impute(df):
    out = df.copy()
    for col in ALL_FEATURES:
        out[col] = out[col].fillna(imputer_medians.get(col, 0))
    return out[ALL_FEATURES].values.astype(float)


def build_feature_row(raw: dict) -> pd.DataFrame:
    """Build a single-row DataFrame with all 237 features. Unknown = NaN/0."""
    d = {f: np.nan for f in ALL_FEATURES}

    # ── Direct inputs ──────────────────────────────────────────────────────
    d['AMT_INCOME_TOTAL'] = raw['income']
    d['AMT_CREDIT']       = raw['credit']
    d['AMT_ANNUITY']      = raw['annuity']
    d['AMT_GOODS_PRICE']  = raw['goods_price']
    d['DAYS_BIRTH']       = int(-raw['age'] * 365)
    d['EXT_SOURCE_1']     = raw['ext1']
    d['EXT_SOURCE_2']     = raw['ext2']
    d['EXT_SOURCE_3']     = raw['ext3']
    d['CNT_CHILDREN']     = raw['children']
    d['CNT_FAM_MEMBERS']  = raw['family_members']

    # Employment
    never_worked = raw['never_worked']
    d['DAYS_EMPLOYED_ANOM'] = 1 if never_worked else 0
    d['DAYS_EMPLOYED']      = np.nan if never_worked else int(-raw['employed_yrs'] * 365)

    # ── Engineered features ────────────────────────────────────────────────
    d['AGE_YEARS']            = round(raw['age'], 1)
    d['EMPLOYED_YEARS']       = 0 if never_worked else max(raw['employed_yrs'], 0)
    d['CREDIT_INCOME_RATIO']  = raw['credit'] / (raw['income'] + 1)
    d['ANNUITY_INCOME_RATIO'] = raw['annuity'] / (raw['income'] + 1)
    d['CREDIT_TERM']          = raw['annuity'] / (raw['credit'] + 1)
    d['GOODS_CREDIT_RATIO']   = raw['goods_price'] / (raw['credit'] + 1)

    ext = [raw['ext1'], raw['ext2'], raw['ext3']]
    d['EXT_SOURCE_MEAN'] = float(np.mean(ext))
    d['EXT_SOURCE_MIN']  = float(np.min(ext))

    # ── Categorical one-hots from UI ───────────────────────────────────────
    gender = raw['gender']
    if gender == 'Male':
        d['CODE_GENDER_M'] = 1
    elif gender == 'XNA':
        d['CODE_GENDER_XNA'] = 1

    contract = raw['contract_type']
    d['NAME_CONTRACT_TYPE'] = 1 if contract == 'Cash loans' else 0

    own_car = raw['own_car']
    d['FLAG_OWN_CAR'] = 1 if own_car == 'Yes' else 0

    own_realty = raw['own_realty']
    d['FLAG_OWN_REALTY'] = 1 if own_realty == 'Yes' else 0

    income_type = raw['income_type']
    col = f'NAME_INCOME_TYPE_{income_type}'
    if col in d:
        d[col] = 1

    education = raw['education']
    col = f'NAME_EDUCATION_TYPE_{education}'
    if col in d:
        d[col] = 1

    family_status = raw['family_status']
    col = f'NAME_FAMILY_STATUS_{family_status}'
    if col in d:
        d[col] = 1

    housing = raw['housing_type']
    col = f'NAME_HOUSING_TYPE_{housing}'
    if col in d:
        d[col] = 1

    occupation = raw['occupation']
    if occupation != 'Unknown':
        col = f'OCCUPATION_TYPE_{occupation}'
        if col in d:
            d[col] = 1

    # Flag defaults for common binary flags
    d['FLAG_MOBIL']       = 1
    d['FLAG_EMP_PHONE']   = 1
    d['FLAG_CONT_MOBILE'] = 1
    d['FLAG_DOCUMENT_3']  = 1

    # Zero-fill remaining NaN for pure binary/flag columns
    flag_cols = [c for c in ALL_FEATURES if c.startswith('FLAG_') or
                 c.startswith('NAME_') or c.startswith('CODE_') or
                 c.startswith('OCCUPATION_') or c.startswith('ORGANIZATION_') or
                 c.startswith('WEEKDAY_') or c.startswith('FONDKAPREMONT_') or
                 c.startswith('HOUSETYPE_') or c.startswith('WALLSMATERIAL_') or
                 c.startswith('NAME_TYPE_SUITE')]
    for c in flag_cols:
        if np.isnan(d[c]):
            d[c] = 0

    return pd.DataFrame([d])[ALL_FEATURES]


def risk_label(p):
    if p >= 0.50:
        return "🔴 HIGH RISK",   "risk-high",   "#e84c4c"
    elif p >= 0.25:
        return "🟠 MEDIUM RISK", "risk-medium", "#ff8c42"
    else:
        return "🟢 LOW RISK",    "risk-low",    "#2ecc71"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🏦 Applicant Details")
st.sidebar.markdown("---")

with st.sidebar:
    st.markdown("### 💰 Financial Info")
    income      = st.number_input("Annual Income (₹)", 10_000, 10_000_000, 180_000, step=5_000)
    credit      = st.number_input("Credit Amount (₹)", 10_000, 5_000_000, 500_000, step=10_000)
    annuity     = st.number_input("Annual Annuity (₹)", 1_000, 500_000, 25_000, step=1_000)
    goods_price = st.number_input("Goods Price (₹)", 0, 5_000_000, 450_000, step=10_000)

    st.markdown("### 👤 Personal Info")
    age            = st.slider("Age (years)", 18, 70, 35)
    gender         = st.selectbox("Gender", ["Female", "Male", "XNA"])
    children       = st.slider("Number of Children", 0, 10, 0)
    family_members = st.slider("Family Members", 1, 15, 2)
    own_car        = st.selectbox("Owns a Car?", ["No", "Yes"])
    own_realty     = st.selectbox("Owns Real Estate?", ["No", "Yes"])

    st.markdown("### 💼 Employment & Background")
    income_type    = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner",
                                                   "State servant", "Student", "Unemployed"])
    education      = st.selectbox("Education", ["Secondary / secondary special", "Higher education",
                                                  "Incomplete higher", "Lower secondary"])
    family_status  = st.selectbox("Family Status", ["Married", "Single / not married",
                                                     "Separated", "Widow", "Unknown"])
    housing_type   = st.selectbox("Housing Type", ["House / apartment", "With parents",
                                                    "Municipal apartment", "Rented apartment",
                                                    "Office apartment"])
    occupation     = st.selectbox("Occupation", ["Unknown", "Laborers", "Core staff", "Managers",
                                                  "Drivers", "Sales staff", "High skill tech staff",
                                                  "Accountants", "Medicine staff", "Security staff",
                                                  "Cooking staff", "Cleaning staff", "IT staff"])
    contract_type  = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
    never_worked   = st.checkbox("Never Employed", value=False)
    employed_yrs   = st.slider("Years Employed", 0, 40, 5, disabled=never_worked)

    st.markdown("### 📊 External Credit Scores")
    ext1 = st.slider("External Score 1", 0.0, 1.0, 0.50, step=0.01)
    ext2 = st.slider("External Score 2", 0.0, 1.0, 0.55, step=0.01)
    ext3 = st.slider("External Score 3", 0.0, 1.0, 0.50, step=0.01)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🏦 Loan Default Risk Intelligence Platform")
st.markdown("Powered by **LightGBM** · 237 features · SHAP explainability")
st.markdown("---")

raw_input = dict(
    income=income, credit=credit, annuity=annuity, goods_price=goods_price,
    age=age, gender=gender, children=children, family_members=family_members,
    own_car=own_car, own_realty=own_realty,
    income_type=income_type, education=education, family_status=family_status,
    housing_type=housing_type, occupation=occupation, contract_type=contract_type,
    never_worked=never_worked, employed_yrs=employed_yrs,
    ext1=ext1, ext2=ext2, ext3=ext3,
)

tab1, tab2, tab3 = st.tabs(["📋 Prediction", "📊 SHAP Explanation", "ℹ️ Model Info"])

# ─────────────────────────── TAB 1 ───────────────────────────────────────────
with tab1:
    if predict_btn:
        features_df = build_feature_row(raw_input)
        X_imp = impute(features_df)
        proba = model.predict_proba(X_imp)[0][1]
        label, css_class, color = risk_label(proba)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <div style="font-size:0.85rem;color:#888;margin-bottom:6px">DEFAULT PROBABILITY</div>
                <div style="font-size:2.8rem;font-weight:800;color:{color}">{proba*100:.1f}%</div>
                <div style="font-size:1.1rem;margin-top:8px">{label}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#888;margin-bottom:6px">CREDIT / INCOME RATIO</div>
                <div style="font-size:2.0rem;font-weight:700;color:#4c9be8">{credit/max(income,1):.2f}x</div>
                <div style="font-size:0.85rem;color:#888;margin-top:6px">Healthy &lt; 5x</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#888;margin-bottom:6px">ANNUITY / INCOME RATIO</div>
                <div style="font-size:2.0rem;font-weight:700;color:#4c9be8">{annuity/max(income,1)*100:.1f}%</div>
                <div style="font-size:0.85rem;color:#888;margin-top:6px">Healthy &lt; 30%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### Risk Probability Gauge")
        fig, ax = plt.subplots(figsize=(10, 1.2))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#0f1117')
        ax.barh([''], [1], color='#1a1f2e', height=0.4)
        ax.barh([''], [proba], color=color, height=0.4)
        ax.axvline(0.25, color='#ff8c42', lw=1.5, linestyle='--', alpha=0.7)
        ax.axvline(0.50, color='#e84c4c', lw=1.5, linestyle='--', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='#888', fontsize=9)
        ax.tick_params(left=False, labelleft=False)
        for sp in ax.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Key engineered features
        st.markdown("### Key Computed Features")
        kf = pd.DataFrame({
            'Feature': ['Credit/Income Ratio', 'Annuity/Income Ratio', 'Credit Term',
                        'Goods/Credit Ratio', 'EXT Source Mean', 'EXT Source Min'],
            'Value': [
                f"{credit/max(income,1):.4f}",
                f"{annuity/max(income,1):.4f}",
                f"{annuity/max(credit,1):.4f}",
                f"{goods_price/max(credit,1):.4f}",
                f"{np.mean([ext1,ext2,ext3]):.4f}",
                f"{np.min([ext1,ext2,ext3]):.4f}",
            ]
        })
        st.dataframe(kf, use_container_width=True, hide_index=True)
    else:
        st.info("👈 Fill in the applicant details in the sidebar and click **Predict Default Risk**.")

# ─────────────────────────── TAB 2 ───────────────────────────────────────────
with tab2:
    if predict_btn:
        st.markdown("### SHAP Feature Contributions")
        st.caption("Red = increases default risk · Blue = decreases default risk")
        try:
            features_df = build_feature_row(raw_input)
            X_imp = impute(features_df)
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_imp)

            if isinstance(shap_values, list):
                sv = shap_values[1][0]
                base_val = explainer.expected_value[1]
            else:
                sv = shap_values[0]
                base_val = explainer.expected_value

            shap_df = pd.DataFrame({
                'Feature':    ALL_FEATURES,
                'SHAP Value': sv,
                'Value':      X_imp[0]
            }).reindex(columns=['Feature','Value','SHAP Value'])
            shap_df = shap_df.sort_values('SHAP Value', key=abs, ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#0f1117')
            ax.set_facecolor('#1a1f2e')
            colors = ['#e84c4c' if v > 0 else '#4c9be8' for v in shap_df['SHAP Value']]
            ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
            ax.axvline(0, color='#888', lw=0.8)
            ax.set_xlabel('SHAP Value', color='#b0bec5')
            ax.set_title('Top 15 Feature Contributions', color='#e0e0e0', fontweight='bold')
            ax.tick_params(colors='#b0bec5')
            for sp in ax.spines.values(): sp.set_color('#2d3561')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            shap_df['Direction'] = shap_df['SHAP Value'].apply(
                lambda x: '⬆ Increases Risk' if x > 0 else '⬇ Decreases Risk')
            shap_df['SHAP Value'] = shap_df['SHAP Value'].map('{:+.4f}'.format)
            shap_df['Value']      = shap_df['Value'].map('{:.4f}'.format)
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"SHAP error: {e}")
    else:
        st.info("👈 Run a prediction first to see SHAP explanations.")

# ─────────────────────────── TAB 3 ───────────────────────────────────────────
with tab3:
    st.markdown("### About This Model")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Algorithm:** LightGBM Classifier

        **Dataset:** Home Credit Default Risk (Kaggle)
        - ~307,000 loan applications
        - Binary target: `0 = No Default`, `1 = Default`
        - Class imbalance: `is_unbalance=True`

        **Preprocessing:**
        - 237 features (raw + engineered)
        - Median imputation for missing values
        - Domain features: ratios, age, employment
        """)
    with col2:
        st.markdown("""
        **Hyperparameters:**
        - n_estimators: 200
        - learning_rate: 0.05
        - num_leaves: 31
        - subsample: 0.8, colsample_bytree: 0.8

        **Risk Thresholds:**
        - 🟢 Low Risk: < 25%
        - 🟠 Medium Risk: 25–50%
        - 🔴 High Risk: ≥ 50%
        """)
    st.caption("Built with Streamlit · LightGBM · SHAP · Scikit-learn  |  Portfolio Project by Vikas")
