from typing import Dict, Optional
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import statsmodels.api as sm
import pickle

# ====== 전역 상수 ======
TEEN_EXCLUDED_YEARS = {2015, 2016}
TEEN_OBESITY_PERCENTILE = 0.95
TEEN_MODEL_THRESHOLD = 0.49
ADULT_MODEL_THRESHOLD = 0.1667  # 기본값(이후에 덮어씀)
ADULT_DEFAULT_HDL = 53.50
MODEL_PATH = "logit_model.pkl"

# ======================================================================
# 🔐 pkl에서 모델 + threshold + columns 로드
# ======================================================================
try:
    with open(MODEL_PATH, "rb") as f:
        loaded = pickle.load(f)

    # pkl 을 {"model":..., "threshold":..., "columns":[...]} 형태로 저장해두었다고 가정
    if isinstance(loaded, dict):
        logit_model = loaded.get("model", None)
        ADULT_MODEL_THRESHOLD = loaded.get("threshold", ADULT_MODEL_THRESHOLD)
        TRAIN_COLUMNS = loaded.get("columns", [])
    else:
        # 옛날 방식(pure model)으로 저장된 경우
        logit_model = loaded
        TRAIN_COLUMNS = []
    if logit_model is None:
        st.error("로지스틱 회귀 모델을 pkl에서 제대로 불러오지 못했습니다.")
except Exception as e:
    st.error(f"[ERROR] 모델 로드 실패: {e}")
    logit_model = None
    TRAIN_COLUMNS = []

# ======================================================================
# 📝 공통 함수
# ======================================================================


def load_teen_model_results_from_file(path: str = "teen_model_results.json"):
    """미리 계산해 둔 청소년 비만 예측 모델 결과를 파일에서 불러옵니다."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def classify_adult_obesity(height_cm, weight_kg):
    """요청된 새로운 비만 분류 기준 적용"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        obe_level = 1.0  # 저체중
    elif 18.5 <= bmi < 23.0:
        obe_level = 2.0  # 정상
    elif 23.0 <= bmi < 25.0:
        obe_level = 3.0  # 비만전단계
    elif 25.0 <= bmi < 30.0:
        obe_level = 4.0  # 1단계 비만
    else:
        obe_level = 5.0  # 2단계 비만 (30.0 이상)

    return bmi, obe_level


def get_br_fq_label(br_fq_code):
    """L_BR_FQ 코드에 따른 한국어 레이블 반환"""
    mapping = {
        1.0: "매일",
        2.0: "주 5~6회",
        3.0: "주 3~4회",
        4.0: "주 1~2회",
        5.0: "월 1회",
        6.0: "거의 안 먹음",
    }
    return mapping.get(br_fq_code, f"{int(br_fq_code)}회 (미분류)")


def get_br_fq_select_options():
    """Streamlit 선택 상자를 위한 옵션 생성"""
    return {
        "매일 (1.0)": 1.0,
        "주 5~6회 (2.0)": 2.0,
        "주 3~4회 (3.0)": 3.0,
        "주 1~2회 (4.0)": 4.0,
        "월 1회 (5.0)": 5.0,
        "거의 안 먹음 (6.0)": 6.0,
    }


def prepare_adult_model_data(df: pd.DataFrame):
    """
    pkl 모델 학습 시 사용한 변수 이름에 맞춰 데이터 준비
    (DIABETES + age, sex, HE_BMI, HE_sbp, HE_dbp, HE_TG, HE_HDL_st2, DM_FH, L_BR_FQ)
    """
    req = [
        "DIABETES",
        "age",
        "sex",
        "HE_BMI",
        "HE_sbp",
        "HE_dbp",
        "HE_TG",
        "HE_HDL_st2",
        "DM_FH",
        "L_BR_FQ",
    ]
    if not set(req).issubset(df.columns):
        return None

    data = df[req].dropna().reset_index(drop=True)
    if len(data) < 100:
        return None

    y = data["DIABETES"].astype(int)
    X = data.drop(columns=["DIABETES"])
    X = sm.add_constant(X)  # const 추가
    return {"X": X, "y": y, "columns": X.columns.tolist()}


def find_best_threshold(y_true, y_prob, metric: str = "f1"):
    """
    y_prob(확률)에 대해 0.01~0.99 범위에서 임계값을 바꿔가며
    F1(또는 다른 metric)이 최대가 되는 threshold를 찾음.
    """
    best_t = 0.5
    best_score = -1.0

    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


def compute_adult_model_results(dataframe: pd.DataFrame, model):
    """
    이미 학습된 로지스틱 회귀 모델(pkl에서 로드한 것)을 이용해서
    성능 지표, 오즈비, ROC 곡선을 계산해서 반환합니다.
    """
    if model is None:
        return None

    prep = prepare_adult_model_data(dataframe)
    if not prep:
        return None

    X, y = prep["X"], prep["y"]

    # 1) 학습 시 사용된 컬럼 순서(TRAIN_COLUMNS)에 맞게 정렬
    if TRAIN_COLUMNS:
        X_aligned = X.reindex(columns=TRAIN_COLUMNS).fillna(0)
    else:
        # pkl에 columns 정보가 없다면 model.params 기준으로라도 맞추기
        X_aligned = X.reindex(columns=model.params.index).fillna(0)

    # 2) 확률 예측
    y_prob = model.predict(X_aligned)

    # 3) F1 기준 최적 임계값 탐색
    best_t, best_f1 = find_best_threshold(y, y_prob, metric="f1")

    # 4) 최종 예측
    y_pred = (y_prob >= best_t).astype(int)

    # 5) 성능 지표
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "f1": best_f1,
        "auc": roc_auc_score(y, y_prob),
        "threshold": float(best_t),
        "sample_size": len(y),
    }

    # 6) 오즈비 / 계수 테이블 (모델 자체 기준)
    odds_ratios = np.exp(model.params)
    coef_df = pd.DataFrame(
        {
            "Coef": model.params,
            "OR": odds_ratios,
            "P-value": model.pvalues,
        }
    )

    # 7) ROC curve 좌표 계산
    fpr, tpr, roc_thresholds = roc_curve(y, y_prob)

    results = {
        "metrics": metrics,
        "odds_summary": coef_df.to_dict("index"),
        "model_params": model.params.to_dict(),
        "model_cols": prep["columns"],
        "roc_curve": {  # ROC 좌표
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
    }
    return results


def predict_diabetes_risk_final(
    age,
    sex,
    height_cm,
    weight_kg,
    sbp,
    dbp,
    dm_fh,
    br_fq,
    model,
    hdl=ADULT_DEFAULT_HDL,
):
    """
    최종 간소화 모델을 사용하여 당뇨병 위험을 예측합니다.
    학습 시 사용한 변수 이름에 맞춰 입력을 구성합니다.
    """

    # 1. BMI 계산 및 분류
    bmi, obe_level = classify_adult_obesity(height_cm, weight_kg)

    # 2. 학습 당시와 동일한 이름으로 DataFrame 생성
    new_data = pd.DataFrame(
        {
            "const": [1.0],
            "age": [age],
            "sex": [sex],
            "HE_BMI": [bmi],
            "HE_sbp": [sbp],
            "HE_dbp": [dbp],
            "HE_TG": [0.0],  # 실시간 TG 정보가 없다면 0으로 둠
            "HE_HDL_st2": [hdl],
            "DM_FH": [dm_fh],
            "L_BR_FQ": [br_fq],
        }
    )

    # 3. 컬럼 순서 맞추기
    if TRAIN_COLUMNS:
        new_data = new_data.reindex(columns=TRAIN_COLUMNS).fillna(0)
    else:
        new_data = new_data.reindex(columns=model.params.index).fillna(0)

    # 4. 예측
    prediction_prob = model.predict(new_data)[0]

    return bmi, obe_level, prediction_prob, hdl


# 청소년용 더미 함수
def prepare_teen_model_data(dataframe: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
    return None


def compute_teen_model_results(dataframe: pd.DataFrame):
    return None


# ======================================================================
# 🔄 데이터 로드
# ======================================================================


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("9ch_final_data.csv")
        df["BMI"] = df["WT"] / ((df["HT"] / 100) ** 2)
    except FileNotFoundError:
        df = pd.DataFrame()
    return df


@st.cache_data
def load_new_data():
    """성인 모델에 필요한 변수 매핑 및 파생 변수 생성"""
    try:
        df_new = pd.read_csv("hn_cleand_data.csv")
    except FileNotFoundError:
        return pd.DataFrame()

    # 1) 대시보드용 한글/대문자 컬럼명으로 매핑
    df_new = df_new.rename(
        columns={
            "year": "YEAR",
            "age": "AGE",
            "sex": "SEX",
            "region": "REGION",
            "ho_incm5": "INCOME",
            "HE_ht": "HT",
            "HE_wt": "WT",
            "HE_BMI": "BMI",
            "HE_obe": "OBESITY",
            "HE_glu": "GLUCOSE",
            "HE_HbA1c": "HbA1c",
            "DE1_dg": "DIABETES",
            "L_BR_FQ": "BREAKFAST",
            "HE_sbp": "SBP",
            "HE_dbp": "DBP",
            "HE_DMfh1": "DM_FH1",
            "HE_DMfh2": "DM_FH2",
            "HE_HDL_st2": "HDL",
            "LS_FRUIT": "F_FRUIT",
            "LS_VEG1": "F_VEG",
        }
    )

    # 2) 가족력 변수 통합
    if "DM_FH1" in df_new.columns and "DM_FH2" in df_new.columns:
        df_new["DM_FH"] = (
            (df_new["DM_FH1"] == 1) | (df_new["DM_FH2"] == 1)
        ).astype(int)


    # 4) pkl 모델이 학습될 때 사용한 "원래 이름" 컬럼들도 다시 만들어주기
    if {
        "AGE",
        "SEX",
        "BMI",
        "SBP",
        "DBP",
        "HDL",
        "BREAKFAST",
    }.issubset(df_new.columns):
        df_new["age"] = df_new["AGE"]
        df_new["sex"] = df_new["SEX"]
        df_new["HE_BMI"] = df_new["BMI"]
        df_new["HE_sbp"] = df_new["SBP"]
        df_new["HE_dbp"] = df_new["DBP"]
        df_new["HE_HDL_st2"] = df_new["HDL"]
        df_new["L_BR_FQ"] = df_new["BREAKFAST"]
        # HE_TG 는 원본 이름 그대로 존재한다고 가정

    return df_new


# 실제 데이터 로드
df = load_data()
df_new = load_new_data()

# 청소년 데이터 전처리
teen_bmi_cutoff = None
if not df.empty:
    df = df[~df["YEAR"].isin(TEEN_EXCLUDED_YEARS)].copy()
    if df["BMI"].notna().any():
        teen_bmi_cutoff = df["BMI"].quantile(TEEN_OBESITY_PERCENTILE)
        df["TEEN_OBESE_TOP5"] = (df["BMI"] >= teen_bmi_cutoff).astype(int)
    else:
        df["TEEN_OBESE_TOP5"] = np.nan
    df["HEALTHY_SCORE"] = (
        df[["F_FRUIT", "F_VEG"]].sum(axis=1)
        if "F_FRUIT" in df.columns and "F_VEG" in df.columns
        else np.nan
    )
    df["UNHEALTHY_SCORE"] = (
        df[["F_FASTFOOD", "SODA_INTAKE"]].sum(axis=1)
        if "F_FASTFOOD" in df.columns and "SODA_INTAKE" in df.columns
        else np.nan
    )
    df["NET_DIET_SCORE"] = (
        df["HEALTHY_SCORE"] - df["UNHEALTHY_SCORE"]
        if df["HEALTHY_SCORE"].notna().any()
        else np.nan
    )
    if "GROUP" in df.columns:
        df["GROUP"] = df["GROUP"].fillna("Unknown").astype(str)
    if "CTYPE" in df.columns:
        df["CTYPE"] = df["CTYPE"].fillna("Unknown").astype(str)
else:
    df["TEEN_OBESE_TOP5"] = np.nan
    df["HEALTHY_SCORE"] = np.nan
    df["UNHEALTHY_SCORE"] = np.nan
    df["NET_DIET_SCORE"] = np.nan

# ⚡️ 성인 모델 성능 계산
adult_model_results_global = compute_adult_model_results(df_new, logit_model)
adult_model_summary_global = (
    adult_model_results_global.get("metrics") if adult_model_results_global else None
)
adult_model_coefs = (
    adult_model_results_global.get("model_params")
    if adult_model_results_global
    else None
)

# 성인 모델에서 찾은 best threshold를 전역 상수로 재설정
if adult_model_summary_global:
    ADULT_MODEL_THRESHOLD = adult_model_summary_global["threshold"]

teen_model_results_global = load_teen_model_results_from_file()
teen_model_summary_global = (
    teen_model_results_global.get("logistic") if teen_model_results_global else None
)

# ======================================================================
# 🌐 Streamlit UI 시작
# ======================================================================

st.set_page_config(
    page_title="건강 데이터 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 사이드바 - 데이터셋 선택
st.sidebar.header("📊 데이터셋 선택")
dataset_choice = st.sidebar.radio(
    "분석할 데이터셋을 선택하세요", ["청소년 데이터", "성인 데이터"], index=1
)

if dataset_choice == "청소년 데이터":
    current_df = df
    is_adult = False
else:
    current_df = df_new
    is_adult = True

# 사이드바 필터
st.sidebar.header("🔍 필터 옵션")

years = sorted(current_df["YEAR"].unique()) if "YEAR" in current_df.columns else []
selected_years = st.sidebar.multiselect("연도 선택", options=years, default=years)

sex_options = ["전체", "남성", "여성"]
selected_sex = st.sidebar.selectbox("성별 선택", sex_options)

min_age = (
    int(current_df["AGE"].min())
    if not current_df.empty
    and "AGE" in current_df.columns
    and current_df["AGE"].notna().any()
    else 0
)
max_age = (
    int(current_df["AGE"].max())
    if not current_df.empty
    and "AGE" in current_df.columns
    and current_df["AGE"].notna().any()
    else 100
)
age_range = st.sidebar.slider(
    "연령 범위", min_value=min_age, max_value=max_age, value=(min_age, max_age)
)

filtered_df = current_df.copy()
if not filtered_df.empty:
    if "YEAR" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["YEAR"].isin(selected_years)]
    if "AGE" in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df["AGE"] >= age_range[0])
            & (filtered_df["AGE"] <= age_range[1])
        ]
    if selected_sex == "남성":
        filtered_df = filtered_df[filtered_df["SEX"] == 1.0]
    elif selected_sex == "여성":
        filtered_df = filtered_df[filtered_df["SEX"] == 2.0]
else:
    filtered_df = pd.DataFrame()


# 메인 타이틀
st.title("📊 건강 데이터 분석 대시보드")
st.markdown("---")

# KPI
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("총 데이터 수", f"{len(filtered_df):,}개")
with col2:
    avg_height = (
        filtered_df["HT"].dropna().mean() if "HT" in filtered_df.columns else np.nan
    )
    st.metric("평균 키", f"{avg_height:.1f}cm" if not pd.isna(avg_height) else "N/A")
with col3:
    avg_weight = (
        filtered_df["WT"].dropna().mean() if "WT" in filtered_df.columns else np.nan
    )
    st.metric(
        "평균 몸무게", f"{avg_weight:.1f}kg" if not pd.isna(avg_weight) else "N/A"
    )
with col4:
    avg_bmi = (
        filtered_df["BMI"].dropna().mean()
        if "BMI" in filtered_df.columns
        else np.nan
    )
    st.metric("평균 BMI", f"{avg_bmi:.2f}" if not pd.isna(avg_bmi) else "N/A")
with col5:
    total_records = len(df) if not is_adult else len(df_new)
    filtered_ratio = (len(filtered_df) / total_records * 100) if total_records > 0 else 0
    st.metric("필터링 비율", f"{filtered_ratio:.1f}%")

st.markdown("---")

# 탭
tab_names = [
    "📈 개요",
    "👥 인구통계",
    "🍎 건강/식습관",
    "📊 상관관계",
    "📋 데이터",
    "🤖 모델 성능",
    "🧑‍💻 성인 예측",
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tab_names)

# ---------------- 탭 1: 개요 ----------------
with tab1:
    st.header("데이터 개요")
    col1_, col2_ = st.columns(2)
    with col1_:
        year_counts = (
            filtered_df["YEAR"].value_counts().sort_index()
            if "YEAR" in filtered_df.columns
            else pd.Series()
        )
        if len(year_counts) > 0:
            fig = px.bar(
                x=year_counts.index,
                y=year_counts.values,
                labels={"x": "연도", "y": "빈도"},
                title="연도별 데이터 분포",
                color=year_counts.values,
                color_continuous_scale="Blues",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col2_:
        sex_counts = (
            filtered_df["SEX"].value_counts()
            if "SEX" in filtered_df.columns
            else pd.Series()
        )
        sex_labels = {1.0: "남성", 2.0: "여성"}
        if len(sex_counts) > 0:
            fig = px.pie(
                values=sex_counts.values,
                names=[sex_labels.get(x, x) for x in sex_counts.index],
                title="성별 분포",
                color_discrete_sequence=["#ff9999", "#66b3ff"],
            )
            st.plotly_chart(fig, use_container_width=True)

    col3_, col4_ = st.columns(2)
    with col3_:
        age_counts = (
            filtered_df["AGE"].value_counts().sort_index()
            if "AGE" in filtered_df.columns
            else pd.Series()
        )
        if len(age_counts) > 0:
            fig = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                labels={"x": "나이", "y": "빈도"},
                title="연령 분포",
                color=age_counts.values,
                color_continuous_scale="Greens",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with col4_:
        if "CTYPE" in filtered_df.columns:
            ctype_counts = filtered_df["CTYPE"].value_counts()
            if len(ctype_counts) > 0:
                fig = px.bar(
                    x=ctype_counts.index,
                    y=ctype_counts.values,
                    labels={"x": "도시 유형", "y": "빈도"},
                    title="도시 유형별 분포",
                    color=ctype_counts.values,
                    color_continuous_scale="Teal",
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        elif "REGION" in filtered_df.columns:
            region_counts = filtered_df["REGION"].value_counts().sort_index()
            if len(region_counts) > 0:
                fig = px.bar(
                    x=region_counts.index,
                    y=region_counts.values,
                    labels={"x": "지역", "y": "빈도"},
                    title="지역별 분포",
                    color=region_counts.values,
                    color_continuous_scale="Teal",
                )
                fig.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

# ---------------- 탭 2: 인구통계 ----------------
with tab2:
    st.header("인구통계 분석")
    col1_, col2_ = st.columns(2)
    with col1_:
        ht_data = (
            filtered_df["HT"].dropna() if "HT" in filtered_df.columns else pd.Series()
        )
        if len(ht_data) > 0:
            fig = px.histogram(
                x=ht_data,
                nbins=30,
                labels={"x": "키 (cm)", "count": "빈도"},
                title="키 분포",
                color_discrete_sequence=["coral"],
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2_:
        wt_data = (
            filtered_df["WT"].dropna() if "WT" in filtered_df.columns else pd.Series()
        )
        if len(wt_data) > 0:
            fig = px.histogram(
                x=wt_data,
                nbins=30,
                labels={"x": "몸무게 (kg)", "count": "빈도"},
                title="몸무게 분포",
                color_discrete_sequence=["gold"],
            )
            st.plotly_chart(fig, use_container_width=True)

    scatter_df = filtered_df[["HT", "WT", "AGE", "SEX", "YEAR"]].dropna(
        how="any", axis=0
    ) if set(["HT", "WT", "AGE", "SEX", "YEAR"]).issubset(filtered_df.columns) else pd.DataFrame()
    if not scatter_df.empty:
        fig = px.scatter(
            scatter_df,
            x="HT",
            y="WT",
            color="AGE",
            size="AGE",
            hover_data=["SEX", "YEAR"],
            labels={"HT": "키 (cm)", "WT": "몸무게 (kg)", "AGE": "나이"},
            title="키 vs 몸무게 (나이별 색상)",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    col3_, col4_ = st.columns(2)
    with col3_:
        year_height = (
            filtered_df.groupby("YEAR")["HT"].mean().dropna()
            if "YEAR" in filtered_df.columns and "HT" in filtered_df.columns
            else pd.Series()
        )
        if len(year_height) > 0:
            fig = px.line(
                x=year_height.index,
                y=year_height.values,
                markers=True,
                labels={"x": "연도", "y": "평균 키 (cm)"},
                title="연도별 평균 키 추이",
            )
            fig.update_traces(line_color="blue", line_width=3)
            st.plotly_chart(fig, use_container_width=True)
    with col4_:
        year_weight = (
            filtered_df.groupby("YEAR")["WT"].mean().dropna()
            if "YEAR" in filtered_df.columns and "WT" in filtered_df.columns
            else pd.Series()
        )
        if len(year_weight) > 0:
            fig = px.line(
                x=year_weight.index,
                y=year_weight.values,
                markers=True,
                labels={"x": "연도", "y": "평균 몸무게 (kg)"},
                title="연도별 평균 몸무게 추이",
            )
            fig.update_traces(line_color="red", line_width=3)
            st.plotly_chart(fig, use_container_width=True)

    col5_, col6_ = st.columns(2)
    with col5_:
        sex_height = (
            filtered_df.groupby("SEX")["HT"].mean().dropna()
            if "SEX" in filtered_df.columns and "HT" in filtered_df.columns
            else pd.Series()
        )
        if len(sex_height) > 0:
            sex_labels_bar = ["남성", "여성"]
            fig = px.bar(
                x=sex_labels_bar[: len(sex_height)],
                y=sex_height.values,
                labels={"x": "성별", "y": "평균 키 (cm)"},
                title="성별 평균 키 비교",
                color=sex_labels_bar[: len(sex_height)],
                color_discrete_sequence=["#ff9999", "#66b3ff"],
            )
            st.plotly_chart(fig, use_container_width=True)
    with col6_:
        sex_weight = (
            filtered_df.groupby("SEX")["WT"].mean().dropna()
            if "SEX" in filtered_df.columns and "WT" in filtered_df.columns
            else pd.Series()
        )
        if len(sex_weight) > 0:
            sex_labels_bar = ["남성", "여성"]
            fig = px.bar(
                x=sex_labels_bar[: len(sex_weight)],
                y=sex_weight.values,
                labels={"x": "성별", "y": "평균 몸무게 (kg)"},
                title="성별 평균 몸무게 비교",
                color=sex_labels_bar[: len(sex_weight)],
                color_discrete_sequence=["#ff9999", "#66b3ff"],
            )
            st.plotly_chart(fig, use_container_width=True)

    bmi_data = (
        filtered_df["BMI"].dropna()
        if "BMI" in filtered_df.columns
        else pd.Series()
    )
    if len(bmi_data) > 0:
        fig = px.histogram(
            x=bmi_data,
            nbins=30,
            labels={"x": "BMI", "count": "빈도"},
            title="BMI 분포",
            color_discrete_sequence=["pink"],
        )
        fig.add_vline(x=18.5, line_dash="dash", line_color="blue")
        fig.add_vline(x=23.0, line_dash="dash", line_color="orange")
        fig.add_vline(x=25.0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- 탭 3: 건강/식습관 ----------------
with tab3:
    if is_adult:
        st.header("🏥 건강 지표 및 식습관 분석")
        col1_, col2_ = st.columns(2)
        with col1_:
            glucose_data = (
                filtered_df["GLUCOSE"].dropna()
                if "GLUCOSE" in filtered_df.columns
                else pd.Series()
            )
            if len(glucose_data) > 0:
                fig = px.histogram(
                    x=glucose_data,
                    nbins=30,
                    labels={"x": "혈당 (mg/dL)", "count": "빈도"},
                    title="혈당 분포",
                    color_discrete_sequence=["lightblue"],
                )
                fig.add_vline(x=126, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

        with col2_:
            hba1c_data = (
                filtered_df["HbA1c"].dropna()
                if "HbA1c" in filtered_df.columns
                else pd.Series()
            )
            if len(hba1c_data) > 0:
                fig = px.histogram(
                    x=hba1c_data,
                    nbins=30,
                    labels={"x": "당화혈색소 (%)", "count": "빈도"},
                    title="당화혈색소 분포",
                    color_discrete_sequence=["lightgreen"],
                )
                fig.add_vline(x=5.7, line_dash="dash", line_color="green")
                fig.add_vline(x=6.5, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("📈 연도별 건강 지표 추이")
        col1_, col2_ = st.columns(2)
        with col1_:
            year_bmi = (
                filtered_df.groupby("YEAR")["BMI"].mean().dropna()
                if "YEAR" in filtered_df.columns and "BMI" in filtered_df.columns
                else pd.Series()
            )
            if len(year_bmi) > 0:
                fig = px.line(
                    x=year_bmi.index,
                    y=year_bmi.values,
                    markers=True,
                    labels={"x": "연도", "y": "평균 BMI"},
                    title="연도별 평균 BMI 추이",
                )
                fig.update_traces(line_color="blue", line_width=3)
                st.plotly_chart(fig, use_container_width=True)

        with col2_:
            year_glucose = (
                filtered_df.groupby("YEAR")["GLUCOSE"].mean().dropna()
                if "YEAR" in filtered_df.columns and "GLUCOSE" in filtered_df.columns
                else pd.Series()
            )
            if len(year_glucose) > 0:
                fig = px.line(
                    x=year_glucose.index,
                    y=year_glucose.values,
                    markers=True,
                    labels={"x": "연도", "y": "평균 혈당 (mg/dL)"},
                    title="연도별 평균 혈당 추이",
                )
                fig.update_traces(line_color="red", line_width=3)
                st.plotly_chart(fig, use_container_width=True)

        if "OBESITY" in filtered_df.columns:
            obesity_counts = filtered_df["OBESITY"].dropna().value_counts().sort_index()
            if len(obesity_counts) > 0:
                obesity_labels = {
                    1.0: "저체중",
                    2.0: "정상",
                    3.0: "과체중/비만",
                    4.0: "1단계 비만",
                    5.0: "2단계 비만",
                    6.0: "3단계 비만",
                }
                display_labels = [
                    obesity_labels.get(x, str(x)) for x in obesity_counts.index
                ]
                fig = px.bar(
                    x=display_labels,
                    y=obesity_counts.values,
                    labels={"x": "비만도", "y": "빈도"},
                    title="비만도 분포",
                    color=display_labels,
                    color_discrete_sequence=[
                        "lightblue",
                        "green",
                        "yellow",
                        "orange",
                        "red",
                        "darkred",
                    ],
                )
                st.plotly_chart(fig, use_container_width=True)

        # ==============================
        # 🔹 당뇨 관련 추가 시각화들
        # ==============================
        if "DIABETES" in filtered_df.columns:
            # 1) 연도별 당뇨 발병률 (성별 구분) - 기존 코드
            st.subheader("🩺 연도별 당뇨 발병률 추이 (성별 구분)")
            diabetes_data = filtered_df[["YEAR", "SEX", "DIABETES"]].dropna()
            if len(diabetes_data) > 0:

                def get_rate(d):
                    return (d["DIABETES"] == 1.0).sum() / len(d) * 100

                year_all = (
                    diabetes_data.groupby("YEAR")
                    .apply(get_rate)
                    .reset_index(name="당뇨발병률")
                )
                year_all["성별"] = "전체"

                year_male = (
                    diabetes_data[diabetes_data["SEX"] == 1.0]
                    .groupby("YEAR")
                    .apply(get_rate)
                    .reset_index(name="당뇨발병률")
                )
                year_male["성별"] = "남성"

                year_female = (
                    diabetes_data[diabetes_data["SEX"] == 2.0]
                    .groupby("YEAR")
                    .apply(get_rate)
                    .reset_index(name="당뇨발병률")
                )
                year_female["성별"] = "여성"

                comb = pd.concat([year_all, year_male, year_female], ignore_index=True)
                fig = px.line(
                    comb,
                    x="YEAR",
                    y="당뇨발병률",
                    color="성별",
                    markers=True,
                    labels={"YEAR": "연도", "당뇨발병률": "당뇨 발병률 (%)"},
                    title="연도별 당뇨 발병률 추이 (성별 구분)",
                    color_discrete_map={
                        "전체": "purple",
                        "남성": "#ff9999",
                        "여성": "#66b3ff",
                    },
                )
                fig.update_traces(line_width=3)
                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            # 2) BMI 구간별 당뇨 유병률
            # ============================
            # 📊 BMI 구간별 당뇨 유병률 (Plotly)
            # ============================
            
            st.subheader("BMI 구간별 당뇨 유병률")
            
            bmi_plot_df = filtered_df[["BMI", "DIABETES"]].dropna()
            
            if len(bmi_plot_df) > 0:
            
                bins = [0, 18.5, 23, 25, 30, np.inf]
                labels = ["저체중", "정상", "과체중", "비만", "고도비만"]
            
                bmi_plot_df["BMI_GROUP"] = pd.cut(bmi_plot_df["BMI"], bins=bins, labels=labels)
            
                grp = bmi_plot_df.groupby("BMI_GROUP")
                diab_rate = grp["DIABETES"].mean() * 100
                n_vals = grp.size()
            
                result_df = pd.DataFrame({
                    "BMI 구간": labels,
                    "당뇨 유병률(%)": [diab_rate.get(lbl, np.nan) for lbl in labels],
                    "n": [n_vals.get(lbl, 0) for lbl in labels]
                })
            
                fig = px.bar(
                    result_df,
                    x="BMI 구간",
                    y="당뇨 유병률(%)",
                    color="BMI 구간",
                    text=result_df.apply(lambda r: f"{r['당뇨 유병률(%)']:.2f}%<br>(n={r['n']:,})", axis=1),
                    title="BMI 구간별 당뇨 유병률 (Plotly)",
                    color_discrete_sequence=px.colors.sequential.Sunset
                )
            
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)


            # ============================================
            # 📈 연도별 비만율 & 당뇨 유병률 추이 + 추세선 (Plotly)
            # ============================================
            
            st.subheader("연도별 비만율과 당뇨 유병률 추이 (Plotly)")

            trend_df = filtered_df[["YEAR", "BMI", "DIABETES"]].dropna()
            
            if len(trend_df) > 0:
                trend_df["OBESE"] = (trend_df["BMI"] >= 25).astype(int)
            
                yearly = trend_df.groupby("YEAR").agg(
                    obesity_rate=("OBESE", lambda s: s.mean() * 100),
                    diabetes_rate=("DIABETES", lambda s: s.mean() * 100),
                ).reset_index()
            
                years = yearly["YEAR"].values
                ob_rate = yearly["obesity_rate"].values
                dm_rate = yearly["diabetes_rate"].values
            
                # 1차 회귀(직선) 계수
                ob_coef = np.polyfit(years, ob_rate, 1)
                dm_coef = np.polyfit(years, dm_rate, 1)
                ob_line = np.poly1d(ob_coef)
                dm_line = np.poly1d(dm_coef)
            
                fig = make_subplots(specs=[[{"secondary_y": True}]])
            
                # ⭕ 점만 (비만율)
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=ob_rate,
                        mode="markers",          # ← 점만
                        name="비만율 (%)",
                        marker=dict(color="orange", size=9),
                    ),
                    secondary_y=False,
                )
            
                # ⭕ 점만 (당뇨 유병률)
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=dm_rate,
                        mode="markers",          # ← 점만
                        name="당뇨 유병률 (%)",
                        marker=dict(color="red", size=9, symbol="square"),
                    ),
                    secondary_y=True,
                )
            
                # 📉 비만율 추세선
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=ob_line(years),
                        mode="lines",
                        line=dict(color="orange", dash="dash"),
                        name="비만율 추세선",
                    ),
                    secondary_y=False,
                )
            
                # 📉 당뇨 유병률 추세선
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=dm_line(years),
                        mode="lines",
                        line=dict(color="red", dash="dash"),
                        name="당뇨 추세선",
                    ),
                    secondary_y=True,
                )
            
                # 추세선 식 텍스트 (그래프 아래쪽에 한 번만)
                ob_a, ob_b = ob_coef
                dm_a, dm_b = dm_coef
                eq_text = (
                    f"비만율 추세선: y = {ob_a:.3f}x + {ob_b:.2f}<br>"
                    f"당뇨 추세선: y = {dm_a:.3f}x + {dm_b:.2f}"
                )
            
                fig.update_layout(
                    title="연도별 비만율과 당뇨 유병률 추이",
                    title_x=0.5,
                    xaxis_title="연도",
                    # ✅ 범례를 그래프 안 왼쪽 위로
                    legend=dict(
                        orientation="v",
                        x=0.02,
                        y=0.98,
                        xanchor="left",
                        yanchor="top",
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="rgba(0,0,0,0.2)",
                        borderwidth=1,
                        font=dict(size=10),
                    ),
                    margin=dict(t=60, b=40, l=60, r=60),
                )
            
                # ✅ 추세선 식을 그래프 안 오른쪽 아래에 박스로 표시
                fig.add_annotation(
                    x=0.98,
                    y=0.05,
                    xref="paper",
                    yref="paper",
                    xanchor="right",
                    yanchor="bottom",
                    showarrow=False,
                    text=eq_text,
                    font=dict(size=9),
                    align="right",
                    bordercolor="rgba(0,0,0,0.3)",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="rgba(255,255,255,0.9)",
                )
            
                fig.update_yaxes(title_text="비만율 (%)", secondary_y=False)
                fig.update_yaxes(title_text="당뇨 유병률 (%)", secondary_y=True)
            
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("연도, BMI, 당뇨 정보가 충분하지 않아 추세를 계산할 수 없습니다.")


            # 4) 성별 당뇨 발병률 비교 (기존 코드)
            st.subheader("📊 성별 당뇨 발병률 비교")
            dsex = filtered_df[["SEX", "DIABETES"]].dropna()
            if len(dsex) > 0:
                rates = {}
                total_pos = (dsex["DIABETES"] == 1.0).sum()
                rates["전체"] = total_pos / len(dsex) * 100
                male = dsex[dsex["SEX"] == 1.0]
                if len(male) > 0:
                    rates["남성"] = (male["DIABETES"] == 1.0).sum() / len(male) * 100
                female = dsex[dsex["SEX"] == 2.0]
                if len(female) > 0:
                    rates["여성"] = (
                        (female["DIABETES"] == 1.0).sum() / len(female) * 100
                    )
                fig = px.bar(
                    x=list(rates.keys()),
                    y=list(rates.values()),
                    labels={"x": "성별", "y": "당뇨 발병률 (%)"},
                    title="성별 당뇨 발병률 비교",
                    color=list(rates.keys()),
                    color_discrete_map={
                        "전체": "purple",
                        "남성": "#ff9999",
                        "여성": "#66b3ff",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

        if "BREAKFAST" in filtered_df.columns:
            breakfast_counts_new = (
                filtered_df["BREAKFAST"].dropna().value_counts().sort_index()
            )
            if len(breakfast_counts_new) > 0:
                breakfast_labels_new = {
                    1.0: "매일",
                    2.0: "주 5~6회",
                    3.0: "주 3~4회",
                    4.0: "주 1~2회",
                    5.0: "월 1회",
                    6.0: "거의 안 먹음",
                }
                fig = px.pie(
                    values=breakfast_counts_new.values,
                    names=[
                        breakfast_labels_new.get(x, str(x))
                        for x in breakfast_counts_new.index
                    ],
                    title="아침식사 빈도 분포",
                    color_discrete_sequence=px.colors.sequential.YlOrBr,
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig, use_container_width=True)
    else:  # 청소년 데이터
        st.header("🍎 청소년 식습관 및 건강 지표 분석")

        # -----------------------------------
        # 1) 과일/채소, 패스트푸드/탄산 섭취 분포
        # -----------------------------------
        col1_, col2_ = st.columns(2)

        with col1_:
            st.subheader("과일 · 채소 섭취 빈도")
            fruit = (
                filtered_df["F_FRUIT"].dropna()
                if "F_FRUIT" in filtered_df.columns
                else pd.Series()
            )
            veg = (
                filtered_df["F_VEG"].dropna()
                if "F_VEG" in filtered_df.columns
                else pd.Series()
            )

            if len(fruit) > 0 or len(veg) > 0:
                freq_map = {
                    1.0: "거의 안 먹음",
                    2.0: "월 1회",
                    3.0: "주 1~2회",
                    4.0: "주 3~4회",
                    5.0: "거의 매일/매일",
                }
                df_fv = pd.DataFrame(
                    {
                        "코드": list(freq_map.keys()),
                        "라벨": list(freq_map.values()),
                    }
                )

                # 코드 기준으로 count 조인
                fruit_counts = fruit.value_counts().reindex(df_fv["코드"]).fillna(0)
                veg_counts = veg.value_counts().reindex(df_fv["코드"]).fillna(0)
                df_fv["과일 섭취"] = fruit_counts.values
                df_fv["채소 섭취"] = veg_counts.values

                fig_fv = px.bar(
                    df_fv,
                    x="라벨",
                    y=["과일 섭취", "채소 섭취"],
                    barmode="group",
                    labels={"value": "명 수", "라벨": "섭취 빈도"},
                    title="과일 · 채소 섭취 빈도 분포",
                )
                fig_fv.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_fv, use_container_width=True)
            else:
                st.info("과일/채소 섭취 정보가 부족합니다.")

        with col2_:
            st.subheader("패스트푸드 · 탄산 섭취 빈도")
            ff = (
                filtered_df["F_FASTFOOD"].dropna()
                if "F_FASTFOOD" in filtered_df.columns
                else pd.Series()
            )
            soda = (
                filtered_df["SODA_INTAKE"].dropna()
                if "SODA_INTAKE" in filtered_df.columns
                else pd.Series()
            )
            if len(ff) > 0 or len(soda) > 0:
                freq_map = {
                    1.0: "거의 안 먹음",
                    2.0: "월 1회",
                    3.0: "주 1~2회",
                    4.0: "주 3~4회",
                    5.0: "거의 매일/매일",
                }
                df_us = pd.DataFrame(
                    {
                        "코드": list(freq_map.keys()),
                        "라벨": list(freq_map.values()),
                    }
                )
                ff_counts = ff.value_counts().reindex(df_us["코드"]).fillna(0)
                soda_counts = soda.value_counts().reindex(df_us["코드"]).fillna(0)
                df_us["패스트푸드"] = ff_counts.values
                df_us["탄산음료"] = soda_counts.values

                fig_us = px.bar(
                    df_us,
                    x="라벨",
                    y=["패스트푸드", "탄산음료"],
                    barmode="group",
                    labels={"value": "명 수", "라벨": "섭취 빈도"},
                    title="패스트푸드 · 탄산음료 섭취 빈도 분포",
                )
                fig_us.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_us, use_container_width=True)
            else:
                st.info("패스트푸드/탄산 섭취 정보가 부족합니다.")

        st.markdown("---")

        # -----------------------------------
        # 2) 건강/불건강 식습관 점수 & BMI/비만도
        # -----------------------------------
        col3_, col4_ = st.columns(2)

        with col3_:
            st.subheader("건강/불건강 식습관 점수")
            if (
                "HEALTHY_SCORE" in filtered_df.columns
                and "UNHEALTHY_SCORE" in filtered_df.columns
            ):
                score_df = filtered_df[["HEALTHY_SCORE", "UNHEALTHY_SCORE"]].dropna()
                if len(score_df) > 0:
                    score_long = score_df.melt(
                        value_vars=["HEALTHY_SCORE", "UNHEALTHY_SCORE"],
                        var_name="구분",
                        value_name="점수",
                    )
                    score_long["구분"] = score_long["구분"].map(
                        {
                            "HEALTHY_SCORE": "건강 식습관 점수\n(과일+채소)",
                            "UNHEALTHY_SCORE": "불건강 식습관 점수\n(패스트푸드+탄산)",
                        }
                    )
                    fig_score = px.box(
                        score_long,
                        x="구분",
                        y="점수",
                        points="all",
                        title="건강/불건강 식습관 점수 분포",
                    )
                    st.plotly_chart(fig_score, use_container_width=True)
                else:
                    st.info("식습관 점수 계산에 필요한 데이터가 부족합니다.")
            else:
                st.info("HEALTHY_SCORE / UNHEALTHY_SCORE 컬럼이 존재하지 않습니다.")

        with col4_:
            st.subheader("BMI 및 상위 5% 비만 여부")
            if "BMI" in filtered_df.columns:
                bmi_data = filtered_df["BMI"].dropna()
                if len(bmi_data) > 0:
                    fig_bmi = px.histogram(
                        bmi_data,
                        nbins=30,
                        labels={"value": "BMI", "count": "명 수"},
                        title="청소년 BMI 분포",
                        color_discrete_sequence=["#ffccbc"],
                    )
                    st.plotly_chart(fig_bmi, use_container_width=True)

            if "TEEN_OBESE_TOP5" in filtered_df.columns:
                obese_counts = (
                    filtered_df["TEEN_OBESE_TOP5"].dropna().value_counts().sort_index()
                )
                if len(obese_counts) > 0:
                    labels = {0.0: "하위 95%", 1.0: "상위 5% (고도 비만군?)"}
                    fig_ob = px.pie(
                        values=obese_counts.values,
                        names=[labels.get(x, str(x)) for x in obese_counts.index],
                        title="BMI 상위 5% 비만군 비율",
                        color_discrete_sequence=["#bbdefb", "#ef5350"],
                    )
                    fig_ob.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(fig_ob, use_container_width=True)

        st.markdown("---")

        # -----------------------------------
        # 3) 식습관 점수 vs BMI 관계
        # -----------------------------------
        st.subheader("식습관 점수와 BMI의 관계")

        if (
            "BMI" in filtered_df.columns
            and "NET_DIET_SCORE" in filtered_df.columns
        ):
            rel_df = filtered_df[["BMI", "NET_DIET_SCORE"]].dropna()
            if len(rel_df) > 0:
                fig_sc = px.scatter(
                    rel_df,
                    x="NET_DIET_SCORE",
                    y="BMI",
                    trendline="ols",
                    labels={
                        "NET_DIET_SCORE": "순 식습관 점수 (건강−불건강)",
                        "BMI": "BMI",
                    },
                    title="순 식습관 점수 vs BMI (추세선 포함)",
                )
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.info("BMI와 NET_DIET_SCORE 정보가 충분하지 않습니다.")
        else:
            st.info("BMI 또는 NET_DIET_SCORE 컬럼이 없어 관계 분석을 할 수 없습니다.")


# ---------------- 탭 4: 상관관계 ----------------
with tab4:
    st.header("상관관계 분석")
    if is_adult:
        health_cols = ["BMI", "GLUCOSE", "HbA1c", "SBP", "DBP", "HDL", "DIABETES"]
        health_cols = [c for c in health_cols if c in filtered_df.columns]
        hdata = filtered_df[health_cols].dropna()
        if len(hdata) > 0:
            corr = hdata.corr()
            fig = px.imshow(
                corr,
                labels=dict(x="변수", y="변수", color="상관계수"),
                x=health_cols,
                y=health_cols,
                color_continuous_scale="RdBu",
                aspect="auto",
                title="건강 지표 상관관계 히트맵",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.header("청소년 상관관계 분석")

        if filtered_df.empty:
            st.info("필터링된 청소년 데이터가 없습니다.")
        else:
            # BMI와 식습관 관련 점수들 중심으로 상관관계
            corr_cols = []
            for c in ["BMI", "HEALTHY_SCORE", "UNHEALTHY_SCORE", "NET_DIET_SCORE", "WT", "HT"]:
                if c in filtered_df.columns:
                    corr_cols.append(c)

            if len(corr_cols) < 2:
                st.info("상관관계를 계산할 수 있는 연속형 변수가 충분하지 않습니다.")
            else:
                corr_df = filtered_df[corr_cols].dropna()
                if len(corr_df) == 0:
                    st.info("상관관계 분석에 사용할 유효 데이터가 없습니다.")
                else:
                    corr = corr_df.corr()

                    pretty_names = {
                        "BMI": "BMI",
                        "WT": "체중",
                        "HT": "키",
                        "HEALTHY_SCORE": "건강 식습관 점수",
                        "UNHEALTHY_SCORE": "불건강 식습관 점수",
                        "NET_DIET_SCORE": "순 식습관 점수",
                    }
                    xlabels = [pretty_names.get(c, c) for c in corr.columns]
                    ylabels = [pretty_names.get(c, c) for c in corr.index]

                    fig_corr = px.imshow(
                        corr,
                        x=xlabels,
                        y=ylabels,
                        color_continuous_scale="RdBu",
                        zmin=-1,
                        zmax=1,
                        labels=dict(color="상관계수"),
                        title="청소년 BMI 및 식습관 지표 상관관계 히트맵",
                    )
                    fig_corr.update_xaxes(side="bottom")
                    st.plotly_chart(fig_corr, use_container_width=True)


# ---------------- 탭 5: 데이터 ----------------
with tab5:
    st.header("데이터 테이블")
    st.subheader("📊 통계 요약")
    col1_, col2_, col3_ = st.columns(3)
    with col1_:
        st.write("**기본 정보**")
        st.write(f"- 총 데이터 수: {len(filtered_df):,}개")
        if "YEAR" in filtered_df.columns:
            st.write(
                f"- 연도 범위: {filtered_df['YEAR'].min()} ~ {filtered_df['YEAR'].max()}"
            )
        else:
            st.write("- 연도: N/A")
        if "AGE" in filtered_df.columns:
            st.write(
                f"- 나이 범위: {filtered_df['AGE'].min()} ~ {filtered_df['AGE'].max()}세"
            )
        else:
            st.write("- 나이: N/A")
    with col2_:
        st.write("**평균값**")
        if "HT" in filtered_df.columns:
            st.write(f"- 평균 키: {filtered_df['HT'].mean():.2f}cm")
        else:
            st.write("- 평균 키: N/A")
        if "WT" in filtered_df.columns:
            st.write(f"- 평균 몸무게: {filtered_df['WT'].mean():.2f}kg")
        else:
            st.write("- 평균 몸무게: N/A")
        if "BMI" in filtered_df.columns:
            st.write(f"- 평균 BMI: {filtered_df['BMI'].mean():.2f}")
        else:
            st.write("- 평균 BMI: N/A")
    with col3_:
        st.write("**분포**")
        if "SEX" in filtered_df.columns:
            sc = filtered_df["SEX"].value_counts()
            for v, c in sc.items():
                name = "남성" if v == 1.0 else "여성"
                st.write(f"- {name}: {c:,}명")

    st.markdown("---")
    st.subheader("필터링된 데이터")
    st.dataframe(filtered_df, use_container_width=True)

# ---------------- 탭 6: 모델 성능 ----------------
with tab6:
    if is_adult:
        st.header("🤖 성인 당뇨병 예측 모델 성능")
        if adult_model_summary_global:
            metrics = adult_model_summary_global
            st.markdown(
                f"- **모델**: Logistic Regression (statsmodels)\n"
                f"- **라벨 기준**: DIABETES=1.0 (의사 진단 여부)\n"
                f"- **적용 임계값 (F1 기준 최적화)**: **{metrics['threshold']:.4f}**"
            )
            metrics_chart = pd.DataFrame(
                {
                    "지표": ["Accuracy", "Recall", "Precision", "F1-Score", "AUC-ROC"],
                    "값": [
                        metrics["accuracy"],
                        metrics["recall"],
                        metrics["precision"],
                        metrics["f1"],
                        metrics["auc"],
                    ],
                }
            )
            fig = px.bar(
                metrics_chart,
                x="지표",
                y="값",
                title="성인 모델 성능 지표",
                color="지표",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            c2.metric("Recall (선별력)", f"{metrics['recall']*100:.1f}%")
            c3.metric("Precision", f"{metrics['precision']*100:.1f}%")
            c4, c5 = st.columns(2)
            c4.metric("F1-Score", f"{metrics['f1']*100:.1f}%")
            c5.metric("AUC-ROC", f"{metrics['auc']:.3f}")
            st.caption(f"학습 표본 수: {metrics['sample_size']:,}건")

            # =========================
            # 📉 ROC Curve 시각화 추가
            # =========================
            if adult_model_results_global and "roc_curve" in adult_model_results_global:
                roc_info = adult_model_results_global["roc_curve"]
                fpr = np.array(roc_info["fpr"])
                tpr = np.array(roc_info["tpr"])

                roc_fig = go.Figure()

                # 모델 ROC 곡선
                roc_fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"ROC (AUC = {metrics['auc']:.3f})",
                        line=dict(width=3, color="firebrick"),
                    )
                )

                # 무작위 분류 기준선 (대각선)
                roc_fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="무작위 분류",
                        line=dict(width=2, dash="dash", color="gray"),
                        showlegend=True,
                    )
                )

                roc_fig.update_layout(
                    title="ROC Curve (민감도-1-특이도)",
                    xaxis_title="1 - 특이도 (False Positive Rate)",
                    yaxis_title="민감도 (True Positive Rate)",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                    height=450,
                )

                st.plotly_chart(roc_fig, use_container_width=True)

            st.markdown("---")
            st.subheader("📊 주요 위험 요인 오즈비 (Odds Ratio)")
            if adult_model_results_global and "odds_summary" in adult_model_results_global:
                odds_df = pd.DataFrame(
                    adult_model_results_global["odds_summary"]
                ).T.drop("const", errors="ignore")
                odds_df = odds_df.rename(
                    columns={"OR": "오즈비(OR)", "P-value": "p-value"}
                ).round(4)
                feature_map = {
                    "age": "나이 (1세당)",
                    "sex": "성별 (남성=1, 여성=2)",
                    "HE_BMI": "BMI (1kg/m²당)",
                    "HE_sbp": "수축기 혈압 (1mmHg당)",
                    "HE_dbp": "이완기 혈압 (1mmHg당)",
                    "HE_TG": "중성지방 (1mg/dL당)",
                    "HE_HDL_st2": "HDL-C (1mg/dL당)",
                    "DM_FH": "가족력 (있음)",
                    "L_BR_FQ": "아침식사 빈도 (1코드당)",
                }
                odds_df.index = [feature_map.get(idx, idx) for idx in odds_df.index]
                odds_df = odds_df.sort_values("오즈비(OR)", ascending=False)
                st.dataframe(odds_df, use_container_width=True)
                # =============================
                # 🎯 오즈비(OR) 상위 4~5개 인포그래픽 스타일 시각화
                # =============================
                or_df = odds_df.copy()
            
                # OR과 1 사이의 거리(로그 스케일)로 "영향력" 정렬
                or_df["log_dist"] = (np.log(or_df["오즈비(OR)"]).abs())
            
                # 상위 5개만 사용 (4개만 원하면 head(4)로 바꿔도 됨)
                top_k = min(5, len(or_df))
                top_or = (
                    or_df.sort_values("log_dist", ascending=False)
                    .head(top_k)
                    .reset_index()
                    .rename(columns={"index": "변수"})
                )
            
                # 증가/감소 방향, 색, 표시 텍스트 만들기
                top_or["direction"] = np.where(top_or["오즈비(OR)"] >= 1, "증가", "감소")
                top_or["color"] = np.where(top_or["오즈비(OR)"] >= 1, "#ff4d4d", "#2979ff")
            
                # 멀티라인 텍스트 (Plotly는 <br>로 줄바꿈)
                top_or["label_text"] = top_or.apply(
                    lambda r: f"당뇨병<br>위험<br>{r['오즈비(OR)']:.2f}배<br>{r['direction']}",
                    axis=1,
                )
            
                # 그래프 생성
                fig_or = go.Figure()
            
                fig_or.add_trace(
                    go.Bar(
                        x=top_or["변수"],
                        y=top_or["오즈비(OR)"],
                        marker_color=top_or["color"],
                        text=top_or["label_text"],
                        textposition="inside",
                        insidetextanchor="middle",
                    )
                )
            
                # 기준선 OR = 1.0
                fig_or.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="기준 (OR = 1.0)",
                    annotation_position="top right",
                )
            
                # 레이아웃 꾸미기
                max_or = float(top_or["오즈비(OR)"].max())
                fig_or.update_layout(
                    title=f"주요 변수별 당뇨병 위험 오즈비 (상위 {top_k}개)",
                    xaxis_title="변수",
                    yaxis_title="오즈비(OR)",
                    yaxis=dict(range=[0, max_or * 1.2]),
                    margin=dict(t=60, b=40, l=40, r=20),
                    showlegend=False,
                )
            
                st.plotly_chart(fig_or, use_container_width=True)


        else:
            st.warning(
                "성인 모델 학습에 필요한 데이터(DIABETES, SBP, HDL 등)가 부족하거나 누락되었습니다."
            )
    else:
        # ============================
        # 👦 청소년 비만 예측 모델 성능
        # ============================
        st.header("👦 청소년 비만 예측 모델 성능")

        if teen_model_summary_global is None:
            st.info(
                "미리 계산된 청소년 모델 결과 파일(teen_model_results.json)을 "
                "찾을 수 없거나 'logistic' 결과가 없습니다."
            )

        else:
            # --- 1) 성능 지표 꺼내기 ---
            summary = teen_model_summary_global

            # logistic 안에 metrics가 있을 수도, summary 자체가 metrics일 수도 있음
            if isinstance(summary, dict) and "metrics" in summary:
                metrics = summary["metrics"]
            else:
                metrics = summary

            acc = metrics.get("accuracy")
            rec = metrics.get("recall")
            prec = metrics.get("precision")
            f1 = metrics.get("f1")
            auc = (
                metrics.get("auc")
                or metrics.get("roc_auc")
                or metrics.get("auroc")
            )
            thr = metrics.get("threshold") or metrics.get("cutoff")
            n_sample = metrics.get("sample_size")

            st.markdown(
                f"""
- **모델**: Logistic Regression (청소년 비만 상위 5% 예측용)  
- **라벨 기준**: BMI 상위 5% (TEEN_OBESE_TOP5 = 1)  
- **적용 임계값 (F1 기준 최적화)**: `{thr:.3f}`  
                """
                if thr is not None
                else """
- **모델**: Logistic Regression (청소년 비만 상위 5% 예측용)  
- **라벨 기준**: BMI 상위 5% (TEEN_OBESE_TOP5 = 1)  
- **적용 임계값**: (정보 없음)
                """
            )

            # --- 1-1) 바 차트용 DF ---
            rows = []
            if acc is not None:
                rows.append({"지표": "Accuracy", "값": acc})
            if rec is not None:
                rows.append({"지표": "Recall", "값": rec})
            if prec is not None:
                rows.append({"지표": "Precision", "값": prec})
            if f1 is not None:
                rows.append({"지표": "F1-Score", "값": f1})
            if auc is not None:
                rows.append({"지표": "AUC-ROC", "값": auc})

            if rows:
                teen_metric_df = pd.DataFrame(rows)
                fig = px.bar(
                    teen_metric_df,
                    x="지표",
                    y="값",
                    title="청소년 모델 성능 지표",
                    color="지표",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            # --- 1-2) KPI 카드 ---
            col1, col2, col3 = st.columns(3)
            if acc is not None:
                col1.metric("Accuracy", f"{acc*100:.1f}%")
            if rec is not None:
                col2.metric("Recall", f"{rec*100:.1f}%")
            if prec is not None:
                col3.metric("Precision", f"{prec*100:.1f}%")

            col4, col5 = st.columns(2)
            if f1 is not None:
                col4.metric("F1-Score", f"{f1*100:.1f}%")
            if auc is not None:
                col5.metric("AUC-ROC", f"{auc:.3f}")

            if n_sample is not None:
                st.caption(f"학습 표본 수: {int(n_sample):,}건")

            st.markdown("---")

            # ======================
            # 2) ROC 커브 그리기
            # ======================
            roc_source = None

            # logistic 안에서 roc 관련 키 찾기
            if isinstance(summary, dict):
                if "roc_curve" in summary:
                    roc_source = summary["roc_curve"]
                else:
                    # 이름이 애매하게 들어간 경우(예: 'roc' 포함)
                    for k, v in summary.items():
                        if isinstance(k, str) and "roc" in k.lower():
                            roc_source = v
                            break

            # logistic 바깥에서 roc_* 찾기
            if roc_source is None and isinstance(teen_model_results_global, dict):
                for k, v in teen_model_results_global.items():
                    if isinstance(k, str) and "roc" in k.lower():
                        roc_source = v
                        break

            if roc_source is not None:
                fpr = np.array(roc_source.get("fpr", []), dtype=float)
                tpr = np.array(roc_source.get("tpr", []), dtype=float)

                if fpr.size > 0 and tpr.size > 0:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode="lines",
                            name="ROC 곡선",
                        )
                    )
                    fig_roc.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="무작위 기준선",
                            line=dict(dash="dash"),
                        )
                    )
                    fig_roc.update_layout(
                        title="청소년 모델 ROC 곡선",
                        xaxis_title="1 - Specificity (FPR)",
                        yaxis_title="Sensitivity (TPR)",
                        xaxis=dict(range=[0, 1]),
                        yaxis=dict(range=[0, 1]),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                        ),
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("teen_model_results에서 ROC 좌표(fpr/tpr)를 찾지 못했습니다.")
            else:
                st.info("청소년 모델 ROC 정보(roc_curve)가 JSON에 포함되어 있지 않습니다.")

            st.markdown("---")

            # ======================
            # 3) 오즈비(OR) 시각화
            # ======================
            odds_source = None

            # logistic 안에서 odds 관련 키 찾기
            if isinstance(summary, dict):
                if "odds_summary" in summary:
                    odds_source = summary["odds_summary"]
                else:
                    for k, v in summary.items():
                        if isinstance(k, str) and "odds" in k.lower():
                            odds_source = v
                            break

            # logistic 바깥에서 odds_* 찾기
            if odds_source is None and isinstance(teen_model_results_global, dict):
                for k, v in teen_model_results_global.items():
                    if isinstance(k, str) and "odds" in k.lower():
                        odds_source = v
                        break

            if odds_source is None:
                st.info("청소년 모델 오즈비(odds) 정보가 JSON에 포함되어 있지 않습니다.")
            else:
                try:
                    # odds_source 형태에 따라 DataFrame 생성 방식 분기
                    if isinstance(odds_source, dict):
                        # value 중 하나를 샘플로 가져와서 dict 여부 확인
                        first_val = next(iter(odds_source.values()))
                        if isinstance(first_val, dict):
                            # {변수명: {Coef:..., OR:..., P-value:...}} 형태
                            odds_df = pd.DataFrame(odds_source).T
                        else:
                            # {변수명: OR값, ...} 또는 {OR:..., P-value:...} 형태
                            odds_df = pd.DataFrame([odds_source])
                    else:
                        # list 등 예상 밖 구조면 그대로 DF로 시도
                        odds_df = pd.DataFrame(odds_source)

                except Exception as e:
                    st.info(f"오즈비 정보를 표로 변환하는 데 실패했습니다: {e}")
                    odds_df = None

                if odds_df is None or odds_df.empty:
                    st.info("오즈비(OR) 표 데이터가 비어 있어 시각화할 수 없습니다.")
                else:
                    # OR 컬럼 이름 찾기 (대소문자/표기 변형 대응)
                    or_col = None
                    for c in odds_df.columns:
                        cname = str(c).lower()
                        if cname in ["or", "odds", "odds_ratio", "oddsratio"]:
                            or_col = c
                            break
                    if or_col is None and "OR" in odds_df.columns:
                        or_col = "OR"

                    if or_col is None:
                        st.info("오즈비(OR) 컬럼을 찾지 못했습니다.")
                    else:
                        # const 행 제거 (있으면)
                        odds_df = odds_df.drop(index="const", errors="ignore")

                        # 값은 float 로 캐스팅
                        odds_df[or_col] = odds_df[or_col].astype(float)

                        # 상위 5개: |log(OR)| 기준
                        odds_df["log_or_abs"] = np.abs(
                            np.log(odds_df[or_col].replace(0, np.nan))
                        )
                        odds_df = odds_df.dropna(subset=["log_or_abs"])
                        if odds_df.empty:
                            st.info("오즈비 값이 0 또는 비정상이라 상위 변수를 계산할 수 없습니다.")
                        else:
                            top_df = odds_df.sort_values(
                                "log_or_abs", ascending=False
                            ).head(5)

                            # 보기 좋게 index → 한글 라벨 매핑 (필요하면 수정)
                            name_map = {
                                "AGE": "나이 (1세당)",
                                "SEX": "성별 (남=1, 여=2)",
                                "BMI": "BMI (1kg/m²당)",
                                "SBP": "수축기 혈압",
                                "DBP": "이완기 혈압",
                                "DM_FH": "가족력 (있음)",
                                "BREAKFAST": "아침식사 빈도",
                            }
                            disp_names = [
                                name_map.get(str(idx), str(idx))
                                for idx in top_df.index
                            ]
                            top_or = top_df[or_col].values

                            colors = [
                                "#e57373" if v > 1 else "#64b5f6" for v in top_or
                            ]
                            texts = [
                                f"위험 {v:.2f}배 증가" if v > 1 else f"위험 {v:.2f}배 감소"
                                for v in top_or
                            ]

                            fig_or = go.Figure()
                            fig_or.add_trace(
                                go.Bar(
                                    x=disp_names,
                                    y=top_or,
                                    marker_color=colors,
                                    text=texts,
                                    textposition="outside",
                                )
                            )
                            fig_or.add_hline(
                                y=1.0,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="기준선 (OR=1.0)",
                                annotation_position="top right",
                            )
                            fig_or.update_layout(
                                title="주요 변수별 비만 위험 오즈비 (상위 5개)",
                                yaxis_title="오즈비 (OR)",
                                xaxis_title="변수",
                            )
                            st.plotly_chart(fig_or, use_container_width=True)


# ---------------- 탭 7: 성인 예측 ----------------
with tab7:
    st.header("🧑‍💻 성인 당뇨병 위험 예측기")
    st.markdown("---")

    if logit_model is None:
        st.warning(
            "모델 학습에 필요한 데이터가 부족하거나 pkl 로드에 실패하여 예측기를 사용할 수 없습니다."
        )
    else:
        st.subheader("1. 신체 및 인구통계 정보 입력")
        ca, cs, ch, cw = st.columns(4)
        with ca:
            age_input = st.slider("나이 (세)", min_value=19, max_value=100, value=45)
        with cs:
            sex_label = st.selectbox(
                "성별", options=["남성 (1.0)", "여성 (2.0)"], index=0
            )
            sex_input = 1.0 if "남성" in sex_label else 2.0
        with ch:
            height_input = st.number_input(
                "키 (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1
            )
        with cw:
            weight_input = st.number_input(
                "몸무게 (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.1
            )

        bmi_current, obe_level_current = classify_adult_obesity(
            height_input, weight_input
        )
        bmi_label_map = {
            1.0: "저체중",
            2.0: "정상",
            3.0: "비만전단계",
            4.0: "1단계 비만",
            5.0: "2단계 비만 이상",
        }
        st.info(
            f"계산된 BMI: **{bmi_current:.2f} kg/m²** "
            f"(분류: **{bmi_label_map.get(obe_level_current, '미분류')}**)"
        )

        st.subheader("2. 건강 지표 및 생활 습관 입력")
        csbp, cdbp, chdl, cfh = st.columns(4)
        with csbp:
            sbp_input = st.number_input(
                "수축기 혈압 (SBP)", min_value=80.0, max_value=200.0, value=120.0, step=1.0
            )
        with cdbp:
            dbp_input = st.number_input(
                "이완기 혈압 (DBP)", min_value=50.0, max_value=120.0, value=80.0, step=1.0
            )
        with chdl:
            hdl_input_val = st.number_input(
                f"HDL-C (mg/dL) (생략 시 {ADULT_DEFAULT_HDL:.1f})",
                min_value=10.0,
                max_value=100.0,
                value=ADULT_DEFAULT_HDL,
                step=1.0,
            )
        with cfh:
            dm_fh_label = st.selectbox(
                "당뇨병 가족력", options=["없음 (0)", "있음 (1)"], index=0
            )
            dm_fh_input = 1 if "있음" in dm_fh_label else 0

        br_options = get_br_fq_select_options()
        br_label = st.selectbox("아침 식사 빈도", options=list(br_options.keys()), index=0)
        br_fq_input = br_options[br_label]

        st.markdown("---")
        if st.button("당뇨병 위험 확률 예측하기", type="primary"):
            try:
                used_hdl = (
                    hdl_input_val
                    if hdl_input_val != ADULT_DEFAULT_HDL
                    else ADULT_DEFAULT_HDL
                )
                bmi_res, obe_res, prob_res, used_hdl = predict_diabetes_risk_final(
                    age_input,
                    sex_input,
                    height_input,
                    weight_input,
                    sbp_input,
                    dbp_input,
                    dm_fh_input,
                    br_fq_input,
                    logit_model,
                    hdl=used_hdl,
                )
                st.subheader("🔮 예측 결과")
                cp, cr = st.columns(2)
                with cp:
                    st.metric("예측된 당뇨병 발병 확률", f"{prob_res * 100:.2f}%")

                risk_status = "❌ 위험군 아님"
                risk_color = "green"
                if prob_res >= ADULT_MODEL_THRESHOLD:
                    risk_status = "✅ 고위험군 (추가 검사 권고)"
                    risk_color = "red"
                with cr:
                    st.markdown(
                        f"<p style='font-size: 24px; color:{risk_color};'><b>{risk_status}</b></p>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")
                st.markdown("#### 입력 데이터 요약")
                st.markdown(
                    f"""
                - **계산된 BMI:** {bmi_res:.2f} kg/m² ({bmi_label_map.get(obe_res, '미분류')})
                - **가족력:** {'있음' if dm_fh_input == 1 else '없음'}
                - **HDL-C:** {used_hdl:.2f} mg/dL
                - **아침 식사 빈도:** {get_br_fq_label(br_fq_input)}
                - **현재 사용 중인 분류 임계값:** {ADULT_MODEL_THRESHOLD:.4f}
                """
                )
            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")

# 사이드바 하단 정보
st.sidebar.markdown("---")
if len(current_df) > 0:
    st.sidebar.info(
        f"""
    **현재 필터링된 데이터:**
    - {len(filtered_df):,}개 행
    - 전체 데이터의 {len(filtered_df)/len(current_df)*100:.1f}%
    """
    )
else:
    st.sidebar.info("현재 데이터가 없습니다.")
