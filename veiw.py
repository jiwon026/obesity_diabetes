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
)
import statsmodels.api as sm
import pickle  # 모델 저장/로드 라이브러리 추가

# ====== 전역 상수 ======
TEEN_EXCLUDED_YEARS = {2015, 2016}
TEEN_OBESITY_PERCENTILE = 0.95
TEEN_MODEL_THRESHOLD = 0.49
ADULT_MODEL_THRESHOLD = 0.1667  # F1 최적화 임계값
ADULT_DEFAULT_HDL = 53.50  # 평균 HDL-C 값
MODEL_PATH = "logit_model.pkl"  # 미리 학습해서 저장해 둔 모델 경로

# ==============================================================================
# 📝 모델 로드 및 준비 함수 (Model Persistence Logic)
# ==============================================================================


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


def load_saved_logit_model(path: str = MODEL_PATH):
    """
    미리 저장된 pkl 파일에서 로지스틱 회귀 모델을 불러옵니다.
    - pkl에 model만 저장했거나
    - {"model": model, "columns": [...]} 형태로 저장했을 때 둘 다 대응
    """
    if not os.path.exists(path):
        print(f"[WARN] 모델 파일을 찾을 수 없습니다: {path}")
        return None

    with open(path, "rb") as f:
        obj = pickle.load(f)

    # {"model": ..., "columns": ...} 형태로 저장한 경우
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]

    # 그냥 모델 객체만 저장한 경우
    return obj


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


def prepare_adult_model_data(df):
    # ✅ pkl 학습 때 사용한 컬럼 이름 기준
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
    X = sm.add_constant(X)
    return {"X": X, "y": y, "columns": X.columns.tolist()}


def compute_adult_model_results(dataframe: pd.DataFrame, model):
    if model is None:
        return None

    prep = prepare_adult_model_data(dataframe)
    if not prep:
        return None

    X, y = prep["X"], prep["y"]

    # 이름이 이미 model.params.index와 같으므로 그대로 사용
    y_prob = model.predict(X)
    y_pred = (y_prob >= ADULT_MODEL_THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, zero_division=0),
        "precision": precision_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred),
        "auc": roc_auc_score(y, y_prob),
        "threshold": ADULT_MODEL_THRESHOLD,
        "sample_size": len(y),
    }

    odds_ratios = np.exp(model.params)
    coef_df = pd.DataFrame(
        {"Coef": model.params, "OR": odds_ratios, "P-value": model.pvalues}
    )

    results = {
        "metrics": metrics,
            "odds_summary": coef_df.to_dict("index"),
        "model_params": model.params.to_dict(),
        "model_cols": prep["columns"],
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
    최종 간소화 모델 (상호작용항 없음)을 사용하여 당뇨병 위험을 예측합니다.
    """

    # 1. BMI 계산 및 분류
    bmi, obe_level = classify_adult_obesity(height_cm, weight_kg)

    # 2. 예측을 위한 DataFrame 생성
    new_data = pd.DataFrame({
        'const': [1],
        'AGE': [age],
        'SEX': [sex],
        'BMI': [bmi],
        'SBP': [sbp],
        'DBP': [dbp],
        'HDL': [hdl],
        'DM_FH': [dm_fh],
        'BREAKFAST': [br_fq]
    })

    # 3. 모델이 학습될 때 사용한 컬럼 기준으로 재인덱싱
    #    없는 컬럼은 0으로 채워서 모양 맞춰줌
    new_data = new_data.reindex(columns=model.params.index).fillna(0)

    # 4. 예측
    prediction_prob = model.predict(new_data)[0]

    return bmi, obe_level, prediction_prob, hdl


# ⚠️ 청소년 모델 학습 로직은 여기서 생략되었습니다.
def prepare_teen_model_data(dataframe: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
    return None


def compute_teen_model_results(dataframe: pd.DataFrame):
    return None


# ==============================================================================
# 🚀 메인 실행 및 Streamlit 로직
# ==============================================================================

# 데이터 로드
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

    if "DM_FH1" in df_new.columns and "DM_FH2" in df_new.columns:
        df_new["DM_FH"] = (
            (df_new["DM_FH1"] == 1) | (df_new["DM_FH2"] == 1)
        ).astype(int)

    if "BMI" in df_new.columns and "AGE" in df_new.columns:
        df_new["BMI_Age_Int"] = df_new["BMI"] * df_new["AGE"]

    return df_new


# 데이터 로드
df = load_data()
df_new = load_new_data()

# 전역 변수 설정 (청소년 모델)
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

# ⚡️ pkl에서 모델 로드 & 성능 계산
logit_model = load_saved_logit_model(MODEL_PATH)
adult_model_results_global = compute_adult_model_results(df_new, logit_model)
adult_model_summary_global = (
    adult_model_results_global.get("metrics") if adult_model_results_global else None
)
adult_model_coefs = (
    adult_model_results_global.get("model_params") if adult_model_results_global else None
)

teen_model_results_global = load_teen_model_results_from_file()
teen_model_summary_global = (
    teen_model_results_global.get("logistic") if teen_model_results_global else None
)

# ==============================================================================
# 📝 Streamlit 페이지 및 위젯
# ==============================================================================

# 페이지 설정
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
)  # 기본값 성인 데이터

# 선택된 데이터셋에 따라 사용할 데이터 결정
if dataset_choice == "청소년 데이터":
    current_df = df
    is_adult = False
else:
    current_df = df_new
    is_adult = True

# 사이드바 필터
st.sidebar.header("🔍 필터 옵션")

# 연도 필터
years = sorted(current_df["YEAR"].unique()) if "YEAR" in current_df.columns else []
selected_years = st.sidebar.multiselect("연도 선택", options=years, default=years)

# 성별 필터
sex_options = ["전체", "남성", "여성"]
selected_sex = st.sidebar.selectbox("성별 선택", sex_options)

# 연령 필터
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

# 데이터 필터링
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

# 청소년 데이터에만 도시 유형 필터 적용
if not is_adult and "CTYPE" in current_df.columns:
    city_types = ["전체"] + list(current_df["CTYPE"].unique())
    selected_city = st.sidebar.selectbox("도시 유형 선택", city_types)
    if selected_city != "전체":
        filtered_df = filtered_df[filtered_df["CTYPE"] == selected_city]

# 메인 타이틀
st.title("📊 건강 데이터 분석 대시보드")
st.markdown("---")

# 주요 지표 (KPI)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("총 데이터 수", f"{len(filtered_df):,}개")

with col2:
    avg_height = (
        filtered_df["HT"].dropna().mean()
        if "HT" in filtered_df.columns
        else np.nan
    )
    st.metric(
        "평균 키",
        f"{avg_height:.1f}cm" if not pd.isna(avg_height) else "N/A",
    )

with col3:
    avg_weight = (
        filtered_df["WT"].dropna().mean()
        if "WT" in filtered_df.columns
        else np.nan
    )
    st.metric(
        "평균 몸무게",
        f"{avg_weight:.1f}kg" if not pd.isna(avg_weight) else "N/A",
    )

with col4:
    avg_bmi = (
        filtered_df["BMI"].dropna().mean()
        if "BMI" in filtered_df.columns
        else np.nan
    )
    st.metric(
        "평균 BMI",
        f"{avg_bmi:.2f}" if not pd.isna(avg_bmi) else "N/A",
    )

with col5:
    total_records = len(df) if not is_adult else len(df_new)
    filtered_ratio = (len(filtered_df) / total_records * 100) if total_records > 0 else 0
    st.metric("필터링 비율", f"{filtered_ratio:.1f}%")

st.markdown("---")

# 탭 생성
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

# 탭 1: 개요
with tab1:
    st.header("데이터 개요")
    col1, col2 = st.columns(2)
    with col1:
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
    with col2:
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

    col3, col4 = st.columns(2)
    with col3:
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
    with col4:
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

# 탭 2: 인구통계
with tab2:
    st.header("인구통계 분석")
    col1, col2 = st.columns(2)
    with col1:
        ht_data = (
            filtered_df["HT"].dropna()
            if "HT" in filtered_df.columns
            else pd.Series()
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
    with col2:
        wt_data = (
            filtered_df["WT"].dropna()
            if "WT" in filtered_df.columns
            else pd.Series()
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

    col3, col4 = st.columns(2)
    with col3:
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
    with col4:
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

    col5, col6 = st.columns(2)
    with col5:
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
    with col6:
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

# 탭 3: 식습관 / 건강 지표
with tab3:
    if is_adult:
        st.header("🏥 건강 지표 및 식습관 분석")

        col1, col2 = st.columns(2)

        with col1:
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
                fig.add_vline(
                    x=126, line_dash="dash", line_color="red"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
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
                fig.add_vline(
                    x=5.7, line_dash="dash", line_color="green"
                )
                fig.add_vline(
                    x=6.5, line_dash="dash", line_color="red"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("📈 연도별 건강 지표 추이")
        col1, col2 = st.columns(2)

        with col1:
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

        with col2:
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
            obesity_counts = (
                filtered_df["OBESITY"].dropna().value_counts().sort_index()
            )
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
                    obesity_labels.get(x, str(x))
                    for x in obesity_counts.index
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

        # 연도별 당뇨 발병률 추이 (성별 구분)
        if "DIABETES" in filtered_df.columns:
            st.subheader("🩺 연도별 당뇨 발병률 추이 (성별 구분)")

            diabetes_data = filtered_df[
                ["YEAR", "SEX", "DIABETES"]
            ].dropna()
            if len(diabetes_data) > 0:

                def get_diabetes_rate(df_):
                    return (
                        (df_["DIABETES"] == 1.0).sum()
                        / len(df_)
                        * 100
                    )

                year_diabetes_all = (
                    diabetes_data.groupby("YEAR")
                    .apply(get_diabetes_rate)
                    .reset_index(name="당뇨발병률")
                )
                year_diabetes_all["성별"] = "전체"

                year_diabetes_male = (
                    diabetes_data[diabetes_data["SEX"] == 1.0]
                    .groupby("YEAR")
                    .apply(get_diabetes_rate)
                    .reset_index(name="당뇨발병률")
                )
                year_diabetes_male["성별"] = "남성"

                year_diabetes_female = (
                    diabetes_data[diabetes_data["SEX"] == 2.0]
                    .groupby("YEAR")
                    .apply(get_diabetes_rate)
                    .reset_index(name="당뇨발병률")
                )
                year_diabetes_female["성별"] = "여성"

                combined_diabetes_data = pd.concat(
                    [year_diabetes_all, year_diabetes_male, year_diabetes_female],
                    ignore_index=True,
                )

                if len(combined_diabetes_data) > 0:
                    fig = px.line(
                        combined_diabetes_data,
                        x="YEAR",
                        y="당뇨발병률",
                        color="성별",
                        markers=True,
                        labels={
                            "YEAR": "연도",
                            "당뇨발병률": "당뇨 발병률 (%)",
                        },
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

            # 성별 당뇨 발병률 비교
            st.subheader("📊 성별 당뇨 발병률 비교")

            diabetes_sex_data = filtered_df[["SEX", "DIABETES"]].dropna()
            if len(diabetes_sex_data) > 0:
                sex_diabetes_rates = {}

                total_diabetes = (diabetes_sex_data["DIABETES"] == 1.0).sum()
                sex_diabetes_rates["전체"] = (
                    total_diabetes / len(diabetes_sex_data) * 100
                )

                male_data = diabetes_sex_data[diabetes_sex_data["SEX"] == 1.0]
                if len(male_data) > 0:
                    sex_diabetes_rates["남성"] = (
                        (male_data["DIABETES"] == 1.0).sum()
                        / len(male_data)
                        * 100
                    )

                female_data = diabetes_sex_data[
                    diabetes_sex_data["SEX"] == 2.0
                ]
                if len(female_data) > 0:
                    sex_diabetes_rates["여성"] = (
                        (female_data["DIABETES"] == 1.0).sum()
                        / len(female_data)
                        * 100
                    )

                if len(sex_diabetes_rates) > 0:
                    fig = px.bar(
                        x=list(sex_diabetes_rates.keys()),
                        y=list(sex_diabetes_rates.values()),
                        labels={"x": "성별", "y": "당뇨 발병률 (%)"},
                        title="성별 당뇨 발병률 비교",
                        color=list(sex_diabetes_rates.keys()),
                        color_discrete_map={
                            "전체": "purple",
                            "남성": "#ff9999",
                            "여성": "#66b3ff",
                        },
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # 아침식사 빈도
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
                fig.update_traces(
                    textposition="inside", textinfo="percent+label"
                )
                st.plotly_chart(fig, use_container_width=True)

    else:  # 청소년 데이터
        st.info("청소년 데이터의 식습관 분석 및 트렌드 시각화 코드는 변경 없이 유지됩니다.")

# 탭 4: 상관관계 (성인 모델 변수 포함)
with tab4:
    st.header("상관관계 분석")

    if is_adult:
        # 성인 데이터 상관관계
        health_cols = ["BMI", "GLUCOSE", "HbA1c", "SBP", "DBP", "HDL", "DIABETES"]
        health_data = filtered_df.copy()
        health_cols = [col for col in health_cols if col in health_data.columns]
        health_data = health_data[health_cols].dropna()
        if len(health_data) > 0:
            health_corr = health_data.corr()
            fig = px.imshow(
                health_corr,
                labels=dict(x="변수", y="변수", color="상관계수"),
                x=health_cols,
                y=health_cols,
                color_continuous_scale="RdBu",
                aspect="auto",
                title="건강 지표 상관관계 히트맵",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("청소년 데이터 상관관계 분석 코드는 변경 없이 유지됩니다.")

# 탭 5: 데이터
with tab5:
    st.header("데이터 테이블")

    # 통계 요약
    st.subheader("📊 통계 요약")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**기본 정보**")
        if "YEAR" in filtered_df.columns and not filtered_df.empty:
            st.write(
                f"- 총 데이터 수: {len(filtered_df):,}개\n"
                f"- 연도 범위: {filtered_df['YEAR'].min()} ~ {filtered_df['YEAR'].max()}"
            )
        else:
            st.write(f"- 총 데이터 수: {len(filtered_df):,}개")
            st.write("- 연도: N/A")
        if "AGE" in filtered_df.columns and not filtered_df.empty:
            st.write(
                f"- 나이 범위: {filtered_df['AGE'].min()} ~ {filtered_df['AGE'].max()}세"
            )
        else:
            st.write("- 나이: N/A")

    with col2:
        st.write("**평균값**")
        st.write(
            f"- 평균 키: {filtered_df['HT'].mean():.2f}cm"
            if "HT" in filtered_df.columns and not filtered_df.empty
            else "- 평균 키: N/A"
        )
        st.write(
            f"- 평균 몸무게: {filtered_df['WT'].mean():.2f}kg"
            if "WT" in filtered_df.columns and not filtered_df.empty
            else "- 평균 몸무게: N/A"
        )
        st.write(
            f"- 평균 BMI: {filtered_df['BMI'].mean():.2f}"
            if "BMI" in filtered_df.columns and not filtered_df.empty
            else "- 평균 BMI: N/A"
        )

    with col3:
        st.write("**분포**")
        if "SEX" in filtered_df.columns and not filtered_df.empty:
            sex_counts = filtered_df["SEX"].value_counts()
            for sex_val, count in sex_counts.items():
                sex_name = "남성" if sex_val == 1.0 else "여성"
                st.write(f"- {sex_name}: {count:,}명")

    st.markdown("---")

    st.subheader("필터링된 데이터")
    st.dataframe(filtered_df, use_container_width=True)

# 탭 6: 모델 성능
with tab6:
    if is_adult:
        st.header("🤖 성인 당뇨병 예측 모델 성능")
        if adult_model_summary_global:
            metrics = adult_model_summary_global
            st.markdown(
                f"- **모델**: Logistic Regression (사전 학습 모델 사용)\n"
                f"- **라벨 기준**: DIABETES=1.0 (의사 진단 여부)\n"
                f"- **적용 임계값 (F1 최적화)**: **0.1667**"
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

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
            col2.metric("Recall (선별력)", f"{metrics['recall']*100:.1f}%")
            col3.metric("Precision", f"{metrics['precision']*100:.1f}%")

            col4, col5 = st.columns(2)
            col4.metric("F1-Score", f"{metrics['f1']*100:.1f}%")
            col5.metric("AUC-ROC", f"{metrics['auc']:.3f}")
            st.caption(f"학습 표본 수: {metrics['sample_size']:,}건")

            st.markdown("---")
            st.subheader("📊 주요 위험 요인 오즈비 (Odds Ratio)")
            st.info(
                "다른 모든 변수들을 통제한 상태에서, 해당 요인이 1 단위 증가할 때 당뇨 발병 오즈가 얼마나 변화하는지 나타냅니다."
            )

            if (
                adult_model_results_global
                and "odds_summary" in adult_model_results_global
            ):
                odds_df = (
                    pd.DataFrame(adult_model_results_global["odds_summary"])
                    .T.drop("const", errors="ignore")
                )
                odds_df = odds_df.rename(
                    columns={"OR": "오즈비(OR)", "P-value": "p-value"}
                ).round(4)

                feature_map = {
                    "AGE": "나이 (1세당)",
                    "SEX": "성별 (남성=1, 여성=2)",
                    "BMI": "BMI (1kg/m²당)",
                    "SBP": "수축기 혈압 (1mmHg당)",
                    "DBP": "이완기 혈압 (1mmHg당)",
                    "HDL": "HDL-C (1mg/dL당)",
                    "DM_FH": "가족력 (있음)",
                    "BREAKFAST": "아침식사 빈도 (1코드당)",
                }
                odds_df.index = [
                    feature_map.get(idx, idx) for idx in odds_df.index
                ]

                odds_df = odds_df.sort_values("오즈비(OR)", ascending=False)
                st.dataframe(odds_df, use_container_width=True)

        else:
            st.warning(
                "성인 모델 학습에 필요한 데이터(DIABETES, SBP, HDL 등)가 부족하거나, "
                "pkl 모델이 없어서 성능을 계산할 수 없습니다."
            )
    else:
        st.info("청소년 모델 성능 분석 코드는 변경 없이 유지됩니다.")

# 탭 7: 성인 예측
with tab7:
    st.header("🧑‍💻 성인 당뇨병 위험 예측기")
    st.markdown("---")

    if logit_model is None:
        st.warning(
            "모델 pkl 파일을 찾을 수 없어서 예측기를 활성화할 수 없습니다. "
            "MODEL_PATH 경로와 pkl 파일을 확인해주세요."
        )
    else:
        st.subheader("1. 신체 및 인구통계 정보 입력")

        col_age, col_sex, col_height, col_weight = st.columns(4)

        with col_age:
            age_input = st.slider("나이 (세)", min_value=19, max_value=100, value=45)

        with col_sex:
            sex_input_label = st.selectbox(
                "성별", options=["남성 (1.0)", "여성 (2.0)"], index=0
            )
            sex_input = 1.0 if "남성" in sex_input_label else 2.0

        with col_height:
            height_input = st.number_input(
                "키 (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1
            )

        with col_weight:
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

        col_sbp, col_dbp, col_hdl, col_fh = st.columns(4)

        with col_sbp:
            sbp_input = st.number_input(
                "수축기 혈압 (SBP)",
                min_value=80.0,
                max_value=200.0,
                value=120.0,
                step=1.0,
            )

        with col_dbp:
            dbp_input = st.number_input(
                "이완기 혈압 (DBP)",
                min_value=50.0,
                max_value=120.0,
                value=80.0,
                step=1.0,
            )

        with col_hdl:
            hdl_input_val = st.number_input(
                f"HDL-C (mg/dL) (생략 시 {ADULT_DEFAULT_HDL:.1f})",
                min_value=10.0,
                max_value=100.0,
                value=ADULT_DEFAULT_HDL,
                step=1.0,
            )

        with col_fh:
            dm_fh_input_label = st.selectbox(
                "당뇨병 가족력", options=["없음 (0)", "있음 (1)"]
            )
            dm_fh_input = 1 if "있음" in dm_fh_input_label else 0

        br_fq_options = get_br_fq_select_options()
        br_fq_label = st.selectbox(
            "아침 식사 빈도", options=list(br_fq_options.keys()), index=0
        )
        br_fq_input = br_fq_options[br_fq_label]

        st.markdown("---")

        if st.button("당뇨병 위험 확률 예측하기", type="primary"):
            try:
                used_hdl = (
                    hdl_input_val
                    if hdl_input_val != ADULT_DEFAULT_HDL
                    else ADULT_DEFAULT_HDL
                )

                (
                    bmi_result,
                    obe_level_result,
                    prob_result,
                    used_hdl,
                ) = predict_diabetes_risk_final(
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

                col_prob, col_risk = st.columns(2)

                with col_prob:
                    st.metric(
                        "예측된 당뇨병 발병 확률",
                        f"{prob_result * 100:.2f}%",
                    )

                risk_status = "❌ 위험군 아님"
                risk_color = "green"
                if prob_result >= ADULT_MODEL_THRESHOLD:
                    risk_status = "✅ 고위험군 (추가 검사 권고)"
                    risk_color = "red"

                with col_risk:
                    st.markdown(
                        f"**<p style='font-size: 24px; color:{risk_color};'>{risk_status}</p>**",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                st.markdown("#### 입력 데이터 요약")
                st.markdown(
                    f"""
                - **계산된 BMI:** {bmi_result:.2f} kg/m² ({bmi_label_map.get(obe_level_result, '미분류')})
                - **가족력:** {'있음' if dm_fh_input == 1 else '없음'}
                - **HDL-C:** {used_hdl:.2f} mg/dL ({'입력값 사용' if hdl_input_val != ADULT_DEFAULT_HDL else '평균값 사용'})
                - **아침 식사 빈도:** {get_br_fq_label(br_fq_input)}
                """
                )

            except Exception as e:
                st.error(
                    f"예측 중 오류가 발생했습니다. 입력값을 확인하거나 데이터/모델 파일을 확인해주세요: {e}"
                )

# 사이드바 하단 정보
st.sidebar.markdown("---")
ratio = (len(filtered_df) / len(current_df) * 100) if len(current_df) > 0 else 0
st.sidebar.info(
    f"""
    **현재 필터링된 데이터:**
    - {len(filtered_df):,}개 행
    - 전체 데이터의 {ratio:.1f}%
    """
)
