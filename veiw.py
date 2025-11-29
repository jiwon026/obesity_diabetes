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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm 
import pickle  # 모델 저장/로드 라이브러리 추가

# ====== 전역 상수 ======
TEEN_EXCLUDED_YEARS = {2015, 2016}
TEEN_OBESITY_PERCENTILE = 0.95
TEEN_MODEL_THRESHOLD = 0.49
ADULT_MODEL_THRESHOLD = 0.1667  # F1 최적화 임계값
ADULT_DEFAULT_HDL = 53.50       # 평균 HDL-C 값
MODEL_PATH = 'logit_model.pkl'  # 🚨 pkl 모델 파일 경로

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
        1.0: '매일', 2.0: '주 5~6회', 3.0: '주 3~4회',
        4.0: '주 1~2회', 5.0: '월 1회', 6.0: '거의 안 먹음'
    }
    return mapping.get(br_fq_code, f'{int(br_fq_code)}회 (미분류)')

def get_br_fq_select_options():
    """Streamlit 선택 상자를 위한 옵션 생성"""
    return {
        '매일 (1.0)': 1.0, 
        '주 5~6회 (2.0
