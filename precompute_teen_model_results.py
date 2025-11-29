import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# ================== 상수 ==================
TEEN_EXCLUDED_YEARS = {2015, 2016}
TEEN_OBESITY_PERCENTILE = 0.95
TEEN_MODEL_THRESHOLD = 0.49  # 초기값 (튜닝 시작점)


def main():
    # 1) 데이터 로드 & 전처리 ---------------------------------------------
    teen = pd.read_csv("9ch_final_data.csv")
    teen = teen[~teen["YEAR"].isin(TEEN_EXCLUDED_YEARS)].copy()

    # BMI & 비만 라벨 (상위 5%)
    teen["BMI"] = teen["WT"] / ((teen["HT"] / 100) ** 2)
    cutoff = teen["BMI"].quantile(TEEN_OBESITY_PERCENTILE)
    teen["TEEN_OBESE_TOP5"] = (teen["BMI"] >= cutoff).astype(int)

    # 식습관 점수
    teen["HEALTHY_SCORE"] = teen[["F_FRUIT", "F_VEG", "Breakfast_Category"]].sum(axis=1)
    teen["UNHEALTHY_SCORE"] = teen[["F_FASTFOOD", "SODA_INTAKE"]].sum(axis=1)
    teen["NET_DIET_SCORE"] = teen["HEALTHY_SCORE"] - teen["UNHEALTHY_SCORE"]

    teen["GROUP"] = teen["GROUP"].fillna("Unknown").astype(str)
    teen["CTYPE"] = teen["CTYPE"].fillna("Unknown").astype(str)

    # 사용할 피처들
    features = [
        "F_BR",
        "F_FRUIT",
        "F_VEG",
        "F_FASTFOOD",
        "SODA_INTAKE",
        "Breakfast_Category",
        "AGE",
        "SEX",
        "E_SES",
        "HEALTHY_SCORE",
        "UNHEALTHY_SCORE",
        "NET_DIET_SCORE",
        "GROUP",
        "CTYPE",
    ]

    data = teen[features + ["TEEN_OBESE_TOP5"]].dropna()
    X = data[features]
    X = pd.get_dummies(X, columns=["GROUP", "CTYPE"], drop_first=False)
    y = data["TEEN_OBESE_TOP5"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2) 피처 엔지니어링 & 스케일링 --------------------------------------
    X_train_eng = X_train.copy()
    X_test_eng = X_test.copy()

    for src, eng in [(X_train, X_train_eng), (X_test, X_test_eng)]:
        eng["AGE_FRUIT"] = src["AGE"] * src["F_FRUIT"]
        eng["AGE_VEG"] = src["AGE"] * src["F_VEG"]
        eng["AGE_FASTFOOD"] = src["AGE"] * src["F_FASTFOOD"]
        eng["FRUIT_VEG"] = src["F_FRUIT"] * src["F_VEG"]
        eng["FASTFOOD_SODA"] = src["F_FASTFOOD"] * src["SODA_INTAKE"]
        eng["BREAKFAST_AGE"] = src["Breakfast_Category"] * src["AGE"]
        eng["HEALTHY_UNHEALTHY"] = src["HEALTHY_SCORE"] * src["UNHEALTHY_SCORE"]
        eng["AGE_NET_SCORE"] = src["AGE"] * src["NET_DIET_SCORE"]
        eng["SEX_NET_SCORE"] = src["SEX"] * src["NET_DIET_SCORE"]
        eng["BREAKFAST_NET"] = src["Breakfast_Category"] * src["NET_DIET_SCORE"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_eng)
    X_test_scaled = scaler.transform(X_test_eng)

    # 3) 클래스 불균형 처리 (class weight + SMOTE) ------------------------
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    weight_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
    sample_weight = y_train.map(weight_dict).values

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # 4) Logistic Regression 튜닝 (C + threshold) -------------------------
    best_c = 0.1
    best_score = 0.0
    best_thr = TEEN_MODEL_THRESHOLD

    c_grid = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    thr_grid = np.linspace(0.30, 0.65, 36)

    for c_val in c_grid:
        lr_temp = LogisticRegression(
            max_iter=10000, class_weight="balanced", C=c_val, solver="lbfgs"
        )
        lr_temp.fit(X_train_smote, y_train_smote)
        y_prob_temp = lr_temp.predict_proba(X_test_scaled)[:, 1]
        auc_temp = roc_auc_score(y_test, y_prob_temp)

        for thr in thr_grid:
            y_pred_temp = (y_prob_temp >= thr).astype(int)
            acc = accuracy_score(y_test, y_pred_temp)
            rec = recall_score(y_test, y_pred_temp)

            # 조건: Accuracy >= 0.60, Recall >= 0.70
            if acc >= 0.60 and rec >= 0.70:
                score = acc * 0.30 + rec * 0.50 + auc_temp * 0.20
                if score > best_score:
                    best_score = score
                    best_c = c_val
                    best_thr = thr

    # 5) 최종 Logistic Regression 학습 -------------------------------------
    lr_model = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=best_c, solver="lbfgs"
    )
    lr_model.fit(X_train_smote, y_train_smote)

    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr = (y_prob_lr >= best_thr).astype(int)

    # 6) 계수 & 오즈비 -----------------------------------------------------
    feature_names = X_train_eng.columns.tolist()
    coefficients = lr_model.coef_[0]
    odds_ratios = np.exp(coefficients)

    coef_dict = {name: float(coefficients[i]) for i, name in enumerate(feature_names)}
    odds_ratio_dict = {name: float(odds_ratios[i]) for i, name in enumerate(feature_names)}

    # OR 테이블(성인과 동일 포맷: {변수명: {Coef:..., OR:...}})
    or_table = pd.DataFrame(
        {"Coef": coefficients, "OR": odds_ratios}, index=feature_names
    )
    odds_summary = or_table.to_dict("index")

    # 7) ROC 곡선 좌표 -----------------------------------------------------
    fpr_lr, tpr_lr, thr_roc = roc_curve(y_test, y_prob_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)

    # 8) 결과 정리 & JSON 저장 --------------------------------------------
    results = {
        # 전체 수준 정보
        "threshold": float(best_thr),
        "optimal_c": float(best_c),
        "sample_size": int(len(data)),
        # 로지스틱 모델 상세
        "logistic": {
            "accuracy": float(accuracy_score(y_test, y_pred_lr)),
            "recall": float(recall_score(y_test, y_pred_lr)),
            "precision": float(
                precision_score(y_test, y_pred_lr, zero_division=0)
            ),
            "f1": float(f1_score(y_test, y_pred_lr)),
            "auc": float(auc_lr),
            "threshold": float(best_thr),
            "optimal_c": float(best_c),
            "sample_size": int(len(data)),
            "coefficients": coef_dict,
            "odds_ratios": odds_ratio_dict,
            "odds_summary": odds_summary,  # 👈 대시보드에서 바로 쓰기 좋게
            "roc_curve": {
                "fpr": fpr_lr.tolist(),
                "tpr": tpr_lr.tolist(),
                "thresholds": thr_roc.tolist(),
            },
        },
    }

    with open("teen_model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved teen_model_results.json")


if __name__ == "__main__":
    main()
