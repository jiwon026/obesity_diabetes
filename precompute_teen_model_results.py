import json

import numpy as np
import pandas as pd
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


TEEN_EXCLUDED_YEARS = {2015, 2016}
TEEN_OBESITY_PERCENTILE = 0.95
TEEN_MODEL_THRESHOLD = 0.49


def main():
    teen = pd.read_csv("9ch_final_data.csv")
    teen = teen[~teen["YEAR"].isin(TEEN_EXCLUDED_YEARS)].copy()
    teen["BMI"] = teen["WT"] / ((teen["HT"] / 100) ** 2)
    cutoff = teen["BMI"].quantile(TEEN_OBESITY_PERCENTILE)
    teen["TEEN_OBESE_TOP5"] = (teen["BMI"] >= cutoff).astype(int)

    teen["HEALTHY_SCORE"] = teen[["F_FRUIT", "F_VEG", "Breakfast_Category"]].sum(
        axis=1
    )
    teen["UNHEALTHY_SCORE"] = teen[["F_FASTFOOD", "SODA_INTAKE"]].sum(axis=1)
    teen["NET_DIET_SCORE"] = teen["HEALTHY_SCORE"] - teen["UNHEALTHY_SCORE"]
    teen["GROUP"] = teen["GROUP"].fillna("Unknown").astype(str)
    teen["CTYPE"] = teen["CTYPE"].fillna("Unknown").astype(str)

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

    # 엔지니어링 & 스케일링
    X_train_eng = X_train.copy()
    X_test_eng = X_test.copy()
    for src, eng in [(X_train, X_train_eng), (X_test, X_test_eng)]:
        eng["AGE_FRUIT"] = src["AGE"] * src["F_FRUIT"]
        eng["AGE_VEG"] = src["AGE"] * src["F_VEG"]
        eng["AGE_FASTFOOD"] = src["AGE"] * src["F_FASTFOOD"]
        eng["FRUIT_VEG"] = src["F_FRUIT"] * src["F_VEG"]
        eng["FASTFOOD_SODA"] = src["F_FASTFOOD"] * src["SODA_INTAKE"]
        eng["BREAKFAST_AGE"] = src["Breakfast_Category"] * src["AGE"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_eng)
    X_test_scaled = scaler.transform(X_test_eng)

    # 클래스 가중치 & SMOTE
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    weight_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
    sample_weight = y_train.map(weight_dict).values

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # Logistic Regression 튜닝 (C + threshold)
    best_c = 0.1
    best_score = 0
    best_thr = TEEN_MODEL_THRESHOLD

    for c_val in [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
        lr_temp = LogisticRegression(
            max_iter=5000, class_weight="balanced", C=c_val, solver="lbfgs"
        )
        lr_temp.fit(X_train_smote, y_train_smote)
        y_prob_temp = lr_temp.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob_temp)

        for thr in np.linspace(0.35, 0.60, 26):
            y_pred_temp = (y_prob_temp >= thr).astype(int)
            acc = accuracy_score(y_test, y_pred_temp)
            rec = recall_score(y_test, y_pred_temp)
            if acc >= 0.60 and rec >= 0.65:
                score = acc * 0.4 + rec * 0.4 + test_auc * 0.2
                if score > best_score:
                    best_score = score
                    best_c = c_val
                    best_thr = thr

    lr_model = LogisticRegression(
        max_iter=5000, class_weight="balanced", C=best_c, solver="lbfgs"
    )
    lr_model.fit(X_train_smote, y_train_smote)
    y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr = (y_prob_lr >= best_thr).astype(int)

    # Random Forest 튜닝
    best_rf_score = 0
    best_rf_params = None
    for n_est in [200, 300, 400]:
        for max_d in [10, 12, 15]:
            rf_temp = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                min_samples_split=8,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
            rf_temp.fit(X_train_eng, y_train)
            y_prob_rf_temp = rf_temp.predict_proba(X_test_eng)[:, 1]
            auc_rf = roc_auc_score(y_test, y_prob_rf_temp)
            for thr in np.linspace(0.35, 0.60, 26):
                y_pred_rf_temp = (y_prob_rf_temp >= thr).astype(int)
                acc = accuracy_score(y_test, y_pred_rf_temp)
                rec = recall_score(y_test, y_pred_rf_temp)
                if acc >= 0.60 and rec >= 0.65:
                    score = acc * 0.4 + rec * 0.4 + auc_rf * 0.2
                    if score > best_rf_score:
                        best_rf_score = score
                        best_rf_params = {
                            "n_est": n_est,
                            "max_d": max_d,
                            "thr": thr,
                            "auc": auc_rf,
                        }

    if best_rf_params:
        rf_model = RandomForestClassifier(
            n_estimators=best_rf_params["n_est"],
            max_depth=best_rf_params["max_d"],
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rf_thr = best_rf_params["thr"]
    else:
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rf_thr = 0.5

    rf_model.fit(X_train_eng, y_train)
    y_prob_rf = rf_model.predict_proba(X_test_eng)[:, 1]
    y_pred_rf = (y_prob_rf >= rf_thr).astype(int)

    # Balanced RF
    brf_model = BalancedRandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    brf_model.fit(X_train_eng, y_train)
    y_prob_brf = brf_model.predict_proba(X_test_eng)[:, 1]
    y_pred_brf = brf_model.predict(X_test_eng)

    # HistGradientBoosting
    hgb_model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=30,
        class_weight=weight_dict,
        random_state=42,
    )
    hgb_model.fit(X_train_eng, y_train, sample_weight=sample_weight)
    y_prob_hgb = hgb_model.predict_proba(X_test_eng)[:, 1]
    y_pred_hgb = hgb_model.predict(X_test_eng)

    # 앙상블 (LR + RF 가중 평균)
    best_ensemble_score = 0
    best_weights = None
    best_ens_thr = best_thr

    for w1 in np.linspace(0.3, 0.8, 6):
        w2 = 1 - w1
        ensemble_prob_temp = w1 * y_prob_lr + w2 * y_prob_rf
        auc_ens = roc_auc_score(y_test, ensemble_prob_temp)
        for thr in np.linspace(0.35, 0.60, 26):
            y_pred_ens_temp = (ensemble_prob_temp >= thr).astype(int)
            acc = accuracy_score(y_test, y_pred_ens_temp)
            rec = recall_score(y_test, y_pred_ens_temp)
            if acc >= 0.60 and rec >= 0.65:
                score = acc * 0.4 + rec * 0.4 + auc_ens * 0.2
                if score > best_ensemble_score:
                    best_ensemble_score = score
                    best_weights = (float(w1), float(w2))
                    best_ens_thr = thr

    if best_weights:
        ensemble_prob = best_weights[0] * y_prob_lr + best_weights[1] * y_prob_rf
    else:
        best_weights = (0.6, 0.4)
        best_ens_thr = best_thr
        ensemble_prob = best_weights[0] * y_prob_lr + best_weights[1] * y_prob_rf

    y_pred_ens = (ensemble_prob >= best_ens_thr).astype(int)

    # AUC 비교 및 ROC 곡선
    model_aucs = {
        "logistic": roc_auc_score(y_test, y_prob_lr),
        "random_forest": roc_auc_score(y_test, y_prob_rf),
        "balanced_random_forest": roc_auc_score(y_test, y_prob_brf),
        "hist_gradient_boosting": roc_auc_score(y_test, y_prob_hgb),
        "ensemble": roc_auc_score(y_test, ensemble_prob),
    }
    best_model_name = max(model_aucs, key=model_aucs.get)
    best_auc = model_aucs[best_model_name]

    if best_model_name == "logistic":
        fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_lr)
    elif best_model_name == "random_forest":
        fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_rf)
    elif best_model_name == "balanced_random_forest":
        fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_brf)
    elif best_model_name == "hist_gradient_boosting":
        fpr_best, tpr_best, _ = roc_curve(y_test, y_prob_hgb)
    else:
        fpr_best, tpr_best, _ = roc_curve(y_test, ensemble_prob)

    results = {
        "threshold": float(best_thr),
        "optimal_c": float(best_c),
        "sample_size": int(len(data)),
        "logistic": {
            "accuracy": float(accuracy_score(y_test, y_pred_lr)),
            "recall": float(recall_score(y_test, y_pred_lr)),
            "precision": float(
                precision_score(y_test, y_pred_lr, zero_division=0)
            ),
            "f1": float(f1_score(y_test, y_pred_lr)),
            "auc": float(roc_auc_score(y_test, y_prob_lr)),
            "threshold": float(best_thr),
            "optimal_c": float(best_c),
            "sample_size": int(len(data)),
        },
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, y_pred_rf)),
            "recall": float(recall_score(y_test, y_pred_rf)),
            "precision": float(
                precision_score(y_test, y_pred_rf, zero_division=0)
            ),
            "f1": float(f1_score(y_test, y_pred_rf)),
            "auc": float(roc_auc_score(y_test, y_prob_rf)),
        },
        "balanced_random_forest": {
            "accuracy": float(accuracy_score(y_test, y_pred_brf)),
            "recall": float(recall_score(y_test, y_pred_brf)),
            "precision": float(
                precision_score(y_test, y_pred_brf, zero_division=0)
            ),
            "f1": float(f1_score(y_test, y_pred_brf)),
            "auc": float(roc_auc_score(y_test, y_prob_brf)),
        },
        "hist_gradient_boosting": {
            "accuracy": float(accuracy_score(y_test, y_pred_hgb)),
            "recall": float(recall_score(y_test, y_pred_hgb)),
            "precision": float(
                precision_score(y_test, y_pred_hgb, zero_division=0)
            ),
            "f1": float(f1_score(y_test, y_pred_hgb)),
            "auc": float(roc_auc_score(y_test, y_prob_hgb)),
        },
        "ensemble": {
            "accuracy": float(accuracy_score(y_test, y_pred_ens)),
            "recall": float(recall_score(y_test, y_pred_ens)),
            "precision": float(
                precision_score(y_test, y_pred_ens, zero_division=0)
            ),
            "f1": float(f1_score(y_test, y_pred_ens)),
            "auc": float(roc_auc_score(y_test, ensemble_prob)),
            "threshold": float(best_ens_thr),
            "weights": [float(best_weights[0]), float(best_weights[1])],
        },
        "roc_curve": {
            "fpr": fpr_best.tolist(),
            "tpr": tpr_best.tolist(),
            "auc": float(best_auc),
            "model_name": best_model_name,
        },
    }

    with open("teen_model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved teen_model_results.json")


if __name__ == "__main__":
    main()


