import json
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

X_PATH = "dataset/processed/X_emm_ecfp46.npy"
Y_PATH = "dataset/processed/y_pic50_ecfp46.npy"
BEST_JSON = "reports/hyperopt_xgb_best_ecfp46.json"

MODEL_OUT = "artifacts/xgb_final_ecfp46.pkl"
METRICS_OUT = "reports/final_randomsplit_metrics_ecfp46.json"

def feature_select_fit(X_train, zero_thresh=0.95, corr_thresh=0.7):
    zero_frac = (X_train == 0).mean(axis=0)
    keep1 = zero_frac < zero_thresh
    X1 = X_train[:, keep1]

    corr = np.corrcoef(X1.astype(float), rowvar=False)
    n = corr.shape[0]
    to_drop = set()
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if abs(corr[i, j]) > corr_thresh:
                to_drop.add(j)
    keep2 = np.array([i not in to_drop for i in range(n)], dtype=bool)
    return keep1, keep2

def feature_select_apply(X, keep1, keep2):
    return X[:, keep1][:, keep2]

def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    best = json.load(open(BEST_JSON, "r"))["best_params"]
    zero_thresh = float(best.get("zero_thresh", 0.95))
    corr_thresh = float(best.get("corr_thresh", 0.7))
    es_rounds = int(best.get("early_stopping_rounds", 50))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    keep1, keep2 = feature_select_fit(X_train, zero_thresh, corr_thresh)
    X_train_fs = feature_select_apply(X_train, keep1, keep2)
    X_test_fs  = feature_select_apply(X_test,  keep1, keep2)

    model = xgb.XGBRegressor(
        booster="gbtree",
        n_estimators=int(best["n_estimators"]),
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        min_child_weight=float(best["min_child_weight"]),
        gamma=float(best["gamma"]),
        reg_lambda=float(best["reg_lambda"]),
        reg_alpha=float(best["reg_alpha"]),
        max_delta_step=float(best["max_delta_step"]),
        early_stopping_rounds=es_rounds,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_fs, y_train,
        eval_set=[(X_test_fs, y_test)],
        verbose=False
    )

    pred = model.predict(X_test_fs)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    joblib.dump(
        {"model": model, "keep1": keep1, "keep2": keep2,
         "zero_thresh": zero_thresh, "corr_thresh": corr_thresh},
        MODEL_OUT
    )

    out = {
        "split": "random 85/15",
        "best_params_source": BEST_JSON,
        "zero_thresh": zero_thresh,
        "corr_thresh": corr_thresh,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_feat_before": int(X.shape[1]),
        "n_feat_after": int(X_train_fs.shape[1]),
        "r2": float(r2),
        "rmse": float(rmse),
    }

    json.dump(out, open(METRICS_OUT, "w"), indent=2)

    print("Saved:", MODEL_OUT)
    print("Saved:", METRICS_OUT)
    print("R2:", r2, "RMSE:", rmse, "n_feat:", X_train_fs.shape[1])

if __name__ == "__main__":
    main()
