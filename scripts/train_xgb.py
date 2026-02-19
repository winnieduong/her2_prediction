import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import joblib

X_PATH = "dataset/processed/X_emm.npy"
Y_PATH = "dataset/processed/y_pic50.npy"

MODEL_OUT = "artifacts/xgb_pic50.pkl"
METRICS_OUT = "artifacts/metrics.json"
FS_OUT = "artifacts/feature_selection.json"

def feature_select_fit(X_train, zero_thresh=0.95, corr_thresh=0.7):
    # 1) remove mostly-zero columns
    zero_frac = (X_train == 0).mean(axis=0)
    keep1 = zero_frac < zero_thresh
    X1 = X_train[:, keep1]

    # 2) remove correlated columns (use correlation on float)
    # WARNING: correlation on binary can be noisy; still follow paper.
    corr = np.corrcoef(X1.astype(float), rowvar=False)
    n = corr.shape[0]
    to_drop = set()
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i+1, n):
            if abs(corr[i, j]) > corr_thresh:
                to_drop.add(j)
    keep2 = np.array([i not in to_drop for i in range(n)], dtype=bool)
    return keep1, keep2

def feature_select_apply(X, keep1, keep2):
    return X[:, keep1][:, keep2]

def main():
    import os
    os.makedirs("artifacts", exist_ok=True)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    keep1, keep2 = feature_select_fit(X_train, zero_thresh=0.95, corr_thresh=0.7)

    X_train_fs = feature_select_apply(X_train, keep1, keep2)
    X_test_fs  = feature_select_apply(X_test,  keep1, keep2)

    # Baseline params theo bài báo (Table 3)
    model = xgb.XGBRegressor(
        booster="gbtree",
        colsample_bytree=0.6027,
        gamma=0.8254,
        learning_rate=0.0421,
        max_depth=8,
        min_child_weight=1,
        n_estimators=245,
        subsample=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_fs, y_train)

    pred = model.predict(X_test_fs)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    joblib.dump({"model": model, "keep1": keep1, "keep2": keep2}, MODEL_OUT)

    with open(METRICS_OUT, "w") as f:
        json.dump({"r2": float(r2), "rmse": float(rmse),
                   "n_train": int(len(y_train)), "n_test": int(len(y_test)),
                   "n_feat_before": int(X.shape[1]),
                   "n_feat_after": int(X_train_fs.shape[1])}, f, indent=2)

    with open(FS_OUT, "w") as f:
        json.dump({"keep1_len": int(keep1.sum()), "keep2_len": int(keep2.sum())}, f, indent=2)

    print("R2:", r2, "RMSE:", rmse)

if __name__ == "__main__":
    main()
