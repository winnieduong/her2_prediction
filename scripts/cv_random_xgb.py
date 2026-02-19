import json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

X_PATH = "dataset/processed/X_emm.npy"
Y_PATH = "dataset/processed/y_pic50.npy"

OUT_JSON = "reports/cv_random_xgb.json"

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

def make_model():
    # Paper-like baseline params (Table 3)
    return xgb.XGBRegressor(
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

def main():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []
    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        X_train, y_train = X[tr], y[tr]
        X_test,  y_test  = X[te], y[te]

        keep1, keep2 = feature_select_fit(X_train, zero_thresh=0.95, corr_thresh=0.7)
        X_train_fs = feature_select_apply(X_train, keep1, keep2)
        X_test_fs  = feature_select_apply(X_test,  keep1, keep2)

        model = make_model()
        model.fit(X_train_fs, y_train)

        pred = model.predict(X_test_fs)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5

        fold_metrics.append({
            "fold": fold,
            "r2": float(r2),
            "rmse": float(rmse),
            "n_feat_after": int(X_train_fs.shape[1]),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        })
        print(f"[Fold {fold}] R2={r2:.4f} RMSE={rmse:.4f} n_feat={X_train_fs.shape[1]}")

    r2s = [m["r2"] for m in fold_metrics]
    rmses = [m["rmse"] for m in fold_metrics]
    summary = {
        "cv": "KFold(5, shuffle=True, random_state=42)",
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "folds": fold_metrics
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:", OUT_JSON)

if __name__ == "__main__":
    main()
