import json
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

X_PATH = "dataset/processed/X_emm.npy"
Y_PATH = "dataset/processed/y_pic50.npy"

OUT_JSON = "reports/hyperopt_xgb_best.json"

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

def cv_rmse(params, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []

    zero_thresh = float(params["zero_thresh"])
    corr_thresh = float(params["corr_thresh"])
    es_rounds = int(params["early_stopping_rounds"])

    for tr, te in kf.split(X):
        X_train, y_train = X[tr], y[tr]
        X_test,  y_test  = X[te], y[te]

        keep1, keep2 = feature_select_fit(X_train, zero_thresh, corr_thresh)
        X_train_fs = feature_select_apply(X_train, keep1, keep2)
        X_test_fs  = feature_select_apply(X_test,  keep1, keep2)

        model = xgb.XGBRegressor(
            booster="gbtree",
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=float(params["min_child_weight"]),
            gamma=float(params["gamma"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            max_delta_step=float(params["max_delta_step"]),
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
        rmse = mean_squared_error(y_test, pred) ** 0.5
        rmses.append(rmse)

    return float(np.mean(rmses))

def main():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    space = {
        # NEW: tune feature-selection thresholds
        "zero_thresh": hp.uniform("zero_thresh", 0.90, 0.99),
        "corr_thresh": hp.uniform("corr_thresh", 0.60, 0.95),

        "n_estimators": hp.quniform("n_estimators", 200, 1400, 1),
        "max_depth": hp.quniform("max_depth", 3, 12, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.12)),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(0.5), np.log(30.0)),
        "gamma": hp.uniform("gamma", 0.0, 3.0),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-2), np.log(30.0)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-6), np.log(5.0)),
        "max_delta_step": hp.uniform("max_delta_step", 0.0, 5.0),
        "early_stopping_rounds": hp.quniform("early_stopping_rounds", 20, 80, 1),
    }

    trials = Trials()

    def objective(params):
        rmse = cv_rmse(params, X, y, n_splits=5)
        return {"loss": rmse, "status": STATUS_OK, "rmse": rmse}

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=200,  # tăng lên 200 để thresholds có cơ hội được khám phá đủ
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    best["early_stopping_rounds"] = int(best["early_stopping_rounds"])

    best_rmse = min([t["result"]["rmse"] for t in trials.trials])

    out = {
        "best_params": best,
        "best_cv_rmse": float(best_rmse),
        "max_evals": 200,
        "cv": "KFold(5, shuffle=True, random_state=42)",
        "notes": "Tunes feature-selection thresholds (zero_thresh, corr_thresh) + XGB params. Feature selection fitted per fold."
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", OUT_JSON)
    print("Best CV RMSE:", best_rmse)
    print("Best params:", best)

if __name__ == "__main__":
    main()
