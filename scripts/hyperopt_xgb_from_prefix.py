import argparse, json
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb

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

def cv_rmse(params, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []
    zero_thresh = float(params["zero_thresh"])
    corr_thresh = float(params["corr_thresh"])
    es_rounds = int(params["early_stopping_rounds"])

    for tr, te in kf.split(X):
        X_train, y_train = X[tr], y[tr]
        X_test, y_test = X[te], y[te]

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
        rmses.append(mean_squared_error(y_test, pred) ** 0.5)

    return float(np.mean(rmses))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="e.g. data/processed/ecfp46_b2048_chiral")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_evals", type=int, default=160)
    args = ap.parse_args()

    X = np.load(f"{args.prefix}_X.npy")
    y = np.load(f"{args.prefix}_y.npy")

    space = {
        "zero_thresh": hp.uniform("zero_thresh", 0.92, 0.99),
        "corr_thresh": hp.uniform("corr_thresh", 0.80, 0.99),

        "n_estimators": hp.quniform("n_estimators", 300, 1800, 1),
        "max_depth": hp.quniform("max_depth", 4, 12, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.006), np.log(0.1)),
        "subsample": hp.uniform("subsample", 0.55, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.45, 1.0),
        "min_child_weight": hp.loguniform("min_child_weight", np.log(0.5), np.log(40.0)),
        "gamma": hp.uniform("gamma", 0.0, 3.0),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-2), np.log(30.0)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-7), np.log(5.0)),
        "max_delta_step": hp.uniform("max_delta_step", 0.0, 5.0),
        "early_stopping_rounds": hp.quniform("early_stopping_rounds", 20, 100, 1),
    }

    trials = Trials()
    best = fmin(
        fn=lambda p: {"loss": cv_rmse(p, X, y), "status": STATUS_OK},
        space=space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    best["early_stopping_rounds"] = int(best["early_stopping_rounds"])

    best_rmse = min([t["result"]["loss"] for t in trials.trials])

    out = {"prefix": args.prefix, "best_params": best, "best_cv_rmse": float(best_rmse), "max_evals": args.max_evals}
    json.dump(out, open(args.out_json, "w"), indent=2)

    print("Saved:", args.out_json)
    print("Best CV RMSE:", best_rmse)
    print("Best params:", best)

if __name__ == "__main__":
    main()