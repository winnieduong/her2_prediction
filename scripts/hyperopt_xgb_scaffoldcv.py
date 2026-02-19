import json
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

META_CSV = "dataset/processed/featurize_meta.csv"
X_PATH   = "dataset/processed/X_emm.npy"
Y_PATH   = "dataset/processed/y_pic50.npy"

OUT_JSON = "reports/hyperopt_xgb_best_scaffoldcv.json"

def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaff) if scaff is not None else ""

def make_scaffold_folds(smiles_list, n_folds=5):
    # group indices by scaffold
    scaff2idx = {}
    for i, smi in enumerate(smiles_list):
        scaff = murcko_scaffold(smi)
        scaff2idx.setdefault(scaff, []).append(i)

    groups = sorted(scaff2idx.values(), key=len, reverse=True)

    # greedy bin packing into folds
    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0]*n_folds
    for g in groups:
        j = int(np.argmin(fold_sizes))
        folds[j].extend(g)
        fold_sizes[j] += len(g)

    return [np.array(f, dtype=int) for f in folds]

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

def scaffold_cv_rmse(params, X, y, folds):
    es_rounds = int(params["early_stopping_rounds"])
    rmses = []

    all_idx = np.arange(len(y))
    for k in range(len(folds)):
        te = folds[k]
        tr = np.setdiff1d(all_idx, te)

        X_train, y_train = X[tr], y[tr]
        X_test,  y_test  = X[te], y[te]

        keep1, keep2 = feature_select_fit(X_train, 0.95, 0.7)
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
    meta = pd.read_csv(META_CSV)
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    assert len(meta) == X.shape[0] == y.shape[0]

    folds = make_scaffold_folds(meta["smiles"].tolist(), n_folds=5)

    space = {
        "n_estimators": hp.quniform("n_estimators", 200, 1200, 1),
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
        rmse = scaffold_cv_rmse(params, X, y, folds)
        return {"loss": rmse, "status": STATUS_OK, "rmse": rmse}

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=120,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    best["early_stopping_rounds"] = int(best["early_stopping_rounds"])

    best_rmse = min([t["result"]["rmse"] for t in trials.trials])

    out = {
        "best_params": best,
        "best_scaffoldcv_rmse": float(best_rmse),
        "max_evals": 120,
        "cv": "5-fold scaffold CV (Murcko scaffolds, greedy balanced bins)",
        "notes": "Uses early stopping. Feature selection fitted per fold."
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", OUT_JSON)
    print("Best scaffold-CV RMSE:", best_rmse)
    print("Best params:", best)

if __name__ == "__main__":
    main()
