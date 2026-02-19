import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import joblib
import os

META_CSV = "dataset/processed/featurize_meta.csv"   # file tạo từ scripts/02_featurize.py
X_PATH   = "dataset/processed/X_emm.npy"
Y_PATH   = "dataset/processed/y_pic50.npy"

MODEL_OUT   = "artifacts/xgb_scaffold.pkl"
METRICS_OUT = "reports/scaffold_split_metrics.json"

def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaff) if scaff is not None else ""

def scaffold_split(smiles_list, test_size=0.15, random_state=42):
    # Group indices by scaffold
    scaff2idx = {}
    for i, smi in enumerate(smiles_list):
        scaff = murcko_scaffold(smi)
        scaff2idx.setdefault(scaff, []).append(i)

    # Sort scaffold groups by size desc (common practice)
    groups = sorted(scaff2idx.values(), key=len, reverse=True)

    n_total = len(smiles_list)
    n_test_target = int(round(test_size * n_total))

    test_idx = []
    train_idx = []
    for g in groups:
        if len(test_idx) + len(g) <= n_test_target:
            test_idx.extend(g)
        else:
            train_idx.extend(g)

    # If test too small, move a few groups from train to test
    if len(test_idx) < n_test_target:
        # move smallest groups from train side
        # (simple fix; good enough for baseline)
        remaining = n_test_target - len(test_idx)
        # rebuild train groups list
        train_groups = [g for g in groups if g and g[0] in train_idx]
        train_groups = sorted(train_groups, key=len)  # smallest first
        moved = 0
        new_train_idx = []
        for g in train_groups:
            if moved < remaining:
                test_idx.extend(g)
                moved += len(g)
            else:
                new_train_idx.extend(g)
        train_idx = new_train_idx

    return np.array(train_idx), np.array(test_idx)

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
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    meta = pd.read_csv(META_CSV)
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    assert len(meta) == X.shape[0] == y.shape[0], "Meta/X/y length mismatch"

    train_idx, test_idx = scaffold_split(meta["smiles"].tolist(), test_size=0.15)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    keep1, keep2 = feature_select_fit(X_train, 0.95, 0.7)
    X_train_fs = feature_select_apply(X_train, keep1, keep2)
    X_test_fs  = feature_select_apply(X_test,  keep1, keep2)

    model = make_model()
    model.fit(X_train_fs, y_train)

    pred = model.predict(X_test_fs)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    joblib.dump({"model": model, "keep1": keep1, "keep2": keep2}, MODEL_OUT)

    out = {
        "split": "Murcko scaffold split",
        "test_size_target": 0.15,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_feat_before": int(X.shape[1]),
        "n_feat_after": int(X_train_fs.shape[1]),
        "r2": float(r2),
        "rmse": float(rmse),
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("Saved:", METRICS_OUT)
    print("R2:", r2, "RMSE:", rmse)

if __name__ == "__main__":
    main()
