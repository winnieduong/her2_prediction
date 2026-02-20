import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

def fp_morgan(mol, radius, n_bits, use_chirality=False, use_features=False):
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits,
        useChirality=use_chirality,
        useFeatures=use_features
    )
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def fp_maccs(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="dataset/processed/her2_enzym_cleaned.csv")
    ap.add_argument("--bits", type=int, default=2048)
    ap.add_argument("--chirality", action="store_true")
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv).dropna(subset=["smiles", "pIC50"]).copy()

    X_list, keep_rows = [], []
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        ecfp4 = fp_morgan(mol, 2, args.bits, args.chirality, args.features)
        ecfp6 = fp_morgan(mol, 3, args.bits, args.chirality, args.features)
        maccs = fp_maccs(mol)
        X_list.append(np.concatenate([ecfp4, ecfp6, maccs], axis=0))
        keep_rows.append(i)

    df2 = df.loc[keep_rows].reset_index(drop=True)
    X = np.vstack(X_list).astype(np.int8)
    y = df2["pIC50"].values.astype(np.float32)

    np.save(f"{args.out_prefix}_X.npy", X)
    np.save(f"{args.out_prefix}_y.npy", y)
    df2.to_csv(f"{args.out_prefix}_meta.csv", index=False)

    print("Saved:", f"{args.out_prefix}_X.npy", X.shape)
    print("Saved:", f"{args.out_prefix}_y.npy", y.shape)

if __name__ == "__main__":
    main()