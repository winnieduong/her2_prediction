import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

INPUT_CSV = "dataset/processed/her2_enzym_cleaned.csv"

OUT_X     = "dataset/processed/X_emm_ecfp46.npy"
OUT_Y     = "dataset/processed/y_pic50_ecfp46.npy"
OUT_META  = "dataset/processed/featurize_meta_ecfp46.csv"

N_BITS = 2048

def fp_morgan(mol, radius, n_bits):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def fp_maccs(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)  # typically 167 bits
    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    ecfp4 = fp_morgan(mol, radius=2, n_bits=N_BITS)
    ecfp6 = fp_morgan(mol, radius=3, n_bits=N_BITS)
    maccs = fp_maccs(mol)

    # concat: 2048 + 2048 + 167 = 4263
    return np.concatenate([ecfp4, ecfp6, maccs], axis=0)

def main():
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["smiles", "pIC50"]).copy()

    X_list, keep_rows = [], []
    for i, row in df.iterrows():
        feat = smiles_to_features(row["smiles"])
        if feat is None:
            continue
        X_list.append(feat)
        keep_rows.append(i)

    df2 = df.loc[keep_rows].reset_index(drop=True)
    X = np.vstack(X_list).astype(np.int8)
    y = df2["pIC50"].values.astype(np.float32)

    np.save(OUT_X, X)
    np.save(OUT_Y, y)
    df2.to_csv(OUT_META, index=False)

    print("Saved:", OUT_X, X.shape)
    print("Saved:", OUT_Y, y.shape)
    print("Saved:", OUT_META, df2.shape)

if __name__ == "__main__":
    main()
