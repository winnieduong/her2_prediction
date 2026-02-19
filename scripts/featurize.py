import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

INPUT_CSV = "dataset/processed/her2_enzym_cleaned.csv"
OUT_X     = "dataset/processed/X_emm.npy"
OUT_Y     = "dataset/processed/y_pic50.npy"
OUT_META  = "dataset/processed/featurize_meta.csv"

N_BITS = 2048
RADIUS = 2  # ECFP4 => radius=2

def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ECFP4
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
    ecfp_arr = np.zeros((N_BITS,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(ecfp, ecfp_arr)

    # MACCS (167 bits; bit 0 thường unused → tuỳ repo, nhiều người bỏ bit 0)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros((maccs.GetNumBits(),), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(maccs, maccs_arr)

    # merge
    emm = np.concatenate([ecfp_arr, maccs_arr], axis=0)
    return emm

def main():
    df = pd.read_csv(INPUT_CSV)

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

    print("X:", X.shape, "y:", y.shape, "meta:", df2.shape)

if __name__ == "__main__":
    main()
