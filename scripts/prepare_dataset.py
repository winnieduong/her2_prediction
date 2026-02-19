import pandas as pd
import numpy as np

INPUT_XLSX = "dataset/raw/HER2_enzym_cleaned.xlsx"
OUT_CSV    = "dataset/processed/her2_enzym_cleaned.csv"

def ic50nm_to_pic50(ic50_nm: float) -> float:
    # pIC50 = -log10(IC50 in M) = -log10(IC50_nM * 1e-9) = 9 - log10(IC50_nM)
    return 9.0 - np.log10(ic50_nm)

def main():
    df = pd.read_excel(INPUT_XLSX)

    # 1) lấy SMILES + IC50_nM
    df = df.rename(columns={"Canonical SMILES": "smiles",
                            "Standardized IC50 (nM)": "ic50_nm"})
    df = df[["ID", "InChIKey", "smiles", "ic50_nm"]].copy()

    # 2) drop missing
    df = df.dropna(subset=["smiles", "ic50_nm"])

    # 3) loại IC50 <= 0
    df = df[df["ic50_nm"] > 0].copy()

    # 4) tính pIC50
    df["pIC50"] = ic50nm_to_pic50(df["ic50_nm"].astype(float))

    # 5) xử lý duplicates: theo InChIKey nếu có, nếu không dùng smiles
    key = "InChIKey" if df["InChIKey"].notna().any() else "smiles"
    df = df.groupby(key, as_index=False).agg({
        "smiles": "first",
        "ic50_nm": "mean",
        "pIC50": "mean",
        "ID": "first"
    })

    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV, "shape=", df.shape)

if __name__ == "__main__":
    main()
