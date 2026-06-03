"""
automate_Akram-Alfadli-Tamir.py
===============================
Skrip otomatisasi preprocessing dataset IBM HR Employee Attrition.
Mengubah data mentah (raw CSV) menjadi data bersih (train & test CSV)
yang siap digunakan untuk pelatihan model Machine Learning.

Author  : Akram Alfadli Tamir
Dataset : WA_Fn-UseC_-HR-Employee-Attrition.csv
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(file_path: str) -> None:
    """
    Fungsi utama yang merangkum seluruh pipeline preprocessing:
      1. Muat data CSV
      2. Drop kolom tidak relevan
      3. Label Encoding target (Attrition)
      4. Ordinal Encoding (BusinessTravel)
      5. One-Hot Encoding fitur nominal
      6. Train-Test Split (80:20, stratified)
      7. Standard Scaling fitur numerik
      8. Export ke train_data.csv & test_data.csv

    Parameters
    ----------
    file_path : str
        Path menuju file CSV dataset mentah.
    """

    # ------------------------------------------------------------------
    # 1. Muat Data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("MEMULAI PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"\n[1/8] Memuat dataset dari: {file_path}")

    df = pd.read_csv(file_path)
    print(f"      -> Dimensi awal: {df.shape[0]} baris, {df.shape[1]} kolom")

    # ------------------------------------------------------------------
    # 2. Drop Kolom Tidak Relevan
    # ------------------------------------------------------------------
    drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    print(f"\n[2/8] Menghapus kolom tidak relevan: {drop_cols}")

    df = df.drop(columns=drop_cols)
    print(f"      -> Sisa kolom: {df.shape[1]}")

    # ------------------------------------------------------------------
    # 3. Label Encoding Target (Attrition)
    # ------------------------------------------------------------------
    print("\n[3/8] Encoding target 'Attrition' (Yes=1, No=0)")

    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    print(f"      -> Distribusi target:\n{df['Attrition'].value_counts().to_string()}")

    # ------------------------------------------------------------------
    # 4. Ordinal Encoding (BusinessTravel)
    # ------------------------------------------------------------------
    travel_map = {
        "Non-Travel": 0,
        "Travel_Rarely": 1,
        "Travel_Frequently": 2,
    }
    print(f"\n[4/8] Ordinal encoding 'BusinessTravel': {travel_map}")

    df["BusinessTravel"] = df["BusinessTravel"].map(travel_map)
    print(f"      -> Distribusi:\n{df['BusinessTravel'].value_counts().sort_index().to_string()}")

    # ------------------------------------------------------------------
    # 5. One-Hot Encoding Fitur Nominal
    # ------------------------------------------------------------------
    nominal_cols = [
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime",
    ]
    print(f"\n[5/8] One-Hot Encoding (drop_first=True): {nominal_cols}")

    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
    print(f"      -> Jumlah kolom setelah encoding: {df.shape[1]}")

    # ------------------------------------------------------------------
    # 6. Pemisahan Fitur & Target + Train-Test Split
    # ------------------------------------------------------------------
    print("\n[6/8] Memisahkan fitur (X) dan target (y)...")

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Buat copy independen untuk menghindari SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()

    print(f"      -> Data latih : {X_train.shape[0]} sampel")
    print(f"      -> Data uji   : {X_test.shape[0]} sampel")

    # ------------------------------------------------------------------
    # 7. Standard Scaling (hanya fitur numerik asli)
    # ------------------------------------------------------------------
    print("\n[7/8] Menerapkan StandardScaler pada fitur numerik...")

    # Identifikasi kolom one-hot (dari prefix nominal)
    ohe_prefixes = [
        "Department_", "EducationField_", "Gender_",
        "JobRole_", "MaritalStatus_", "OverTime_",
    ]
    ohe_cols = [
        col for col in X_train.columns
        if any(col.startswith(prefix) for prefix in ohe_prefixes)
    ]
    numerical_cols = [col for col in X_train.columns if col not in ohe_cols]

    print(f"      -> Fitur numerik (di-scale)     : {len(numerical_cols)}")
    print(f"      -> Fitur one-hot (TIDAK di-scale): {len(ohe_cols)}")

    scaler = StandardScaler()

    # Cast ke float64 terlebih dahulu agar tidak ada FutureWarning dtype
    X_train[numerical_cols] = X_train[numerical_cols].astype("float64")
    X_test[numerical_cols] = X_test[numerical_cols].astype("float64")

    X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])

    print("      -> Scaling selesai (fit pada train, transform pada test)")

    # ------------------------------------------------------------------
    # 8. Export Data Bersih ke CSV
    # ------------------------------------------------------------------
    print("\n[8/8] Menyimpan data bersih ke CSV...")

    # Tentukan direktori output = direktori tempat skrip ini berada
    output_dir = os.path.dirname(os.path.abspath(__file__))

    train_df = X_train.copy()
    train_df["Attrition"] = y_train.values

    test_df = X_test.copy()
    test_df["Attrition"] = y_test.values

    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    train_size = os.path.getsize(train_path) / 1024
    test_size = os.path.getsize(test_path) / 1024

    print(f"      -> train_data.csv ({train_df.shape[0]} baris, {train_df.shape[1]} kolom, {train_size:.1f} KB)")
    print(f"      -> test_data.csv  ({test_df.shape[0]} baris, {test_df.shape[1]} kolom, {test_size:.1f} KB)")

    # ------------------------------------------------------------------
    # Selesai
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI!")
    print("=" * 60)


if __name__ == "__main__":
    # Path ke dataset (mundur satu direktori ke root tempat raw data berada)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    preprocess_data(dataset_path)
