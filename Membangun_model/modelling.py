"""
modelling.py
============
Skrip pelatihan model DASAR (Basic) menggunakan mlflow.autolog()
untuk dataset IBM HR Employee Attrition.

File ini sengaja menggunakan autolog() agar dapat dibandingkan
dengan modelling_tuning.py yang menggunakan manual logging secara penuh.

Author  : Akram Alfadli Tamir
"""

import os
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import mlflow

# import dagshub
# dagshub.init(repo_owner='Akram-Dwf', repo_name='Eksperimen_SML_Akram-Alfadli-Tamir', mlflow=True)

warnings.filterwarnings("ignore")


def load_data(train_path: str, test_path: str):
    """Muat data train dan test, pisahkan X dan y."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Attrition"])
    y_train = train_df["Attrition"]
    X_test = test_df.drop(columns=["Attrition"])
    y_test = test_df["Attrition"]

    return X_train, X_test, y_train, y_test


def main():
    """Fungsi utama pipeline modelling dasar dengan autolog."""
    print("=" * 60)
    print("  MODELLING DASAR (BASIC) — mlflow.autolog()")
    print("  Dataset: IBM HR Employee Attrition")
    print("  Algoritma: RandomForestClassifier (Default)")
    print("=" * 60)

    # Path data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "..", "preprocessing", "train_data.csv")
    test_path = os.path.join(base_dir, "..", "preprocessing", "test_data.csv")

    # Muat data
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)
    print(f"\n  Data latih : {X_train.shape[0]} sampel, {X_train.shape[1]} fitur")
    print(f"  Data uji   : {X_test.shape[0]} sampel, {X_test.shape[1]} fitur")

    # Aktifkan autolog — MLflow akan otomatis mencatat
    # parameter, metrik, dan model tanpa perintah manual
    mlflow.set_experiment("HR_Attrition_Experiment")
    mlflow.autolog()

    print("\n  [INFO] mlflow.autolog() diaktifkan.")
    print("  [INFO] Semua parameter, metrik, dan model akan dicatat otomatis.\n")

    # Training
    with mlflow.start_run(run_name="RandomForest_Autolog"):
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        print("  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"]))

    print("=" * 60)
    print("  SELESAI! Cek hasil di MLflow UI atau DagsHub.")
    print("=" * 60)


if __name__ == "__main__":
    main()
