"""
modelling_tuning.py
===================
Skrip pelatihan model, hyperparameter tuning, dan MLflow tracking MANUAL
untuk dataset IBM HR Employee Attrition.

ATURAN MUTLAK: TIDAK menggunakan mlflow.autolog() di manapun.
Semua logging dilakukan secara manual (log_param, log_metric, log_artifact).

Author  : Akram Alfadli Tamir
"""

import os
import json
import warnings

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interaktif agar bisa jalan di CI/server
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils import estimator_html_repr

import mlflow
from mlflow.sklearn import log_model as log_sklearn_model

# ======================================================================
# DagsHub Integration (uncomment setelah repo DagsHub siap)
# ======================================================================
import dagshub
dagshub.init(repo_owner='Akram-Dwf', repo_name='Eksperimen_SML_Akram-Alfadli-Tamir', mlflow=True)
warnings.filterwarnings("ignore")


def load_data(train_path: str, test_path: str):
    """Muat data train dan test, pisahkan X dan y."""
    print("=" * 60)
    print("MEMUAT DATA")
    print("=" * 60)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["Attrition"])
    y_train = train_df["Attrition"]
    X_test = test_df.drop(columns=["Attrition"])
    y_test = test_df["Attrition"]

    print(f"  Data latih : {X_train.shape[0]} sampel, {X_train.shape[1]} fitur")
    print(f"  Data uji   : {X_test.shape[0]} sampel, {X_test.shape[1]} fitur")
    print(f"  Distribusi target (train): {dict(y_train.value_counts())}")
    print(f"  Distribusi target (test) : {dict(y_test.value_counts())}")

    return X_train, X_test, y_train, y_test


def train_and_tune(X_train, y_train):
    """Latih RandomForestClassifier dengan GridSearchCV."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 60)

    # Definisi hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    total_combinations = 1
    for v in param_grid.values():
        total_combinations *= len(v)

    print(f"  Algoritma       : RandomForestClassifier")
    print(f"  Parameter grid  : {param_grid}")
    print(f"  Total kombinasi : {total_combinations}")
    print(f"  CV folds        : 5")
    print(f"  Scoring         : f1")
    print(f"\n  Memulai tuning... (mohon tunggu)\n")

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"\n  Best CV F1 Score : {best_score:.4f}")
    print(f"  Best Parameters  : {best_params}")

    return grid_search, best_model, best_params


def evaluate_model(model, X_test, y_test):
    """Evaluasi model pada data uji dan kembalikan dictionary metrik."""
    print("\n" + "=" * 60)
    print("EVALUASI MODEL PADA DATA UJI")
    print("=" * 60)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
    }

    print(f"  Accuracy  : {metrics['accuracy']}")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1-Score  : {metrics['f1_score']}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"]))

    return y_pred, metrics


def create_artifacts(model, grid_search, X_test, y_test, y_pred, metrics, feature_names, artifact_dir):
    """Buat 4 file artefak wajib dan simpan ke artifact_dir."""
    print("=" * 60)
    print("MEMBUAT ARTEFAK")
    print("=" * 60)

    os.makedirs(artifact_dir, exist_ok=True)

    # --- Artefak 1: confusion_matrix.png ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No (0)", "Yes (1)"],
        yticklabels=["No (0)", "Yes (1)"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  [1/4] confusion_matrix.png  -> disimpan")

    # --- Artefak 2: feature_importance.png ---
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_n = 10

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    feat_imp.head(top_n).plot(kind="barh", ax=ax2, color="steelblue")
    ax2.invert_yaxis()
    ax2.set_title(f"Top {top_n} Feature Importance", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Importance", fontsize=12)
    ax2.set_ylabel("Feature", fontsize=12)
    plt.tight_layout()

    fi_path = os.path.join(artifact_dir, "feature_importance.png")
    fig2.savefig(fi_path, dpi=150)
    plt.close(fig2)
    print(f"  [2/4] feature_importance.png -> disimpan")

    # --- Artefak 3: metric_info.json ---
    json_path = os.path.join(artifact_dir, "metric_info.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  [3/4] metric_info.json      -> disimpan")

    # --- Artefak 4: estimator.html ---
    html_path = os.path.join(artifact_dir, "estimator.html")
    html_content = estimator_html_repr(grid_search.best_estimator_)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  [4/4] estimator.html        -> disimpan")

    return cm_path, fi_path, json_path, html_path


def log_to_mlflow(best_model, grid_search, best_params, metrics, artifact_paths):
    """Log semua parameter, metrik, artefak, dan model ke MLflow secara MANUAL."""
    print("\n" + "=" * 60)
    print("MLFLOW MANUAL LOGGING")
    print("=" * 60)

    mlflow.set_experiment("HR_Attrition_Experiment")

    with mlflow.start_run(run_name="RandomForest_GridSearchCV") as run:
        run_id = run.info.run_id
        print(f"  Run ID: {run_id}")

        # --- Log Parameters (manual) ---
        print("\n  [Params] Logging hyperparameter terbaik...")
        mlflow.log_params(best_params)
        for k, v in best_params.items():
            print(f"    -> {k}: {v}")

        # --- Log Metrics (manual) ---
        print("\n  [Metrics] Logging metrik evaluasi...")
        mlflow.log_metrics(metrics)
        for k, v in metrics.items():
            print(f"    -> {k}: {v}")

        # --- Log Artifacts (manual, 4 file) ---
        print("\n  [Artifacts] Logging artefak...")
        for path in artifact_paths:
            mlflow.log_artifact(path)
            print(f"    -> {os.path.basename(path)}")

        # --- Log Model (manual) ---
        print("\n  [Model] Logging model sklearn...")
        log_sklearn_model(best_model, "random_forest_model")
        print("    -> random_forest_model (logged)")

    print(f"\n  MLflow Run selesai! Run ID: {run_id}")
    return run_id


def main():
    """Fungsi utama yang mengorkestrasi seluruh pipeline."""
    print("\n" + "=" * 60)
    print("  PIPELINE MODELLING & MLFLOW TRACKING")
    print("  Dataset: IBM HR Employee Attrition")
    print("  Algoritma: RandomForestClassifier + GridSearchCV")
    print("=" * 60 + "\n")

    # Tentukan path relatif terhadap lokasi skrip
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "train_data.csv")
    test_path = os.path.join(base_dir, "test_data.csv")
    artifact_dir = os.path.join(base_dir, "artifacts")

    # 1. Muat data
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    # 2. Training & Tuning
    grid_search, best_model, best_params = train_and_tune(X_train, y_train)

    # 3. Evaluasi
    y_pred, metrics = evaluate_model(best_model, X_test, y_test)

    # 4. Buat artefak
    artifact_paths = create_artifacts(
        best_model, grid_search,
        X_test, y_test, y_pred, metrics,
        feature_names=X_train.columns.tolist(),
        artifact_dir=artifact_dir,
    )

    # 5. Log ke MLflow
    run_id = log_to_mlflow(best_model, grid_search, best_params, metrics, artifact_paths)

    print("\n" + "=" * 60)
    print("PIPELINE SELESAI!")
    print(f"  Cek hasil di MLflow UI: mlflow ui")
    print(f"  Run ID: {run_id}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
