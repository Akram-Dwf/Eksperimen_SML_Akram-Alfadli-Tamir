<div align="center">

# 🏢 HR Employee Attrition — End-to-End MLOps System

### Membangun Sistem Machine Learning (MSML)

![Python](https://img.shields.io/badge/Python-3.12.7-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Hub-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Alerting-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

---

**Proyek MLOps end-to-end** yang mencakup seluruh lifecycle Machine Learning — mulai dari *data preprocessing*, *experiment tracking*, *CI/CD pipeline*, *model serving*, hingga *production monitoring & alerting* — menggunakan dataset **IBM HR Analytics Employee Attrition & Performance**.

**Akram Alfadli Tamir**

[📊 DagsHub MLflow](https://dagshub.com/Akram-Dwf/Eksperimen_SML_Akram-Alfadli-Tamir) · [🐳 Docker Hub](https://hub.docker.com/r/akramdwf/hr-attrition-model) · [⚙️ Workflow-CI Repo](https://github.com/Akram-Dwf/Workflow-CI)

</div>

---

## 📑 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Struktur Repositori](#-struktur-repositori)
- [Dataset](#-dataset)
- [Kriteria 1 — Eksperimen & Otomatisasi Preprocessing](#-kriteria-1--eksperimen--otomatisasi-preprocessing)
- [Kriteria 2 — Membangun Model & MLflow Tracking](#-kriteria-2--membangun-model--mlflow-tracking)
- [Kriteria 3 — CI/CD Pipeline & Docker](#-kriteria-3--cicd-pipeline--docker)
- [Kriteria 4 — Monitoring & Logging](#-kriteria-4--monitoring--logging)
- [Cara Menjalankan](#-cara-menjalankan)
- [Tech Stack](#-tech-stack)
- [Lisensi](#-lisensi)

---

## 🎯 Tentang Proyek

Proyek ini merupakan implementasi **sistem Machine Learning end-to-end** yang dirancang untuk memprediksi **attrition (resign) karyawan** di sebuah perusahaan. Sistem ini dibangun dengan pendekatan **MLOps best practices** yang mencakup:

| Fase | Deskripsi |
|---|---|
| 🔬 **Eksperimen** | EDA & preprocessing terstruktur di Jupyter Notebook |
| ⚙️ **Otomatisasi** | Pipeline preprocessing otomatis via Python script + GitHub Actions |
| 🧠 **Modelling** | Training & hyperparameter tuning dengan manual MLflow tracking |
| 📦 **Containerization** | Docker image model via `mlflow models build-docker` |
| 🚀 **Serving** | REST API menggunakan FastAPI dengan endpoint `/predict` |
| 📈 **Monitoring** | 10 metrik Prometheus + Grafana dashboard & 3 alerting rules |

> **Aturan Mutlak**: Seluruh logging MLflow dilakukan secara **manual** (`mlflow.log_param`, `mlflow.log_metric`, `mlflow.log_artifact`). **Tidak ada** penggunaan `mlflow.autolog()` di manapun dalam proyek ini.

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  WA_Fn-UseC_-HR-Employee-Attrition.csv (1470 rows, 35 cols)    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PIPELINE                         │
│  automate_Akram-Alfadli-Tamir.py                                │
│  ┌──────┐ ┌────────┐ ┌─────────┐ ┌────────┐ ┌───────┐          │
│  │ Drop │→│ Label  │→│ Ordinal │→│One-Hot │→│ Scale │          │
│  │ Cols │ │Encoding│ │Encoding │ │Encoding│ │(Std)  │          │
│  └──────┘ └────────┘ └─────────┘ └────────┘ └───┬───┘          │
│                                                  │               │
│                              train_data.csv ◄────┤               │
│                              test_data.csv  ◄────┘               │
└──────────────────────────┬──────────────────────────────────────┘
                           │  GitHub Actions: preprocessing.yml
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MODELLING & EXPERIMENT TRACKING                 │
│                                                                  │
│  modelling.py ──────────► RandomForest (Baseline)               │
│  modelling_tuning.py ──► RandomForest + GridSearchCV (36 comb.) │
│                                                                  │
│  ┌──────────────────────────────────────┐                       │
│  │         MLflow Manual Logging         │                       │
│  │  • log_params()  → Best hyperparams  │                       │
│  │  • log_metrics() → Acc/Prec/Rec/F1   │                       │
│  │  • log_artifact()→ 4 artifacts       │                       │
│  │  • log_model()   → sklearn model     │                       │
│  └──────────────────────────────────────┘                       │
│                          │                                       │
│                          ▼                                       │
│                  DagsHub Remote Tracking                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CI/CD & CONTAINERIZATION                       │
│                                                                  │
│  MLProject + conda.yaml ─► mlflow run .                         │
│                              │                                   │
│                              ▼                                   │
│                   mlflow models build-docker                     │
│                              │                                   │
│                              ▼                                   │
│                   Docker Hub: akramdwf/hr-attrition-model        │
│                                                                  │
│  GitHub Actions: ci.yml (auto build & push on push to main)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MODEL SERVING & MONITORING                        │
│                                                                  │
│  FastAPI (inference.py) ──► POST /predict                       │
│         │                   GET  /metrics                        │
│         │                   GET  /health                         │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Prometheus   │───►│   Grafana    │───►│  Alerting   │         │
│  │  (10 metrics) │    │ (Dashboard)  │    │  (3 rules)  │         │
│  └──────────────┘    └─────────────┘    └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Struktur Repositori

Proyek ini terdiri dari **3 repositori** yang saling terhubung:

### 1. Repositori Utama (`SMSML_Akram Alfadli Tamir`) ← *Anda di sini*

```
SMSML_Akram Alfadli Tamir/
│
├── 📁 preprocessing/                        # Kriteria 1: Preprocessing
│   ├── Template_Eksperimen_MSML.ipynb       # Notebook EDA & preprocessing
│   ├── automate_Akram-Alfadli-Tamir.py      # Skrip otomatisasi preprocessing
│   ├── train_data.csv                       # Data latih bersih (1176 sampel)
│   └── test_data.csv                        # Data uji bersih (294 sampel)
│
├── 📁 Membangun_model/                      # Kriteria 2: Modelling & Tracking
│   ├── modelling.py                         # Model baseline (tanpa tuning)
│   ├── modelling_tuning.py                  # Model + GridSearchCV + MLflow
│   ├── requirements.txt                     # Dependencies
│   ├── DagsHub.txt                          # Link DagsHub tracking
│   ├── 📁 artifacts/                        # Artefak evaluasi model
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── metric_info.json
│   │   └── estimator.html
│   └── 📁 mlruns/                           # MLflow local tracking data
│
├── 📁 .github/workflows/                   # Kriteria 1: CI Pipeline
│   └── preprocessing.yml                    # GitHub Actions preprocessing
│
├── WA_Fn-UseC_-HR-Employee-Attrition.csv    # Dataset mentah (raw)
├── .gitignore
└── README.md                                # Dokumentasi ini
```

### 2. Repositori CI/CD ([Workflow-CI](https://github.com/Akram-Dwf/Workflow-CI))

```
Workflow-CI/
├── 📁 MLProject/
│   ├── MLProject                            # Konfigurasi MLflow Project
│   ├── conda.yaml                           # Environment dependencies
│   ├── modelling.py                         # Training script
│   └── Tautan_Docker_Hub.txt                # Link Docker Hub image
├── 📁 .github/workflows/
│   └── ci.yml                               # CI/CD: build docker & push
└── README.md
```

### 3. Folder Monitoring & Logging

```
Monitoring dan Logging/
├── inference.py                             # FastAPI serving + /metrics
├── prometheus_exporter.py                   # 10 metrik Prometheus
├── prometheus.yml                           # Konfigurasi Prometheus
├── simulate_traffic.py                      # Simulasi traffic otomatis
├── 1.bukti_serving.png                      # Screenshot API serving
├── 📁 bukti monitoring Prometheus/          # 10 screenshot metrik Prometheus
├── 📁 bukti monitoring Grafana/             # Dashboard Grafana
└── 📁 bukti alerting Grafana/               # 3 alerting rules + notifikasi
```

---

## 📊 Dataset

| Atribut | Detail |
|---|---|
| **Nama** | IBM HR Analytics Employee Attrition & Performance |
| **Sumber** | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Jumlah Baris** | 1.470 karyawan |
| **Jumlah Kolom** | 35 fitur |
| **Target** | `Attrition` (Yes/No) — apakah karyawan resign |
| **Distribusi** | Imbalanced — 83.9% No, 16.1% Yes |
| **Missing Values** | Tidak ada |

### Fitur yang Digunakan

Setelah preprocessing, dataset memiliki **43 fitur** dan **1 target** (44 kolom total):

- **Fitur Numerik (24)**: Age, DailyRate, DistanceFromHome, Education, EnvironmentSatisfaction, HourlyRate, JobInvolvement, JobLevel, JobSatisfaction, MonthlyIncome, MonthlyRate, NumCompaniesWorked, PercentSalaryHike, PerformanceRating, RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, BusinessTravel
- **Fitur One-Hot Encoded (19)**: Hasil encoding dari Department, EducationField, Gender, JobRole, MaritalStatus, OverTime
- **Kolom yang Di-drop**: EmployeeCount, EmployeeNumber, Over18, StandardHours (konstan/non-prediktif)

---

## 🔬 Kriteria 1 — Eksperimen & Otomatisasi Preprocessing

### 📓 Notebook EDA (`Template_Eksperimen_MSML.ipynb`)

Eksplorasi data awal yang mencakup:
- Preview data (`head()`, `info()`, `describe()`)
- Pengecekan missing values dan duplikat
- Visualisasi distribusi target (countplot)

### ⚙️ Skrip Otomatisasi (`automate_Akram-Alfadli-Tamir.py`)

Pipeline preprocessing dibungkus dalam fungsi modular `preprocess_data()`:

```python
def preprocess_data(file_path: str) -> None:
    # 1. Muat data CSV
    # 2. Drop kolom tidak relevan (4 kolom)
    # 3. Label Encoding target Attrition (Yes=1, No=0)
    # 4. Ordinal Encoding BusinessTravel (0, 1, 2)
    # 5. One-Hot Encoding fitur nominal (drop_first=True)
    # 6. Train-Test Split (80:20, stratify, random_state=42)
    # 7. StandardScaler (fit pada train, transform pada test)
    # 8. Export ke train_data.csv & test_data.csv
```

> **Anti Data Leakage**: Scaler di-`fit_transform()` hanya pada training set, lalu di-`transform()` pada test set.

### 🔄 GitHub Actions (`preprocessing.yml`)

Pipeline CI yang otomatis berjalan setiap push ke `main`:

```yaml
Steps:
  1. Checkout repository
  2. Setup Python 3.12.7
  3. Install dependencies (pandas, numpy, scikit-learn)
  4. Run preprocessing script
  5. Upload train_data.csv & test_data.csv sebagai artifact
```

---

## 🧠 Kriteria 2 — Membangun Model & MLflow Tracking

### Model Baseline (`modelling.py`)

RandomForestClassifier dengan **parameter default** sebagai baseline pembanding.

### Model Tuning (`modelling_tuning.py`)

RandomForestClassifier dengan **GridSearchCV** untuk hyperparameter tuning:

| Parameter | Nilai yang Diuji |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 10, 20, None |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |
| **Total Kombinasi** | **36** |
| **CV Folds** | 5 |
| **Scoring** | F1-Score |

### 📋 MLflow Manual Logging

Seluruh pencatatan eksperimen dilakukan **secara manual** tanpa `autolog()`:

| Jenis Logging | Fungsi MLflow | Detail |
|---|---|---|
| **Parameters** | `mlflow.log_params()` | Best hyperparameters dari GridSearchCV |
| **Metrics** | `mlflow.log_metrics()` | Accuracy, Precision, Recall, F1-Score |
| **Artifacts** | `mlflow.log_artifact()` | 4 file artefak (lihat di bawah) |
| **Model** | `mlflow.sklearn.log_model()` | Model sklearn tersimpan |

### 📦 4 Artefak Wajib

| # | Artefak | Deskripsi |
|---|---|---|
| 1 | `confusion_matrix.png` | Visualisasi confusion matrix (heatmap) |
| 2 | `feature_importance.png` | Top 10 fitur terpenting |
| 3 | `metric_info.json` | Dictionary metrik dalam format JSON |
| 4 | `estimator.html` | Representasi HTML model dari sklearn |

### 🔗 DagsHub Integration

Eksperimen di-track secara remote ke DagsHub:
- **DagsHub**: [https://dagshub.com/Akram-Dwf/Eksperimen_SML_Akram-Alfadli-Tamir](https://dagshub.com/Akram-Dwf/Eksperimen_SML_Akram-Alfadli-Tamir)

---

## 🐳 Kriteria 3 — CI/CD Pipeline & Docker

### MLflow Project

Proyek dikemas sebagai **MLflow Project** dengan struktur standar:

```yaml
# MLProject
name: hr_attrition_ci
conda_env: conda.yaml
entry_points:
  main:
    command: "python train.py"
```

### GitHub Actions CI/CD (`ci.yml`)

Pipeline otomatis yang berjalan setiap push ke `main`:

```
1. ✅ Checkout code
2. ✅ Setup Python 3.12.7
3. ✅ Install dependencies (mlflow, scikit-learn)
4. ✅ Run MLflow Project (mlflow run . --env-manager=local)
5. ✅ Extract Run ID dari mlruns/
6. ✅ Upload mlruns/ sebagai GitHub Artifact
7. ✅ Build Docker image (mlflow models build-docker)
8. ✅ Login ke Docker Hub
9. ✅ Tag & Push image ke Docker Hub
```

### 🐳 Docker Hub

Model tersedia sebagai Docker image yang siap di-deploy:
- **Image**: [`akramdwf/hr-attrition-model:latest`](https://hub.docker.com/r/akramdwf/hr-attrition-model)

---

## 📈 Kriteria 4 — Monitoring & Logging

### 🚀 Model Serving (FastAPI)

API inference dibangun menggunakan **FastAPI** dengan endpoint:

| Method | Endpoint | Deskripsi |
|---|---|---|
| `POST` | `/predict` | Menerima fitur HR dan mengembalikan prediksi attrition |
| `GET` | `/metrics` | Mengekspos metrik Prometheus (scrape target) |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/docs` | Swagger UI (dokumentasi interaktif) |

### 📊 10 Metrik Prometheus (Advance)

| # | Nama Metrik | Tipe | Deskripsi |
|---|---|---|---|
| 1 | `http_requests_total` | Counter | Total HTTP request (by method/endpoint/status) |
| 2 | `http_request_duration_seconds` | Histogram | Durasi setiap HTTP request |
| 3 | `predictions_total` | Counter | Total prediksi yang berhasil |
| 4 | `predicted_attrition_positive_total` | Counter | Total prediksi resign (Yes) |
| 5 | `predicted_attrition_negative_total` | Counter | Total prediksi stay (No) |
| 6 | `model_errors_total` | Counter | Total error saat inference |
| 7 | `cpu_usage_percent` | Gauge | Penggunaan CPU (%) |
| 8 | `memory_usage_bytes` | Gauge | Penggunaan RAM (bytes) |
| 9 | `request_payload_size_bytes` | Histogram | Ukuran payload request |
| 10 | `active_requests` | Gauge | Request yang sedang diproses |

### 📊 Grafana Dashboard

Dashboard monitoring yang memvisualisasikan semua 10 metrik secara real-time, termasuk:
- Service uptime status
- CPU & Memory usage graphs
- HTTP request rate & active requests
- Prediction counts (positive vs negative)
- Model error tracking

### 🔔 Grafana Alerting (3 Rules)

| # | Alert Rule | Kondisi | Aksi |
|---|---|---|---|
| 1 | **CPU Usage High** | CPU > threshold | Notifikasi ke contact point |
| 2 | **HTTP Request Spike** | Request rate > threshold | Notifikasi ke contact point |
| 3 | **Prediction Volume** | Prediction count > threshold | Notifikasi ke contact point |

### 🔥 Traffic Simulator (`simulate_traffic.py`)

Skrip untuk menghasilkan traffic realistis ke API:
- **200 requests** dengan data payload acak
- **Burst mode** (15% chance): 5-10 request sekaligus tanpa jeda
- **Error injection** (8%): Payload invalid untuk memicu validation error
- Interval acak antara 0.05 - 0.5 detik

---

## 🚀 Cara Menjalankan

### Prerequisites

```bash
Python 3.12.7
pip (package manager)
```

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Akram-Dwf/Eksperimen_SML_Akram-Alfadli-Tamir.git
cd Eksperimen_SML_Akram-Alfadli-Tamir

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r Membangun_model/requirements.txt
pip install fastapi uvicorn prometheus-client psutil requests
```

### 2. Jalankan Preprocessing

```bash
python preprocessing/automate_Akram-Alfadli-Tamir.py
```

### 3. Jalankan Modelling & MLflow Tracking

```bash
# Baseline (tanpa tuning)
python Membangun_model/modelling.py

# Dengan hyperparameter tuning
python Membangun_model/modelling_tuning.py
```

### 4. Jalankan Model Serving (Monitoring)

```bash
# Terminal 1: Jalankan API server
cd "Monitoring dan Logging"
python inference.py
# Akses: http://localhost:8000/docs

# Terminal 2: Simulasi traffic
cd "Monitoring dan Logging"
python simulate_traffic.py

# Cek metrik
# http://localhost:8000/metrics
```

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|---|---|
| **Bahasa** | Python 3.12.7 |
| **ML Framework** | scikit-learn (RandomForestClassifier, GridSearchCV) |
| **Experiment Tracking** | MLflow 2.19.0 + DagsHub |
| **Data Processing** | pandas, NumPy |
| **Visualisasi** | Matplotlib, Seaborn |
| **API Framework** | FastAPI, Uvicorn |
| **Input Validation** | Pydantic |
| **Monitoring** | Prometheus (prometheus-client), psutil |
| **Dashboard & Alerting** | Grafana |
| **Containerization** | Docker, MLflow Docker Builder |
| **CI/CD** | GitHub Actions |
| **Version Control** | Git, GitHub |

---

## 📜 Lisensi

Proyek ini dibuat untuk keperluan tugas akhir **Membangun Sistem Machine Learning (MSML)** — Dicoding.

---

<div align="center">

**Dibuat dengan ❤️ oleh Akram Alfadli Tamir**

*End-to-End MLOps Pipeline — From Data to Deployment to Monitoring*

</div>
