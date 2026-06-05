import pandas as pd
import requests
import json
import os

# Ambil 1 baris data uji untuk contoh inferensi
base_dir = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(base_dir, "..", "preprocessing", "test_data.csv")
df = pd.read_csv(test_path)

# Ambil baris pertama, buang kolom target
X_test = df.drop(columns=["Attrition"]).iloc[[0]]

# Format payload JSON yang diterima oleh MLflow
payload = {
    "dataframe_records": X_test.to_dict(orient="records")
}

print("Mengirim request inferensi ke http://127.0.0.1:5001/invocations ...")

# Tembak API serve MLflow
response = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print(f"Status Code: {response.status_code}")
print(f"Hasil Prediksi: {response.json()}")
