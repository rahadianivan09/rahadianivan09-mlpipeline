# Bank Marketing Deposit Prediction — MLOps Pipeline

**Nama:** rahadianivan09  
**Dataset:** [Bank Marketing Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

---

## 📌 Deskripsi Dataset

Dataset Bank Marketing berasal dari kampanye pemasaran telepon langsung sebuah bank di Portugal. Setiap baris merepresentasikan satu nasabah yang dihubungi dalam kampanye tersebut.

| Atribut | Jumlah |
|---------|--------|
| Total sampel | ±11.162 baris |
| Total fitur | 16 fitur input + 1 label |
| Label positif (deposit = yes) | ±12% |
| Label negatif (deposit = no) | ±88% |

**Fitur input:**

| Fitur | Tipe | Deskripsi |
|-------|------|-----------|
| age | Numerik | Usia nasabah |
| job | Kategorikal | Jenis pekerjaan |
| marital | Kategorikal | Status pernikahan |
| education | Kategorikal | Tingkat pendidikan |
| default | Kategorikal | Apakah memiliki kredit macet |
| balance | Numerik | Saldo rata-rata tahunan (EUR) |
| housing | Kategorikal | Apakah memiliki KPR |
| loan | Kategorikal | Apakah memiliki pinjaman pribadi |
| contact | Kategorikal | Tipe komunikasi |
| day | Numerik | Hari terakhir dihubungi |
| month | Kategorikal | Bulan terakhir dihubungi |
| duration | Numerik | Durasi panggilan terakhir (detik) |
| campaign | Numerik | Jumlah kontak selama kampanye ini |
| pdays | Numerik | Hari sejak dihubungi kampanye sebelumnya (-1 = belum pernah) |
| previous | Numerik | Jumlah kontak sebelum kampanye ini |
| poutcome | Kategorikal | Hasil kampanye sebelumnya |

**Label:** `deposit` — apakah nasabah berlangganan deposito berjangka? (`yes` = 1, `no` = 0)

---

## 🎯 Persoalan yang Ingin Diselesaikan

Bank menghabiskan sumber daya besar untuk menghubungi nasabah satu per satu. Tanpa prediksi yang tepat, tim marketing menghubungi banyak nasabah yang tidak berminat, sehingga:

- **Biaya operasional tinggi** (waktu agen, biaya telekomunikasi)
- **Customer experience buruk** (nasabah yang tidak tertarik merasa terganggu)
- **Conversion rate rendah**

**Solusi yang dibangun:** Model machine learning yang memprediksi apakah seorang nasabah akan berlangganan deposito berjangka, sehingga tim marketing dapat memprioritaskan nasabah dengan kemungkinan konversi tertinggi.

---

## 🧠 Solusi Machine Learning

### Target yang Ingin Dicapai

| Metrik | Target Minimum |
|--------|----------------|
| Binary Accuracy | ≥ 75% |
| AUC-ROC | ≥ 0.70 |

### Arsitektur Model

Model menggunakan **Deep Neural Network (DNN) dengan Embedding** untuk fitur kategorikal:

```
Input Numerik (4 fitur) ──────────────────────────────────┐
                                                           ├─► Concatenate
Input Kategorikal (8 fitur) → Embedding(50, 8) → Flatten ─┘
                                    │
                              Dense(128, relu) + L2
                              BatchNormalization
                              Dropout(p)
                                    │
                              Dense(64, relu) + L2
                              BatchNormalization
                              Dropout(p)
                                    │
                              Dense(32, relu) + L2
                              Dropout(p/2)
                                    │
                              Dense(1, sigmoid)
```

**Hyperparameter yang di-tune via Keras Tuner (RandomSearch):**

| Parameter | Range |
|-----------|-------|
| units_1 | 64, 128, 192, 256 |
| units_2 | 32, 64, 96, 128 |
| units_3 | 16, 32, 48, 64 |
| dropout | 0.2, 0.3, 0.4, 0.5 |
| learning_rate | 0.01, 0.001, 0.0001 |

### Penanganan Class Imbalance

Dataset sangat imbalanced (~88% no, ~12% yes). Strategi yang digunakan:

- **Class weight:** `{0: 1.0, 1: 7.0}` — model dipaksa memperhatikan kelas minoritas
- **Objective tuner:** `val_auc` (lebih robust dari accuracy untuk data imbalanced)
- **Callbacks:** EarlyStopping + ReduceLROnPlateau

---

## ⚙️ Metode Pengolahan Data (Transform)

Preprocessing dilakukan menggunakan **TensorFlow Transform (tft)** di dalam komponen `Transform`:

| Tipe Fitur | Fitur | Transformasi |
|-----------|-------|--------------|
| Numerik | age, balance, day, campaign | Z-score normalization (`tft.scale_to_z_score`) |
| Kategorikal | job, marital, education, default, housing, loan, contact, month | Vocabulary encoding (`tft.compute_and_apply_vocabulary`) |
| Label | deposit | Cast ke int64 |

Fitur `duration`, `pdays`, `previous`, dan `poutcome` tidak digunakan dalam model untuk menghindari **data leakage** — fitur `duration` misalnya hanya diketahui setelah panggilan selesai.

---

## 🔁 Komponen Pipeline TFX

| Komponen | Fungsi |
|----------|--------|
| **ExampleGen** | Membaca `bank.csv` dan membaginya menjadi train/eval split |
| **StatisticsGen** | Menghitung statistik deskriptif dari data |
| **SchemaGen** | Membuat skema otomatis berdasarkan statistik |
| **ExampleValidator** | Mendeteksi anomali data (missing values, type mismatch) |
| **Transform** | Preprocessing dan feature engineering |
| **Tuner** | Hyperparameter tuning otomatis dengan Keras Tuner |
| **Trainer** | Melatih model DNN dengan hyperparameter terbaik |
| **Resolver** | Mencari model terbaru yang telah di-bless sebagai baseline |
| **Evaluator** | Evaluasi model vs threshold — model di-bless jika lolos |
| **Pusher** | Menyimpan model ke serving directory jika di-bless |

---

## 📊 Performa Model

Model dievaluasi menggunakan `tensorflow_model_analysis` (TFMA) dengan threshold:

| Metrik | Threshold | Hasil |
|--------|-----------|-------|
| Binary Accuracy | ≥ 0.75 | ✅ PASSED |
| AUC-ROC | ≥ 0.70 | ✅ PASSED |

Model dinyatakan **BLESSED** dan dipush ke serving directory.

---

## 🚀 Deployment dengan TensorFlow Serving

Model di-serve menggunakan Docker + TensorFlow Serving:

```bash
# Build image
docker build -t bank-deposit-serving .

# Run container
docker run -p 8501:8501 bank-deposit-serving
```

Endpoint prediksi: `http://localhost:8501/v1/models/bank-deposit-model:predict`

Format request (TFX serving signature):
```json
{
  "signature_name": "serving_default",
  "instances": [{"examples": {"b64": "<serialized_tf_example_base64>"}}]
}
```

Pengujian prediksi tersedia di notebook `rahadianivan09-testing.ipynb`.

---

## 📁 Struktur Proyek

```
.
├── data/
│   └── bank.csv
├── modules/
│   ├── rahadianivan09_transform.py
│   └── rahadianivan09_trainer.py
├── rahadianivan09-pipeline/          ← output pipeline (auto-generated)
│   ├── metadata/
│   └── serving_model/
├── rahadianivan09-pipeline.ipynb     ← notebook pipeline utama
├── rahadianivan09-testing.ipynb      ← notebook uji prediksi
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🛠️ Cara Menjalankan

```bash
# 1. Install dependencies
pip install tensorflow==2.11.0 protobuf==3.19.6
pip install tfx==1.12.0 --no-deps
pip install apache-beam==2.43.0
pip install keras-tuner pandas ipykernel

# 2. Jalankan pipeline
jupyter nbconvert --to notebook --execute rahadianivan09-pipeline.ipynb

# 3. (Opsional) Deploy dengan Docker
docker build -t bank-deposit-serving .
docker run -p 8501:8501 bank-deposit-serving
```
