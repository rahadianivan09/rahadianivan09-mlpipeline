#!/bin/bash
# ============================================================
# setup.sh — Staged TFX install untuk GitHub Codespace
# rahadianivan09 — Bank Deposit MLOps Pipeline
# ============================================================
# Jalankan sekali: bash setup.sh
# Estimasi waktu: 10-15 menit
# ============================================================

set -e  # Stop jika ada error

echo "============================================"
echo " Bank Deposit MLOps — Environment Setup"
echo "============================================"

# ── STEP 0: Set env vars penting ─────────────────────────
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TF_CPP_MIN_LOG_LEVEL=3

# Tambahkan ke .bashrc supaya persist
echo 'export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python' >> ~/.bashrc
echo 'export TF_CPP_MIN_LOG_LEVEL=3' >> ~/.bashrc

echo ""
echo "[STEP 1/5] Install TensorFlow + protobuf..."
pip install tensorflow==2.11.0 protobuf==3.19.6 -q
echo "✅ TensorFlow 2.11.0 + protobuf 3.19.6 done"

echo ""
echo "[STEP 2/5] Install TFX (no-deps untuk hindari konflik)..."
pip install tfx==1.12.0 --no-deps -q
echo "✅ TFX 1.12.0 done"

echo ""
echo "[STEP 3/5] Install Apache Beam..."
pip install apache-beam==2.43.0 -q
echo "✅ Apache Beam 2.43.0 done"

echo ""
echo "[STEP 4/5] Install dependencies lain..."
pip install \
    keras-tuner==1.1.3 \
    pandas==1.5.3 \
    numpy==1.23.5 \
    requests \
    ipykernel \
    dill \
    pydantic==1.10.13 \
    pyarrow==9.0.0 \
    -q
echo "✅ Dependencies lain done"

echo ""
echo "[STEP 5/5] Buat folder struktur proyek..."
mkdir -p data
mkdir -p modules
mkdir -p rahadianivan09-pipeline/metadata
mkdir -p rahadianivan09-pipeline/serving_model

# Copy module files ke folder modules/ jika belum ada
if [ -f "rahadianivan09_transform.py" ] && [ ! -f "modules/rahadianivan09_transform.py" ]; then
    cp rahadianivan09_transform.py modules/
    echo "  → rahadianivan09_transform.py disalin ke modules/"
fi
if [ -f "rahadianivan09_trainer.py" ] && [ ! -f "modules/rahadianivan09_trainer.py" ]; then
    cp rahadianivan09_trainer.py modules/
    echo "  → rahadianivan09_trainer.py disalin ke modules/"
fi

echo "✅ Folder struktur siap"

echo ""
echo "============================================"
echo " Verifikasi instalasi..."
echo "============================================"
python -c "
import tensorflow as tf
import tfx
import apache_beam
print(f'TensorFlow : {tf.__version__}')
print(f'TFX        : {tfx.__version__}')
print(f'Beam       : {apache_beam.__version__}')
"

echo ""
echo "============================================"
echo " ✅ Setup selesai!"
echo " Langkah selanjutnya:"
echo "   1. Upload bank.csv ke folder data/"
echo "   2. Jalankan rahadianivan09-pipeline.ipynb"
echo "============================================"
