# 🔥 Fire and Smoke Detection using MobileNetV2

## 📌 Project Overview
This project implements a computer vision pipeline to detect fire and smoke using a pretrained deep learning model (MobileNetV2). The system classifies images into two categories: fire and smoke.

---

## 📂 Dataset
- Dataset manually organized into two classes:
  - Fire
  - Smoke
- Data split:
  - Training: 70%
  - Validation: 15%
  - Testing: 15%

---

## ⚙️ Data Preprocessing
- Image resizing to 224x224
- Normalization (pixel values scaled to 0–1)
- Data augmentation:
  - Rotation
  - Horizontal flipping
  - Zooming

---

## 🤖 Model Implementation
- Pretrained MobileNetV2 used
- Transfer Learning applied
- Base model layers frozen
- Custom classification layer added

---

## 📊 Results
- Accuracy: 75%
- Strong performance on fire detection
- Weak performance on smoke detection

---

## 📉 Evaluation Metrics
- Classification Report
- Confusion Matrix

Example:   Confusion Matrix:
[[11 1]
[ 3 1]]


---

## 🧠 Analysis

### ✔️ Strengths:
- High accuracy for fire detection
- Efficient model using transfer learning

### ❌ Weaknesses:
- Low performance on smoke detection
- Confuses smoke with background

### ⚠️ Limitations:
- Small dataset
- Class imbalance
- Limited real-world diversity

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/train.py