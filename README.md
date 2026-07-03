# AICTE-Internship-potato-project

A Deep Learning based web application that detects potato leaf diseases from uploaded images using TensorFlow and Streamlit.

## 🚀 Live Demo

[(https://potatodisease.streamlit.app/)](https://potatodisease.streamlit.app/)

---

## 📌 Features

- 🌿 Detect Potato Leaf Diseases
- 📷 Upload Leaf Images
- 🤖 CNN-based Prediction
- 📊 Confidence Score
- 💊 Disease Description
- ✅ Treatment Recommendation
- ☁️ Live Deployment using Streamlit Cloud

---

## 🦠 Supported Diseases

- Potato Early Blight
- Potato Late Blight
- Healthy Potato Leaf

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

---

## 📂 Project Structure

```
Plant-Disease-Detection
│
├── Potato___Early_blight/
├── Potato___Late_blight/
├── Potato___healthy/
├── trained_plant_disease_model.keras
├── web.py
├── requirements.txt
├── diseases.png
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/akshzz007/AICTE-Internship-potato-project.git
```

Move inside folder

```bash
cd AICTE-Internship-potato-project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
streamlit run web.py
```

---

## 📷 How It Works

1. Upload a potato leaf image.
2. Click **Predict Disease**.
3. The CNN model predicts the disease.
4. Confidence score is displayed.
5. Disease description and treatment recommendation are shown.

---
