# 🌱 Agro Aid – Crop & Fertilizer Recommendation System

## 📌 Overview

Agro Aid is an AI-based agricultural decision support system that recommends:

- 🌾 Suitable Crop  
- 🌱 Soil Type  
- 🌿 Crop Variety  
- 🧪 Recommended Fertilizer  

The system uses Machine Learning models trained on agricultural datasets and provides predictions through an interactive Tkinter-based GUI.

---

## 🚀 Features

- Multi-output crop prediction (Crop, Soil Type, Variety)
- Fertilizer recommendation system
- Ensemble Machine Learning model
- Random Forest-based fertilizer classifier
- Interactive chatbot-style GUI
- Clean and modular code structure

---

## 🧠 Machine Learning Models Used

### 🌾 Crop Prediction Model

- MultiOutputClassifier
- Voting Classifier combining:
  - Random Forest
  - Decision Tree
  - Support Vector Machine (SVM)

**Input Parameters:**
- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH Value
- Rainfall

**Outputs:**
- Crop
- Soil Type
- Variety

---

### 🧪 Fertilizer Recommendation Model

- Random Forest Classifier

**Input Parameters:**
- Temperature
- Humidity
- Moisture
- Soil Type
- Crop
- Nitrogen
- Potassium
- Phosphorus

**Output:**
- Recommended Fertilizer

---

## 🖥️ Tech Stack

- Python
- Scikit-learn
- Pandas
- Joblib
- Tkinter (GUI)
- Ensemble Learning

---

## 📊 Dataset

- sensor_Crop_Dataset.csv  
- data_core.csv  

The datasets include environmental conditions, soil nutrients, and crop-related attributes.

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install pandas scikit-learn joblib
```

### 2️⃣ Train the Models

```bash
python Crop_training_model.py
python Fertilizer_training_model.py
```

### 3️⃣ Run the Application

```bash
python Agro.py
```

---

## 📂 Project Structure

```
Agro-Aid/
│
├── Agro.py
├── Crop_training_model.py
├── Fertilizer_training_model.py
├── Crop_sample_test.py
├── Fertilizer_sample_test.py
├── sensor_Crop_Dataset.csv
├── data_core.csv
├── .gitignore
└── README.md
```

---

## 🎯 Future Improvements

- Web deployment using Flask/Django
- Real-time sensor integration (IoT)
- Cloud deployment (AWS/Azure)
- Mobile application version
- Model optimization and hyperparameter tuning

---

## 👨‍💻 Author

Kamesh Kumar  
AI & Machine Learning Enthusiast
