import pandas as pd
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


print("\n🧠 Model is training...")
print("⏳ Please wait...\n")

# Give terminal feedback immediately
time.sleep(1)

# =============================
# Load Dataset
# =============================
crop_df = pd.read_csv("sensor_Crop_Dataset.csv")
print("✅ Dataset loaded")

# =============================
# Encode Labels
# =============================
label_encoders = {}

for col in ['Crop', 'Soil_Type', 'Variety']:
    le = LabelEncoder()
    crop_df[col] = le.fit_transform(crop_df[col])
    label_encoders[col] = le

print("✅ Labels encoded")

# =============================
# Features & Targets
# =============================
X = crop_df[
    ['Nitrogen', 'Phosphorus', 'Potassium',
     'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
]

y = crop_df[['Crop', 'Soil_Type', 'Variety']]

# =============================
# Train-Test Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data split completed")

# =============================
# Model Definition
# =============================
voting = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    voting='soft'
)

model = MultiOutputClassifier(voting)

print("🚀 Training started...")

# =============================
# Train Model
# =============================
model.fit(X_train, y_train)

print("✅ Training completed")

# =============================
# Accuracy
# =============================
accuracy = model.score(X_test, y_test)

# =============================
# Save Model
# =============================
joblib.dump(model, "ensemble_crop_model.pkl")
joblib.dump(label_encoders, "crop_label_encoders.pkl")

print("\n🎉 Model saved successfully")
print(f"📊 Crop Model Accuracy: {accuracy:.4f}")
