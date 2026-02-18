import joblib
import pandas as pd

# Load model and encoders
model = joblib.load('fertilizer_model.pkl')
encoders = joblib.load('fertilizer_label_encoders.pkl')

# Sample input
sample_input = {
    'Temperature': 26,
    'Humidity': 75,
    'Moisture': 40,
    'Soil_Type': 'Loamy',
    'Crop': 'Wheat',          
    'Nitrogen': 50,
    'Potassium': 30,
    'Phosphorus': 40
}

# Encode categorical values
sample_input['Soil_Type'] = encoders['Soil_Type'].transform([sample_input['Soil_Type']])[0]
sample_input['Crop'] = encoders['Crop'].transform([sample_input['Crop']])[0]

# Prepare input in SAME order as training
features = [
    sample_input['Temperature'],
    sample_input['Humidity'],
    sample_input['Moisture'],
    sample_input['Soil_Type'],
    sample_input['Crop'],
    sample_input['Nitrogen'],
    sample_input['Potassium'],
    sample_input['Phosphorus']
]

# Predict
predicted_code = model.predict([features])[0]
fertilizer_name = encoders['FertilizerName'].inverse_transform([predicted_code])[0]

print("✅ Fertilizer Model Test")
print(f"Recommended Fertilizer: {fertilizer_name}")
