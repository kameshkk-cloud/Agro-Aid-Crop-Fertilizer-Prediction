Agro Aid – Crop & Fertilizer Recommendation System

Overview  
Agro Aid is an AI-based agricultural decision support system that recommends suitable crops, soil types, crop varieties, and fertilizers based on environmental and soil conditions. The system uses Machine Learning models trained on agricultural datasets and provides predictions through an interactive Tkinter-based GUI.

Features  
- Multi-output crop prediction (Crop, Soil Type, Variety)  
- Fertilizer recommendation system  
- Ensemble Machine Learning model  
- Random Forest-based fertilizer classifier  
- Interactive chatbot-style GUI  
- Clean and modular code structure  

Machine Learning Models Used  

Crop Prediction Model  
- MultiOutputClassifier  
- Voting Classifier combining:  
  - Random Forest  
  - Decision Tree  
  - Support Vector Machine (SVM)  

Input Parameters:  
- Nitrogen  
- Phosphorus  
- Potassium  
- Temperature  
- Humidity  
- pH Value  
- Rainfall  

Outputs:  
- Crop  
- Soil Type  
- Variety  

Fertilizer Recommendation Model  
- Random Forest Classifier  

Input Parameters:  
- Temperature  
- Humidity  
- Moisture  
- Soil Type  
- Crop  
- Nitrogen  
- Potassium  
- Phosphorus  

Output:  
- Recommended Fertilizer  

Tech Stack  
- Python  
- Scikit-learn  
- Pandas  
- Joblib  
- Tkinter  
- Ensemble Learning  

Dataset  
- sensor_Crop_Dataset.csv  
- data_core.csv  

The datasets include environmental conditions, soil nutrients, and crop-related attributes.

How to Run  

Install dependencies:  
pip install pandas scikit-learn joblib  

Train the models:  
python Crop_training_model.py  
python Fertilizer_training_model.py  

Run the application:  
python Agro.py  

Project Structure  
Agro-Aid/  
├── Agro.py  
├── Crop_training_model.py  
├── Fertilizer_training_model.py  
├── Crop_sample_test.py  
├── Fertilizer_sample_test.py  
├── sensor_Crop_Dataset.csv  
├── data_core.csv   
└── .gitignore  
.
