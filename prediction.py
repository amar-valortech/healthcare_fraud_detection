import pandas as pd
import numpy as np
import joblib

# Load the model and preprocessing pipeline
model = joblib.load('model/only_model.joblib')
def preprocess_single_data(data):
    # Convert the data into a DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, index=[0])
    
    # handle missing values by replacing with mode
    for column in data.columns:
        mode_value = data[column].mode().iloc[0]
        data[column] = data[column].replace(np.nan, mode_value)
    
def predict_single_fraud(data):
    # Preprocess the single data point
    data_processed = preprocess_single_data(data)
    
    # Standardize the data
    # data_scaled = scaler.transform(data_processed)
    
    # Make predictions
    prediction = model.predict(data_processed)[0]
    probability = model.predict_proba(data_processed)[0, 1]
    
    return prediction, probability

# Example usage:
# New single data point
new_data_point = {
    'incident_severity': 'Major Damage',
    'insured_hobbies': 9,
    'total_claim_amount': 59670,
    'months_as_customer': 116,
    'policy_annual_premium': 951.46,
    'incident_date': 30,
    'capital-loss': -35500,
    'capital-gains': 0,
    'insured_education_level': 3,
    'incident_city':5,
}

# Make predictions
prediction = predict_single_fraud(new_data_point)

# Display predictions
print(f'Fraud Prediction: {prediction}')
# print(f'Probability of Fraud: {probability:.4f}')
