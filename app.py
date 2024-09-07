import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

class FraudDetectionApp:
    def __init__(self):
        self.model = joblib.load('model/only_model.joblib')
        
        # Assuming the model has an attribute 'feature_names_in_' which stores the feature names used during training
        self.feature_names = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else [
            'incident_severity', 'insured_hobbies', 'total_claim_amount', 'months_as_customer', 'policy_annual_premium', 
            'incident_date', 'capital-loss', 'capital-gains', 'insured_education_level', 'incident_city', 'incident_state'
        ]
        
        self.categorical_columns = ['incident_severity', 'insured_hobbies', 'insured_education_level', 'incident_city', 'incident_state']
        self.encoders = {col: LabelEncoder() for col in self.categorical_columns}
        self.fit_encoders()

    def fit_encoders(self):
        # Example unique values for fitting the encoders
        example_data = {
            'incident_severity': ['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage'],
            'insured_hobbies': ['sleeping', 'reading', 'board-games', 'bungie-jumping', 'base-jumping', 'golf', 'camping', 'dancing', 'skydiving', 'movies', 'hiking', 'yachting', 'paintball', 'chess', 'kayaking', 'polo', 'basketball', 'video-games', 'cross-fit', 'exercise'],
            'insured_education_level': ['MD', 'PhD', 'Associate', 'Masters', 'High School', 'College', 'JD'],
            'incident_city': ['Columbus', 'Riverwood', 'Arlington', 'Springfield', 'Hillsdale', 'Northbend', 'Northbrook'],
            'incident_state': ['OH', 'GA', 'TX', 'IL', 'NJ', 'WA']
        }
        for col in self.categorical_columns:
            self.encoders[col].fit(example_data[col])

    # def preprocess_single_data(self, data):
    #     if not isinstance(data, pd.DataFrame):
    #         data = pd.DataFrame(data, index=[0])
    #     for col in self.categorical_columns:
    #         if col in data.columns:
    #             data[col] = self.encoders[col].transform(data[col])
    #     # Ensure the column order matches the training data
    #     data = data[self.feature_names]
    #     return data

    # 
    
    def preprocess_single_data(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, index=[0])
        
        for col in self.categorical_columns:
            if col in data.columns:
                # Transform known labels and handle unseen labels
                data[col] = [self.encoders[col].transform([x])[0] if x in self.encoders[col].classes_ else -1 for x in data[col]]
        
        # Ensure the column order matches the training data
        data = data[self.feature_names]
        return data


    def predict_single_fraud(self, data):
        data_processed = self.preprocess_single_data(data)
        prediction = self.model.predict(data_processed)[0]
        return prediction

    def run(self):
        st.title('Insurance Fraud Prediction')

        # Input fields
        incident_severity = st.selectbox('Incident Severity', ['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage'])
        insured_hobbies = st.selectbox('Insured Hobbies', ['sleeping', 'reading', 'board-games', 'bungie-jumping', 'base-jumping', 'golf', 'camping', 'dancing', 'skydiving', 'movies', 'hiking', 'yachting', 'paintball', 'chess', 'kayaking', 'polo', 'basketball', 'video-games', 'cross-fit', 'exercise'])
        total_claim_amount = st.number_input('Total Claim Amount')
        months_as_customer = st.number_input('Months as Customer')
        policy_annual_premium = st.number_input('Policy Annual Premium')
        incident_date = st.number_input('Incident Date', min_value=1, max_value=31, step=1)
        capital_loss = st.number_input('Capital Loss')
        capital_gains = st.number_input('Capital Gains')
        insured_education_level = st.selectbox('Insured Education Level', ['MD', 'PhD', 'Associate', 'Masters', 'High School', 'College', 'JD'])
        incident_city = st.selectbox('Incident City', ['Columbus', 'Riverwood', 'Arlington', 'Springfield', 'Hillsdale', 'Northbend', 'Northbrook'])
        incident_state = st.selectbox('Incident State', [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 
            'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 
            'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 
            'WI', 'WY'
        ])

        # Collecting user input
        new_data_point = {
            'incident_severity': incident_severity,
            'insured_hobbies': insured_hobbies,
            'total_claim_amount': total_claim_amount,
            'months_as_customer': months_as_customer,
            'policy_annual_premium': policy_annual_premium,
            'incident_date': incident_date,
            'capital-loss': capital_loss,
            'capital-gains': capital_gains,
            'insured_education_level': insured_education_level,
            'incident_city': incident_city,
            'incident_state': incident_state,
        }

        # Prediction button
        if st.button('Predict'):
            prediction = self.predict_single_fraud(new_data_point)
            if prediction == 0:
                st.write('This case is not fraudulent.')
            else:
                st.write('This case is fraudulent.')

        # Generate sample data
        if st.button('Generate Sample Data'):
            sample_non_fraud = self.generate_sample_data(fraud=False)
            sample_fraud = self.generate_sample_data(fraud=True)
            st.write("Non-Fraud Sample Data:")
            st.write(sample_non_fraud)
            st.write("Fraud Sample Data:")
            st.write(sample_fraud)

    def generate_sample_data(self, fraud=False):
        sample_data = {
            'incident_severity': ['Major Damage' if fraud else 'Minor Damage'],
            'insured_hobbies': ['skydiving' if fraud else 'reading'],
            'total_claim_amount': [50000 if fraud else 1000],
            'months_as_customer': [1 if fraud else 60],
            'policy_annual_premium': [10000 if fraud else 200],
            'incident_date': [15],
            'capital-loss': [1000 if fraud else 0],
            'capital-gains': [5000 if fraud else 0],
            'insured_education_level': ['PhD' if fraud else 'College'],
            'incident_city': ['Riverwood' if fraud else 'Northbrook'],
            'incident_state': ['GA' if fraud else 'IL'],
        }
        return pd.DataFrame(sample_data)

if __name__ == '__main__':
    app = FraudDetectionApp()
    app.run()