# Insurance Fraud Prediction Model
 
This project focuses on building and evaluating a machine learning model to detect fraudulent insurance claims. 
The project involves data preprocessing, model training using a RandomForestClassifier, model evaluation with 
various metrics and visualizations, and a Streamlit UI for interacting with the model.

Create and activate a virtual environment:

```bash 
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`

```

Install the required packages:

```bash
    pip install -r requirements.txt
```

### Project Structure
```bash
insurance-fraud-detection/
│
├── dataset/
│   └── insurance_claims.csv
│
├── model/
│   └── only_model.joblib
│
├── train.py
├── prediction.py
├── app.py
├── requirements.txt
└── README.md
```


### Data Preprocessing
#### Data Loading
The data is loaded from a CSV file located at dataset/insurance_claims.csv. During loading, the following steps are 
performed:

- Drop the _c39 column.
- Replace '?' with NaN.

#### Data Cleaning
Fill missing values for 'property_damage', 'police_report_available', and 'collision_type' columns with their mode.
Drop duplicate records.

#### Encoding and Feature Selection
Encode categorical variables using Label Encoding.
Drop unnecessary columns that are not relevant for the model.
Select the final set of features for the model.

#### Preprocessed Features
The final set of features used for model training:

incident_severity
insured_hobbies
total_claim_amount
months_as_customer
policy_annual_premium
incident_date
capital-loss
capital-gains
insured_education_level
incident_city
fraud_reported (target variable)

####  Model Training
The model is trained using a RandomForestClassifier with a pipeline that includes preprocessing steps and 
hyperparameter tuning using GridSearchCV.

#### Training Steps
Train-test split: The data is split into training and testing sets with a 70-30 split.
Pipeline setup: A pipeline is created to include preprocessing and model training.
Hyperparameter tuning: A grid search is performed to find the best hyperparameters.
Model training: The best model is trained on the training data.
Model saving: The trained model is saved as fraud_insurance_pipeline.joblib.

#### Model Evaluation
The trained model is evaluated using the test set. The evaluation metrics include:

Classification Report: Precision, Recall, F1-score.
AUC Score: Area Under the ROC Curve.
Confusion Matrix: Visual representation of true vs. predicted values.
ROC Curve: Receiver Operating Characteristic curve.


### Usage

#### Training the Model
To train the model, run the following command:

```bash
python train.py
```
#### Evaluating the Model

To evaluate the model, run the following command:

```bash
python predict.py
```
#### Running the Streamlit App
To run the Streamlit app, use the following command:
```bash
streamlit run streamlit_app.py
```