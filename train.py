import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings("ignore")

# Load and preprocess data
data = pd.read_csv("dataset/insurance_claims.csv").drop(columns="_c39")
data.replace('?', np.nan, inplace=True)

# Function to check data
def check_data(data):
    return pd.DataFrame({
        'type': data.dtypes,
        'amount_unique': data.nunique(),
        'unique_values': [data[x].unique() for x in data.columns],
        'null_values': data.isna().sum(),
        'percentage_null_values(%)': round((data.isnull().sum() / data.shape[0]) * 100, 2)
    })

print(check_data(data).sort_values("null_values", ascending=False))

# Fill missing values with mode
for column in data.columns:
    mode_value = data[column].mode().iloc[0]
    data[column] = data[column].replace(np.nan, mode_value)

# Encode categorical variables
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'O':
        data[col] = le.fit_transform(data[col])

# Drop less important columns
to_drop = ['policy_number', 'policy_bind_date', 'insured_zip', 'incident_location',
           'auto_year', 'auto_make', 'auto_model']
data.drop(columns=to_drop, inplace=True)

# Correlation heatmap
plt.figure(figsize=(23, 23))
corr_matrix = data.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(round(corr_matrix, 2), mask=mask, vmin=-1, vmax=1, annot=True, cmap='magma')
plt.title('Triangle Correlation Heatmap', fontsize=18, pad=16)
plt.show()

# Drop less correlated features
to_drop = ['injury_claim', 'property_claim', 'vehicle_claim', 'incident_type', 'age',
           'incident_hour_of_the_day', 'insured_occupation']
data.drop(columns=to_drop, inplace=True)

# Feature importance
X = data.iloc[:, :-1]
Y = data['fraud_reported']
model = RandomForestClassifier(n_estimators=1000)
model.fit(X, Y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
final_feat = feat_importances.nlargest(10).index.tolist()
final_feat.append('fraud_reported')
data_new = data[final_feat]

# Prepare data for modeling
df_model = data_new.copy()
X = df_model.drop(columns='fraud_reported')
y = df_model['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=43)

# Train the final model
final_model = RandomForestClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_leaf=4,
    min_samples_split=10,
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
final_model.fit(X_train, y_train)

# Evaluate the model
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Save the model
joblib.dump(final_model, 'model/only_model.joblib')
print("Model saved successfully.")
