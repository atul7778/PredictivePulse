import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('patient_data.csv')

# Clean Systolic and Diastolic columns (like '130+')
def convert_range_to_float(value):
    if isinstance(value, str):
        if '+' in value:
            return float(value.replace('+', '').strip())
        elif '-' in value:
            parts = value.split('-')
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
    return float(value)

data['Systolic'] = data['Systolic'].apply(convert_range_to_float)
data['Diastolic'] = data['Diastolic'].apply(convert_range_to_float)

# All feature columns (13)
features = ['C', 'Age', 'History', 'Patient', 'TakeMedication', 'Severity',
            'BreathShortness', 'VisualChanges', 'NoseBleeding',
            'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet']

X = data[features]
y = data['Stages']

# Encode categorical features
encoders = {}
for column in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    encoders[column] = le

# Encode target separately
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
with open('model_artifacts/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model_artifacts/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

with open('model_artifacts/target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)

print("Model and encoders saved successfully.")
