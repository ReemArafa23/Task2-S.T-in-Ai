import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data
train = pd.read_csv('data/train.csv')
X_train = train.drop('quality', axis=1)
y_train = train['quality']

# Load params
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Train model based on params
if params['train']['model_type'] == 'random_forest':
    model = RandomForestClassifier(n_estimators=params['train']['n_estimators'], random_state=42)
else:
    model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.joblib')
print("Model trained and saved.")