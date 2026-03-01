import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score

# Load data and model
test = pd.read_csv('data/test.csv')
X_test = test.drop('quality', axis=1)
y_test = test['quality']
model = joblib.load('model.joblib')

# Predict
predictions = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

# Save metrics
with open('metrics.json', 'w') as f:
    json.dump({'accuracy': acc, 'f1_score': f1}, f, indent=4)

# Save plot data for DVC confusion matrix
pd.DataFrame({'actual': y_test, 'predicted': predictions}).to_csv('predictions.csv', index=False)
print("Validation complete. Metrics and plot data saved.")