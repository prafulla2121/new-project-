import joblib
import pandas as pd

# Load model
model = joblib.load("lgbm_final.pkl")

# Load new data (with same features as training)
new_data = pd.read_csv("datasets/new_data.csv")

# Make predictions
preds = model.predict(new_data)
proba = model.predict_proba(new_data)[:, 1]

print("Predictions:", preds[:10])
print("Probabilities:", proba[:10])
