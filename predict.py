
import joblib
import pandas as pd
from pathlib import Path

# Load the saved model
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "decision_tree.joblib")
features = joblib.load(BASE_DIR / "models" / "feature_columns.joblib")

print("\n===================================")
print("   DIABETES RISK PREDICTION SYSTEM")
print("===================================")
print("Enter the required health measurements.\n")

values = []

for feature in features:
    while True:
        try:
            value = float(input(f"Enter {feature}: "))

            if not pd.notna(value) or value < 0:
                print("Enter a valid non-negative number.")
                continue

            values.append(value)
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

# Arrange inputs in the correct feature order
input_data = pd.DataFrame([values], columns=features)

# Predict using the trained model
prediction = int(model.predict(input_data)[0])

print("\n========== PREDICTION RESULT ==========")

if prediction == 1:
    print("Model output: Higher-predicted class (1)")
else:
    print("Model output: Lower-predicted class (0)")

print("\nNote: This is an educational ML prediction,")
print("not a medical diagnosis.")
print("=======================================")
