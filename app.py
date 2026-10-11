
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature names
model = joblib.load("models/decision_tree.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            # Read input values from the form
            for feature in feature_columns:
                form_data[feature] = request.form.get(feature, "").strip()

            # Check that every field is filled
            if any(value == "" for value in form_data.values()):
                raise ValueError("Please fill in all fields.")

            # Convert inputs to numbers
            input_data = pd.DataFrame(
                [[float(form_data[col]) for col in feature_columns]],
                columns=feature_columns
            )

            # Validate that values are finite
            if input_data.isnull().any().any():
                raise ValueError("Please enter valid numeric values.")

            # Make prediction: 0 = negative, 1 = positive
            result = int(model.predict(input_data)[0])

            prediction = {
                "label": (
                    "Higher predicted risk"
                    if result == 1
                    else "Lower predicted risk"
                ),
                "result": result
            }

        except (ValueError, TypeError):
            error = "Enter valid numbers in all fields."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)
