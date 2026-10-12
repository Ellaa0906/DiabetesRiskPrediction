
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("diabetes_data_upload.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Select target column
df.columns = df.columns.str.strip().str.lower()
target = "outcome"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' was not found.")

# 3. Clean dataset
df = df.drop_duplicates().copy()
df = df.dropna(subset=[target]).copy()

# 4. Separate features and target
X = df.drop(columns=[target])
y = df[target].astype(int)

print("\nTarget distribution:")
print(y.value_counts())

# 5. Identify numerical columns
numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

# 6. Preprocess missing values and scale features
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical_cols
        )
    ]
)

# 7. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Create and train the ML model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
    XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)
    )
])

print("\nTraining the model...")
model.fit(X_train, y_train)

# 9. Make predictions
predictions = model.predict(X_test)

# 10. Evaluate model
print("\n========== MODEL EVALUATION ==========")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 11. Save trained model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgboost_model.joblib")
joblib.dump(X.columns.tolist(), "models/feature_columns.joblib")

print("\nModel saved successfully!")
print("Model path: models/decision_tree.joblib")
print("Features saved: models/feature_columns.joblib")
