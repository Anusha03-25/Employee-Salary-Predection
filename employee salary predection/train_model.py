import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("employee_data.csv")

X = data.drop("salary", axis=1)
y = data["salary"]

# Preprocessing: One-hot encoding for categorical features
preprocessor = ColumnTransformer([
    ("encoder", OneHotEncoder(), ["education", "role"])
], remainder="passthrough")

# Pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])

# Train
model.fit(X, y)

# Save
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
