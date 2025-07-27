import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

data=pd.read_csv("/content/employee_data_large.csv")

X = data.drop("salary", axis=1)
y = data["salary"]

preprocessor = ColumnTransformer([("encoder", OneHotEncoder(), ["education", "role"])], remainder="passthrough")

model = Pipeline([("preprocessing", preprocessor),("regressor", LinearRegression())])

model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
