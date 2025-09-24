import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv("../../data/train.csv")

features = ["OverallQual", "GrLivArea", "GarageCars", "YearBuilt", "Neighborhood"]
X = data[features]
y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical = ["Neighborhood"]
numeric = ["OverallQual", "GrLivArea", "GarageCars", "YearBuilt"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

joblib.dump(model, "../../models/house_price_model.pkl")
print("Model trained and saved")
