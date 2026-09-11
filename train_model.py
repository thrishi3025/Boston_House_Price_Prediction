import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. Load Dataset
# =========================================================

df = pd.read_csv("BostonHousingData.csv")

print("Dataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# =========================================================
# 2. Separate Features and Target
# =========================================================

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# =========================================================
# 3. Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# =========================================================
# 4. Create Machine Learning Pipeline
# =========================================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# =========================================================
# 5. Train Model
# =========================================================

model.fit(X_train, y_train)

# =========================================================
# 6. Prediction
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# 7. Model Evaluation
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------------------")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 :", r2)

# =========================================================
# 8. Save Model
# =========================================================

joblib.dump(model, "house_price_model.pkl")

print("\nModel saved as house_price_model.pkl")
