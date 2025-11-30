# src/train_model.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ---------- PATHS ----------
DATA_PATH = Path("../data/supermarket_sales.csv")
MODEL_PATH = Path("../api/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# ---------- LOAD DATA ----------
df = pd.read_csv(DATA_PATH)

# ---------- FEATURES ----------
categorical_cols = ["Item_Category", "Store_Type", "Outlet_Location_Type", "Outlet_Size"]
numerical_cols = ["Item_Weight", "Item_MRP", "Outlet_Sales"]

X = df[categorical_cols + numerical_cols]

# ---------- CREATE TARGET (Revenue) ----------
store_effect = {"Grocery": 1500, "Supermarket Type1": 3000, "Supermarket Type2": 4500}
size_effect = {"Small": -500, "Medium": 0, "Large": 1500}

np.random.seed(42)
y = (
    df["Outlet_Sales"] * df["Item_MRP"] +
    df["Item_Weight"] * 100 +
    df["Store_Type"].map(store_effect).fillna(0) +
    df["Outlet_Size"].map(size_effect).fillna(0) +
    np.random.normal(0, 1000, len(df))
).round(2)

# ---------- TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- PREPROCESSING ----------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# ---------- MODELS ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42)
}

results = {}
pipelines = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results[name] = {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}
    pipelines[name] = pipeline

# ---------- SELECT BEST MODEL ----------
best_model_name = max(results, key=lambda k: results[k]["R2"])
best_pipeline = pipelines[best_model_name]

print("=== MODEL METRICS ===")
for model, metrics in results.items():
    print(f"{model}: {metrics}")

print(f"\n✅ Best model: {best_model_name}")

# ---------- SAVE BEST MODEL ----------
joblib.dump(best_pipeline, MODEL_PATH / "model.pkl")
print(f"✅ Saved model to {MODEL_PATH / 'model.pkl'}")
