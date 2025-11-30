import pandas as pd
import numpy as np
import os
import random

# -------------------------------
# CONFIG
# -------------------------------
ROWS = 1500
random.seed(42)
np.random.seed(42)

# -------------------------------
# CATEGORIES
# -------------------------------
ITEM_CATEGORIES = ["Beverages", "Snacks", "Dairy", "Household", "Frozen", "Produce"]
STORE_TYPES = ["Grocery", "Supermarket Type1", "Supermarket Type2"]
LOCATION_TIERS = ["Tier 1", "Tier 2", "Tier 3"]
OUTLET_SIZES = ["Small", "Medium", "Large"]

# -------------------------------
# DATA GENERATION
# -------------------------------
df = pd.DataFrame({
    "Item_Category": random.choices(ITEM_CATEGORIES, k=ROWS),
    "Store_Type": random.choices(STORE_TYPES, k=ROWS),
    "Outlet_Location_Type": random.choices(LOCATION_TIERS, k=ROWS),
    "Outlet_Size": random.choices(OUTLET_SIZES, k=ROWS),
    "Item_Weight": np.round(np.random.uniform(5, 25, ROWS), 2),
    "Item_MRP": np.round(np.random.uniform(10, 120, ROWS), 2),
    "Outlet_Sales": np.random.randint(5, 500, ROWS)
})

# -------------------------------
# SAVE CSV
# -------------------------------
os.makedirs("data", exist_ok=True)

path = "data/supermarket_features.csv"
df.to_csv(path, index=False)

print("\n✅ HUMAN-READABLE DATASET CREATED")
print("📁 Saved at:", path)
print("\nColumns Included:")
print(df.columns.tolist())
print("\nPreview:")
print(df.head())
