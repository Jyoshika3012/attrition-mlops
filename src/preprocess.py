import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_preprocess(filepath: str):
    df = pd.read_csv(filepath)

    # Drop columns that add no value
    df.drop(columns=["EmployeeCount", "EmployeeNumber",
                      "Over18", "StandardHours"], inplace=True)

    # Target column — convert Yes/No to 1/0
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    # Encode all categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Save encoders so API can use them later
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/encoders.pkl")

    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    return train_test_split(X, y, test_size=0.2, random_state=42)