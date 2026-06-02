from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Employee Attrition Predictor")

# Load model and encoders when API starts
model = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")

# This defines exactly what data the API expects
class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

@app.get("/")
def home():
    return {"message": "Attrition Predictor API is running"}

@app.post("/predict")
def predict(data: EmployeeData):
    # Convert input to dataframe
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # Encode categorical columns using saved encoders
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "attrition": bool(prediction),
        "probability": round(float(probability), 4),
        "risk": "High" if probability > 0.5 else "Low"
    }