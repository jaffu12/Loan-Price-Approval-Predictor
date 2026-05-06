from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Define input structure
class LoanData(BaseModel):
    Gender: int
    Married: int
    Dependents: int
    Education: int
    Self_Employed: int
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: int

@app.get("/")
def home():
    return {"message": "FastAPI running 🚀"}

@app.post("/predict")
def predict(data: LoanData):
    df = pd.DataFrame([data.dict()])
    
    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction),
        "status": "Approved" if prediction == 1 else "Rejected"
    }