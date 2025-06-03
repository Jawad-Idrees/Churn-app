from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import uvicorn
import joblib
#from custom_transformers import drop_columns
import cloudpickle as pkl

# Load the model from MLflow
# model_uri = 'runs:/9f22e965ab5a4b60b7e240a70c9a7cd5/churn_pipeline_model'
# model = mlflow.sklearn.load_model(model_uri)


model = joblib.load('fastapi_app/pipeline.pkl')

# with open("pipeline.pkl", "rb") as f:
#     model = pkl.load(f)
# Create FastAPI app
app = FastAPI(title="Churn Prediction API")

# Define request schema
class UserInput(BaseModel):
    Gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    CustomerID: str  # Will be dropped in the pipeline

# @app.post("/predict")
# def predict(data: UserInput):
#     input_df = pd.DataFrame([data.dict()])
#     prediction = model.predict(input_df)[0]
#     print(int(prediction))
#     return {"prediction": int(prediction)}  # 0 or 1

@app.post("/predict")
def predict(data: UserInput):
    input_df = pd.DataFrame([data.dict()])
    print("Input to model:\n", input_df)
    prediction = model.predict(input_df)[0]
    print(int(prediction))
    return {"prediction": int(prediction)}



# For running directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
