from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # Add this import
import uvicorn
import numpy as np
import shap
from pydantic import BaseModel, Field
from src.utils import load_object

app = FastAPI(title="Explainable Fraud Detection API")

# ADD THIS BLOCK RIGHT HERE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your index.html to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
model = load_object("artifacts/model.pkl")
scaler = load_object("artifacts/scaler.pkl")

# Initialize SHAP Explainer (XGBoost specific)
# We use the model's tree structure for fast explanations
explainer = shap.TreeExplainer(model)

class TransactionRequest(BaseModel):
    features: list = Field(..., example=[0.0] * 29)

@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is live!", "docs": "/docs"}

@app.post("/predict")
def predict(request: TransactionRequest):
    # Move the length check to the very top
    if not request.features or len(request.features) != 29:
        raise HTTPException(
            status_code=400, 
            detail="Expected 29 features"
        )
    
    try:
        # 1. Preprocess & Scale
        data_array = np.array(request.features).reshape(1, -1)
        scaled_data = scaler.transform(data_array)
        
        # 2. Prediction
        prediction = int(model.predict(scaled_data)[0])
        probability = float(model.predict_proba(scaled_data)[0][1])

        # 3. SHAP Explainability
        # Calculate SHAP values for this specific input
        shap_values = explainer.shap_values(scaled_data)
        
        # Get feature names (V1...V28, Amount)
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        
        # Create a dictionary of {Feature_Name: SHAP_Value}
        # A positive SHAP value increases the fraud probability
        explanations = dict(zip(feature_names, shap_values[0].tolist()))
        
        # Sort features by impact (absolute SHAP value) and take top 3
        top_drivers = sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        return {
            "is_fraud": prediction,
            "fraud_probability": round(probability, 4),
            "label": "Fraudulent" if prediction == 1 else "Legitimate",
            "explanation": {
                "top_3_drivers": {feature: round(val, 4) for feature, val in top_drivers},
                "insight": "Positive values push the model toward a Fraud prediction."
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8000)