from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
import os
import shap
from pydantic import BaseModel, Field
from src.utils import load_object

# --- 1. SETUP PATHS ---
# This ensures Python finds the 'static' folder regardless of how you run the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# --- 2. INITIALIZE APP ---
app = FastAPI(
    title="Fraud-Sentinel-MLOps API",
    description="Real-time Explainable Fraud Detection",
    version="1.0.0"
)

# Enable CORS so your browser doesn't block requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. LOAD ML ARTIFACTS ---
try:
    model = load_object(os.path.join(ARTIFACTS_DIR, "model.pkl"))
    scaler = load_object(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    print("✅ Model, Scaler, and SHAP Explainer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading artifacts: {e}")

# --- 4. DATA MODELS ---
class TransactionRequest(BaseModel):
    features: list = Field(..., example=[0.0] * 29)

# --- 5. API ENDPOINTS ---

@app.post("/predict")
async def predict(request: TransactionRequest):
    try:
        if len(request.features) != 29:
            raise HTTPException(status_code=400, detail="Expected 29 features")

        # Preprocess
        data_array = np.array(request.features).reshape(1, -1)
        scaled_data = scaler.transform(data_array)

        # Predict
        prediction = int(model.predict(scaled_data)[0])
        probability = float(model.predict_proba(scaled_data)[0][1])

        # SHAP Explainability
        shap_values = explainer.shap_values(scaled_data)
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        explanations = dict(zip(feature_names, shap_values[0].tolist()))
        top_drivers = sorted(explanations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        return {
            "is_fraud": prediction,
            "fraud_probability": round(probability, 4),
            "label": "Fraudulent" if prediction == 1 else "Legitimate",
            "explanation": {
                "top_3_drivers": {feature: round(val, 4) for feature, val in top_drivers}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. STATIC FILES & DASHBOARD ---

# Mount the static folder so http://127.0.0.1:8000/static/index.html works
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    print(f"⚠️ Warning: {STATIC_DIR} not found. Create a folder named 'static'!")

@app.get("/dashboard")
async def get_dashboard():
    file_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": f"index.html not found in {STATIC_DIR}"}

@app.get("/")
async def root():
    return {"message": "API is online. Visit /dashboard for the UI or /docs for API info."}

# --- 7. START SERVER ---
if __name__ == "__main__":
    # Debug info
    print(f"DEBUG: Static directory is set to: {STATIC_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000)