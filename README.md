# Fraud-detection-mlops-pipeliene: 
Full-Stack Explainable Fraud Detection

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Render-brightgreen)](https://fraud-detection-mlops-pipeline.onrender.com/dashboard)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Project Overview
**Fraud-Sentinel-MLOps** is a production-grade, end-to-end machine learning solution designed to identify fraudulent credit card transactions in real-time. Unlike "black-box" models, this system integrates **SHAP (SHapley Additive exPlanations)** to provide feature-level transparency for every decision.

The project bridges the gap between Data Science and ML Engineering by wrapping a high-performance **XGBoost** model in a **FastAPI** microservice, containerized with **Docker**, and presented through an interactive **Tailwind CSS** dashboard.

###  [Live Dashboard Link](https://fraud-detection-mlops-pipeline.onrender.com/dashboard)


##  Key Features

## 1. Explainable AI (XAI)
- Integrated **SHAP** values to explain the "Why" behind every prediction.
- Visualizes the top 3 drivers (e.g., Transaction Amount, V14, V4) that contribute to a fraud score.

## 2. Full-Stack Architecture
- **Backend:** FastAPI for high-performance, asynchronous request handling.
- **Frontend:** A clean, responsive dashboard built with **Tailwind CSS** and **Chart.js** for real-time data visualization.

## 3. MLOps Rigor
- **Containerization:** Fully Dockerized to ensure consistent behavior across development and production environments.
- **Experiment Tracking:** Utilized **MLflow** during training to log hyperparameters and model versions.
- **Validation:** Automated **Unit Testing** with Pytest to ensure API contract reliability.

## 4. High-Performance API
- Built with **FastAPI** for asynchronous request handling.
- **Pydantic** data validation ensures that only correctly formatted transaction data is processed.
- **CORS Middleware** configured for secure frontend-backend communication.

## 5. Interactive Dashboard
- **Tailwind CSS:** A modern, responsive UI for fraud analysts.
- **Chart.js:** Dynamic bar charts that visualize SHAP feature importance in real-time.

---

## System Architecture & Data Flow

1. **Input**: 29 features (V1–V28 PCA + Amount)
2. **Validation**: Pydantic schema enforcement
3. **Inference**: XGBoost model prediction
4. **Explainability**: SHAP feature attribution
5. **Response**: JSON output (label + probability + explanations)
6. **Visualization**: Interactive dashboard rendering
   
---

##  Tech Stack

- **MLOps:** MLflow (Experiment Tracking), Pytest (CI Testing & Quality Assurance)
- **Machine Learning:** XGBoost Classifier (Optimized via Threshold Tuning),Scikit-learn, SHAP (XAI), NumPy, Pandas
- **Backend:** FastAPI, Uvicorn, Pydantic, aiofiles
- **Frontend:** HTML5, Tailwind CSS, JavaScript (Fetch API, Chart.js)
- **Monitoring/XAI:** SHAP, MLflow
- **Testing:** Pytest, HTTPX
- **DevOps & Cloud:** Docker, Render, GitHub Actions (CI/CD)

---

##  Project Structure

```
├── artifacts/           # Serialized models and scalers
├── src/                 # Training & evaluation logic
├── static/              # Frontend files
├── tests/               # Unit and integration tests
├── app.py               # FastAPI app
├── main.py              # Training pipeline
├── Dockerfile           # Container configuration
├── requirements.txt     # Dependencies
```

---

##  Installation & Execution

## Docker Deployment (Recommended)

```bash
git clone https://github.com/PanchangniDhangar/fraud-detection-mlops-pipeline.git
cd fraud-detection-mlops-pipeline
docker build -t fraud-app .
docker run -p 8000:8000 fraud-app
```

Access the dashboard at:

```
http://localhost:8000/dashboard
```
---

## Local Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
---

##  Testing & Quality Assurance

Run the test suite:

```bash
pytest
```

Tests validate:

* API availability
* Input schema validation
* Model prediction consistency
* SHAP output correctness

---

##  Model Performance & Impact

* **ROC-AUC:** ~99%
* **Optimization Goal:** Maximize Recall
* **Reason:** False negatives are costlier than false positives

### Business Impact

* Faster fraud investigation
* Improved analyst trust
* Transparent decision-making
* Reduced resolution time

---

##  Author

**Panchangni Dhangar**  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/panchangni-dhangar/)

---

##  Summary

Fraud-Sentinel-MLOps is a full-stack, explainable, production-ready ML system that demonstrates how modern data science, software engineering, and MLOps come together to solve real-world financial problems.

---
