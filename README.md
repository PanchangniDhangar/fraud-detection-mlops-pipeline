# Fraud-detection-mlops-pipeliene: Full-Stack Explainable Fraud Detection

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Render-brightgreen)](https://fraud-detection-mlops-pipeline.onrender.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

##  Project Overview
**Fraud-Sentinel-MLOps** is a production-grade, end-to-end machine learning solution designed to identify fraudulent credit card transactions in real-time. Unlike "black-box" models, this system integrates **SHAP (SHapley Additive exPlanations)** to provide feature-level transparency for every decision.

The project bridges the gap between Data Science and ML Engineering by wrapping a high-performance **XGBoost** model in a **FastAPI** microservice, containerized with **Docker**, and presented through an interactive **Tailwind CSS** dashboard.

###  [Live Dashboard Link](https://fraud-detection-mlops-pipeline.onrender.com)


##  Key Features

## 1. Explainable AI (XAI)
- Integrated **SHAP** values to explain the "Why" behind every prediction.
- Visualizes the top 3 drivers (e.g., Transaction Amount, V14, V4) that contribute to a fraud score.

## 2. MLOps & Reproducibility
- **MLflow:** Used for experiment tracking, metric logging (Precision/Recall/AUC), and model versioning.
- **DVC:** (Optional) Data versioning to ensure pipeline consistency.
- **Docker:** Fully containerized architecture to ensure "it works on any machine."

## 3. High-Performance API
- Built with **FastAPI** for asynchronous request handling.
- **Pydantic** data validation ensures that only correctly formatted transaction data is processed.
- **CORS Middleware** configured for secure frontend-backend communication.

## 4. Interactive Dashboard
- **Tailwind CSS:** A modern, responsive UI for fraud analysts.
- **Chart.js:** Dynamic bar charts that visualize SHAP feature importance in real-time.

---

##  Architecture & Tech Stack

- **Model:** XGBoost Classifier (Optimized via Threshold Tuning)
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** HTML5, Tailwind CSS, JavaScript (Fetch API, Chart.js)
- **Monitoring/XAI:** SHAP, MLflow
- **Testing:** Pytest, HTTPX
- **Deployment:** Docker, Render, GitHub Actions (CI/CD)

---

##  Project Structure

```text
├── artifacts/               # Saved models (pkl) and scalers
├── data/                    # Dataset samples (Kaggle 2023)
├── src/                     # Modular Python logic
│   ├── data.py              # Data ingestion & transformation
│   ├── train.py             # Training & MLflow logging
│   ├── evaluate.py          # Model metrics & evaluation
│   └── utils.py             # Helper functions (Pickle/Logging)
├── tests/                   # Automated unit tests (Pytest)
├── app.py                   # FastAPI backend + SHAP logic
├── index.html               # Frontend dashboard
├── Dockerfile               # Container configuration
├── requirements.txt         # Project dependencies
└── main.py                  # Pipeline orchestrator
