import os
import sys
import logging
from src.data import process_data
from src.train import train_model
from src.evaluate import evaluate_model

# Ensure the root directory is in the python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def run_pipeline(data_path):
    try:
        logging.info(">>> Stage 1: Data Ingestion & Transformation started")
        X_train, X_test, y_train, y_test = process_data(data_path)
        logging.info(">>> Stage 1 completed successfully")

        logging.info(">>> Stage 2: Model Training & MLflow Logging started")
        model = train_model(X_train, y_train)
        logging.info(">>> Stage 2 completed successfully")

        logging.info(">>> Stage 3: Model Evaluation started")
        evaluate_model(model, X_test, y_test)
        logging.info(">>> Stage 3 completed successfully")
        
        print("\nPipeline execution finished! Check MLflow for details.")

    except Exception as e:
        logging.error(f"Pipeline failed at stage: {e}")
        raise e

if __name__ == "__main__":
    # Update this path to where your Kaggle CSV is stored
    DATA_FILE_PATH = "data/creditcard_2023.csv" 
    run_pipeline(DATA_FILE_PATH)