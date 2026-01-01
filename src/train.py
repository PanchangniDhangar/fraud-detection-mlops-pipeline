import mlflow
import mlflow.sklearn  # Change this from mlflow.xgboost
from xgboost import XGBClassifier
from src.utils import save_object

def train_model(X_train, y_train):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    
    with mlflow.start_run():
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'random_state': 42
        }
        
        # Initialize and Train
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_params(params)
        
        # USE SKLEARN LOGGING TO AVOID THE TYPE ERROR
        mlflow.sklearn.log_model(model, "model")
        
        # Save model artifact locally for FastAPI
        save_object("artifacts/model.pkl", model)
        
        return model