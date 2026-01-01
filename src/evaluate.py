from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import mlflow

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, preds, output_dict=True)
    auc_score = roc_auc_score(y_test, probs)
    
    # Log metrics to MLflow
    mlflow.log_metric("precision", report['1']['precision'])
    mlflow.log_metric("recall", report['1']['recall'])
    mlflow.log_metric("roc_auc", auc_score)
    
    print(f"ROC-AUC Score: {auc_score}")
    print(confusion_matrix(y_test, preds))