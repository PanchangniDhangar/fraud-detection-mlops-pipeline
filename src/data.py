import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import save_object

def process_data(data_path):
    df = pd.read_csv(data_path)
    
    # Drop 'id' if it exists in your Kaggle dataset
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for use in FastAPI later
    save_object("artifacts/scaler.pkl", scaler)
    
    return X_train_scaled, X_test_scaled, y_train, y_test