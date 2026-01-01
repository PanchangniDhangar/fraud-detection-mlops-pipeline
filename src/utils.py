import os
import pickle
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.error(f"Error saving object at {file_path}")
        raise e

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.error(f"Error loading object at {file_path}")
        raise e