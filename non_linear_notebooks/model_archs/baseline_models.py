import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
import logging
import sys
import re
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.metric import evaluate, save_metrics_to_csv


def evaluate_baseline_model(model, x_val, y_val, model_name):
    """
    Evaluate a baseline model.
    
    Args:
        model: Trained model
        x_val: Validation features
        y_val: Validation labels
        model_name: Name of the model for logging
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Extract the first token features if input is 3D
    if len(x_val.shape) == 3:
        x_val_features = np.array([ins[0] for ins in x_val], dtype=np.float32)
    else:
        x_val_features = x_val
    
    y_pred = model.predict_proba(x_val_features)[:, 1]
    acc, f1, auroc = evaluate(y_val, y_pred, show=True)
    
    logging.info(f"{model_name} performance:")
    logging.info(f"Accuracy: {acc*100:.2f}%")
    logging.info(f"F1-Score: {f1*100:.2f}%")
    logging.info(f"AUROC: {auroc*100:.2f}%")
    
    return {
        "acc": acc,
        "f1": f1,
        "auroc": auroc
    }


def check_if_exists(csv_file, model_name ):
    import csv
    
    model_exists = False
    with open(csv_file, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the first column is "Model Name"
            if row[0] == model_name:
                model_exists = True
                break
    return model_exists