"""
Evaluation Metrics for MTRE

This module provides functions for computing and saving evaluation metrics
used in hallucination detection experiments.

Key Functions:
    - evaluate(): Compute accuracy, F1, and AUROC from predictions
    - save_metrics_to_csv(): Append metrics to a CSV file
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
import logging


def evaluate(y_true, y_pred, show=False, threshold=0.5, save_location=None, method=None):
    """
    Evaluate model predictions against ground truth.

    Args:
        y_true (np.ndarray): Ground truth labels (0 or 1)
        y_pred (np.ndarray): Predicted probabilities/scores
        show (bool): If True, print metrics to console
        threshold (float): Decision threshold for binary classification
        save_location (str, optional): Path to save visualization (unused)
        method (str, optional): Method name for logging (unused)

    Returns:
        tuple: (accuracy, f1_score, auroc)
            - accuracy: Proportion of correct predictions
            - f1_score: Harmonic mean of precision and recall
            - auroc: Area under ROC curve (-1 if only one class present)

    Example:
        >>> acc, f1, auroc = evaluate(y_true, y_pred, show=True)
        Accuracy: 85.50%
        F1-Score: 82.30%
        AUROC: 91.20%
    """
    # Convert continuous predictions to binary predictions
    y_pred_b = np.array([1 if y_hat > threshold else 0 for y_hat in y_pred])
    y_pred = np.squeeze(y_pred)
    valid_indices = np.isfinite(y_true) & np.isfinite(y_pred)
    y_pred = y_pred[valid_indices]
    y_true = y_true[valid_indices]
    y_pred_b = y_pred_b[valid_indices]
    # Calculate main metrics
    acc = accuracy_score(y_true, y_pred_b)
    f1 = f1_score(y_true, y_pred_b)
    auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else -1
    
    if show:
        # Print calculated metrics
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"F1-Score: {f1*100:.2f}%")
        print(f"AUROC: {auroc*100:.2f}%")
        
    
    return acc, f1, auroc


def save_metrics_to_csv(csv_file, model_name, metrics):
    """
    Save evaluation metrics to a CSV file.
    
    Args:
        csv_file: Path to CSV file
        model_name: Name of the model
        metrics: Dictionary of metrics (acc, ap, f1, auroc)
    """
    import csv
    
    model_exists = False
    # Create file with header if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Example header – replace with your actual columns
            writer.writerow(["Model Name", "Accuracy (%)", "F1-Score (%)", "AUROC (%)"])
            print(f"Created file: {csv_file}")

    with open(csv_file, mode="r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the first column is "Model Name"
            if row[0] == model_name:
                model_exists = True
                break

    # If model doesn't exist, append new row
    if not model_exists:
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            # If it's the first time writing to the file, write the header
            if file.tell() == 0:
                writer.writerow(["Model Name", "Accuracy (%)", "F1-Score (%)", "AUROC (%)"])
            
            # Write the performance metrics for the current model
            writer.writerow([
                model_name,
                metrics["acc"] * 100,
                metrics["f1"] * 100,
                metrics["auroc"] * 100
            ])
    else:
        print(f"Model {model_name} already exists in the CSV. Skipping write.")
        return True
    
    print(f"Performance metrics for {model_name} written to {csv_file}.")