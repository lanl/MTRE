import torch
import numpy as np
import os
import sys
import logging
import csv
import re
import copy
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.metric import evaluate
from sklearn.metrics import accuracy_score

class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def update_csv(name, acc, f1, auroc, csv_file):
        """Update CSV with metrics, overwriting if entry exists."""
        if not csv_file:
            return
        try:
            with open(csv_file, "r", newline="") as f:
                existing_rows = list(csv.reader(f))
        except FileNotFoundError:
            existing_rows = []
        
        updated = False
        for row in existing_rows:
            if row and row[0] == name:
                row[1:] = [acc*100, f1*100, auroc*100]
                updated = True
                break
        if not updated:
            existing_rows.append([name, acc*100, f1*100, auroc*100])

        with open(csv_file, "w", newline="") as f:
            csv.writer(f).writerows(existing_rows)

def log_confidence_and_metrics(upto, val_preds, val_labels, old_preds, old_preds_binary):
    """
    Logs confidence change, prediction flips, and performance metrics.
    Returns updated old_preds, old_preds_binary, and computed metrics.
    """
    # Confidence change logging
    if old_preds is not None:
        diff = np.mean(np.abs(val_preds - old_preds))
        logging.info(f"Confidence change Loc [{upto}] → Loc [{upto+1}]: {diff:.6f}")
    old_preds = val_preds

    # Compute accuracy & flips
    val_preds_binary = (val_preds > 0).astype(int)
    if old_preds_binary is not None:
        flips = np.sum(val_preds_binary != old_preds_binary)
        logging.info(f"Loc [{upto}:{upto+1}] Prediction flips: {flips}/{len(val_preds_binary)} "
                     f"({flips/len(val_preds_binary)*100:.2f}%)")
    old_preds_binary = val_preds_binary

    val_accuracy = accuracy_score(val_labels, val_preds_binary)
    val_preds_flat = np.array(val_preds).reshape(-1)  # Force 1D
    val_labels_flat = np.array(val_labels).reshape(-1)

    acc, f1, auroc = evaluate(val_labels_flat, val_preds_flat, show=False)
    logging.info(f"Loc [{upto+1}] Val Acc: {val_accuracy:.4f} | AUC: {auroc:.4f} | F1: {f1:.4f}")

    return old_preds, old_preds_binary, (val_accuracy, acc, f1, auroc)

def eval_model(
    x_val, y_val, model, batch_size, device,
    model_name, dataset_name, prompt, type_num, response,
    csv_file, early_stopping, embed_dim, num_heads, num_layers
):
    epsilon = 1e-10
    val_loader = DataLoader(BinaryDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    best_auc = 0
    best_upto = 0
    
    def forward_with_mask(inputs, upto):
        """Forward pass accumulating predictions up to a given location."""
        batch_log_sum_1 = torch.zeros(inputs.size(0), device=device)
        batch_log_sum_2 = torch.zeros(inputs.size(0), device=device)
        
        for loc in range(upto + 1):
            preds, _ = model(inputs[:, loc, :])
            non_zero_mask = (inputs[:, loc, :].norm(dim=1) > epsilon)
            batch_log_sum_1 += torch.log(preds[:, 0]) * non_zero_mask
            batch_log_sum_2 += torch.log(1 - preds[:, 0]) * non_zero_mask
        
        return batch_log_sum_1 - batch_log_sum_2

    logging.info("Performance Aggregated Attention:")
    old_preds, old_preds_binary = None, None
    aggregated_acc, aggregated_auc = [], []

    model.eval()
    for upto in range(10):
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = forward_with_mask(inputs, upto)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        old_preds, old_preds_binary, metrics = log_confidence_and_metrics(
            upto, val_preds, val_labels, old_preds, old_preds_binary
        )
        val_accuracy, acc, f1, auroc = metrics
        if auroc > best_auc:
            best_upto = upto
            best_auc = auroc
        # Save metrics
        att_model_name = (
            f"Attention_{model_name}_{prompt}_{dataset_name}_"
            f"{num_heads}_{embed_dim}_{num_layers}_{type_num}_{response}_{early_stopping}_{upto}"
        )
        if 'fold' not in response:
            update_csv(att_model_name, acc, f1, auroc, csv_file)

            aggregated_acc.append(val_accuracy)
            aggregated_auc.append(auroc)

        if upto == early_stopping - 1:
            break
        print('Best Upto:',best_upto+1)

