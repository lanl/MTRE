"""
MTRE: Multi-Token Reliability Estimation Model Architecture

This module implements the core MTRE model using multi-head self-attention
to aggregate logits from the first N tokens for hallucination detection.

Key Components:
    - MultiHeadAttentionLayer: Single attention layer with residual connections
    - AttentionModel: Full MTRE model with stacked attention + classification head
    - LogisticRegressionModel: Simple baseline for comparison

Training Functions:
    - train_model(): Train MTRE with early stopping
    - eval_model(): Evaluate trained model on validation set
    - run_model_train_eval(): Main training/evaluation pipeline

Architecture:
    Input logits [batch, vocab_size]
        -> Project to embed_dim
        -> Stack of N MultiHeadAttention layers
        -> Aggregate (average pooling)
        -> FC layers with dropout
        -> Sigmoid output (reliability score)

Usage:
    model = AttentionModel(input_dim=32000, embed_dim=128, num_heads=8)
    output, attention_weights = model(logits_batch)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import logging
import csv
import re
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.metric import evaluate
from utils.func import reshape_data, create_data_loaders
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from non_linear_notebooks.model_archs.helpers import *
from non_linear_notebooks.model_archs.calib import calib_model




class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Multi-Head Attention Layer using PyTorch's nn.MultiheadAttention.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super(MultiHeadAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for the Multi-Head Attention Layer.

        Args:
            x (Tensor): Input tensor of shape (sequence_length, batch_size, embed_dim).

        Returns:
            Tensor: Output tensor of shape (sequence_length, batch_size, embed_dim).
        """
        # Self-attention: query, key, value are all x
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        # Apply dropout
        attn_output = self.dropout(attn_output)
        # Residual connection and layer normalization
        x = self.layer_norm(x + attn_output)
        return x, attn_weights
        
class AttentionModel(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        """
        Enhanced Attention Model with multiple multi-head attention layers.

        Args:
            input_dim (int): Number of input features.
            embed_dim (int): Embedding dimension for projections.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of stacked attention layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Input projection: (batch_size, input_dim) -> (batch_size, input_dim, embed_dim)
        self.input_projection = nn.Linear(input_dim, embed_dim)

        # # Optional: Positional Encoding
        # self.positional_encoding = PositionalEncoding(embed_dim)

        # Stack multiple Multi-Head Attention Layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttentionLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Aggregation layer: (batch_size, input_dim, embed_dim) -> (batch_size, embed_dim)
        self.aggregate = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.token_weights = nn.Parameter(torch.zeros(10))
        # self.sigmoid_token = nn.Sigmoid() 

    def forward(self, x):
        """
        Forward pass for the Attention Model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[Tensor, List[Tensor]]: Output predictions and list of attention weights from each layer.
        """
        # Input projection
        x = x.unsqueeze(1)
        x = self.input_projection(x)  # (batch_size, input_dim, embed_dim)

        # Prepare for MultiheadAttention: (sequence_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (input_dim, batch_size, embed_dim)

        attention_weights = []
        for attn_layer in self.attention_layers:
            x, attn = attn_layer(x)  # Each attn has shape (batch_size, num_heads, sequence_length, sequence_length)
            attention_weights.append(attn)

        # Permute back to (batch_size, input_dim, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size, input_dim, embed_dim)

        # Aggregate features (e.g., average pooling)
        x = self.aggregate(x.transpose(1, 2))  # (batch_size, embed_dim, 1)
        x = x.squeeze(2)  # (batch_size, embed_dim)

        # Fully Connected Layers
        x = self.fc1(x)  # (batch_size, 128)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)  # (batch_size, 64)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)  # (batch_size, 1)
        x = self.sigmoid(x)  # (batch_size, 1)
        

        return x, attention_weights

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        """
        True Logistic Regression Model.

        Args:
            input_dim (int): Number of input features.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # weights + bias
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for logistic regression.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output probabilities of shape (batch_size, 1).
        """
        logits = self.linear(x)          # (batch_size, 1)
        probs = self.sigmoid(logits)     # (batch_size, 1)
        return probs, logits
# # ------------------------- Small utilities -------------------------
# class MLPModel(nn.Module):
#     def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=3, dropout=0.1):
#         """
#         Simple Logistic Regression Model with optional hidden layers.

#         Args:
#             input_dim (int): Number of input features.
#             hidden_dim (int): Dimension of hidden layer.
#             dropout (float): Dropout probability.
#         """
#         super().__init__()
#         self.input_dim = input_dim
        
#         # First fully connected layer
#         self.fc1 = nn.Linear(input_dim, embed_dim)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
        
#         # Second fully connected layer
#         self.fc2 = nn.Linear(embed_dim, embed_dim // 2)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
        
#         # Output layer (logistic regression)
#         self.output_layer = nn.Linear(embed_dim // 2, 1)
#         self.sigmoid = nn.Sigmoid()
#         self.token_weights = nn.Parameter(torch.zeros(10))
        
#     def forward(self, x):
#         """
#         Forward pass for the Logistic Regression Model.

#         Args:
#             x (Tensor): Input tensor of shape (batch_size, input_dim).

#         Returns:
#             Tensor: Output predictions of shape (batch_size, 1).
#         """
#         # # First fully connected layer with ReLU activation and dropout
#         # x = self.fc1(x)  # (batch_size, hidden_dim)
#         # x = self.relu1(x)
#         # x = self.dropout1(x)
        
#         # Second fully connected layer with ReLU activation and dropout
#         # x = self.fc2(x)  # (batch_size, hidden_dim // 2)
#         # x = self.relu2(x)
#         # x = self.dropout2(x)
        
#         # Output layer with sigmoid activation for binary classification
#         x = self.output_layer(x)  # (batch_size, 1)
#         x = self.sigmoid(x)  # (batch_size, 1)
        
#         return x, 0



def run_model_train_eval(
    input_dim, train_loader, val_loader,
    x_val, y_val, x_train, y_train,
    model_name, dataset_name, prompt, csv_file,
    config_list, device='cuda', early_stopping=10,
    trained_model_path=None, type_num='', response='', tau=False
):
    batch_size = 128

    def load_trained_model(path):
        """Load model and metadata from saved path."""
        logging.info(f'Using Model Path: {path}')
        params = parse_model_params(path)
        logging.info(f'Config Dict: {params}')

        model = AttentionModel(input_dim, **{
            'embed_dim': params['embed_dim'],
            'num_heads': params['num_heads'],
            'num_layers': params['num_layers'],
            'dropout': 0.0
        }).to(device)
        model.load_state_dict(torch.load(path))

        props = parse_filename_information(path)
        return model, params, props

    if trained_model_path:
        model, params, props = load_trained_model(trained_model_path)

        # Update properties from filename
        model_name = props['model_name']
        prompt = props['prompt']
        dataset_name = props['dataset']
        type_num = props['type_num']
        response = props['response']

        eval_model(
            x_val, y_val, model, batch_size, device,
            model_name, dataset_name, prompt, type_num, response,
            csv_file, early_stopping,
            params['embed_dim'], params['num_heads'], params['num_layers']
        )

    else:
        for config_dict in config_list:
            if tau == True:
                calib_model(input_dim, train_loader, val_loader,
                    x_val, y_val, x_train, y_train, config_dict, csv_file, 2)
                # calib_model(input_dim, train_loader, val_loader,
                #  x_val, y_val, x_train, y_train, config_dict, csv_file, 4) #More folds if wanted.
            else:
                model = train_model(
                    config_dict, train_loader, device,
                    x_val, y_val, x_train, y_train,
                    early_stopping, input_dim, model_name, dataset_name,
                    prompt, batch_size, val_loader, type_num, response, csv_file
                )

                logging.info('Evaluating trained model')

                eval_model(
                    x_val, y_val, model, batch_size, device,
                    model_name, dataset_name, prompt, type_num, response,
                    csv_file, early_stopping,
                    config_dict['embed_dim'], config_dict['num_heads'], config_dict['num_layers']
                )
            
def train_model(
    config_dict, train_loader, device,
    x_val, y_val, x_train, y_train, early_stopping,
    input_dim, model_name, dataset_name, prompt,
    batch_size, val_loader, type_num, response, csv_file, weights=None, alpha=None
):
    # Extract hyperparameters
    patience = 30
    min_delta = 0.0005
    epochs = 1000
    
    # Initialize model
    if 'log_reg' in config_dict and config_dict['log_reg']:
        model = LogisticRegressionModel(input_dim, **{k: config_dict[k] for k in ['embed_dim', 'num_heads', 'num_layers', 'dropout']}).to(device)
    else:
        model = AttentionModel(input_dim, **{k: config_dict[k] for k in ['embed_dim', 'num_heads', 'num_layers', 'dropout']}).to(device)
    
    # Optimizer & loss
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config_dict['lr'])

    # Tracking best metrics
    best_model = None
    best_loss = float('inf')
    best_val_accuracy = 0.0
    best_val_auroc = 0.0
    epochs_no_improve = 0

    def filter_non_zero(inputs, labels):
        """Mask out zero vectors."""
        mask = inputs.sum(dim=1) != 0
        return inputs[mask], labels[mask]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = filter_non_zero(inputs, labels)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)

        # Early stopping check
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

        # Validation phase
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = filter_non_zero(inputs, labels)

                outputs, _ = model(inputs)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_preds_binary = (val_preds > 0.5).astype(int)

        val_accuracy = accuracy_score(val_labels, val_preds_binary)
        _, _, val_auroc = evaluate(val_labels, val_preds, show=False)
        
        logging.info(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val AUC: {val_auroc:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy, best_val_auroc = val_accuracy, val_auroc
            best_model = copy.deepcopy(model)
            epochs_no_improve = 0

            logging.info(f"Saved Best Model | Val Acc: {best_val_accuracy:.4f}, Val AUC: {best_val_auroc:.4f}")

        # Periodic evaluation
        if epoch % 50 == 0:
            for eval_name, eval_fn in [
                # ("Train", eval_model),
                ("Val: MTRE", eval_model)
            ]:
                logging.info(f"Evaluating {eval_name}")
                eval_fn(x_val if "Val" in eval_name else x_train,
                        y_val if "Val" in eval_name else y_train,
                        best_model, batch_size, device, model_name, dataset_name,
                        prompt if "Val" in eval_name else 'train',
                        type_num, response, csv_file, early_stopping,
                        config_dict['embed_dim'], config_dict['num_heads'], config_dict['num_layers'])
            

    return best_model


