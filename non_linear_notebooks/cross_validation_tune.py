from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import cross_validate
from tqdm import tqdm  # Import tqdm
#from non_linear_notebooks.model_archs.calib_tau import calib_model_train

from non_linear_notebooks.model_archs.calib_tau import calib_model_train
from sklearn.model_selection import train_test_split

def evaluate(y_true, y_pred, show=False, threshold=0.5, save_location=None, method=None):
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
    
    # Optionally display metrics and plots
    if show:
        # Print calculated metrics
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"F1-Score: {f1*100:.2f}%")
        print(f"AUROC: {auroc*100:.2f}%")
        
    
    return acc, f1, auroc 
    
class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
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
    def __init__(self, input_dim):
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

    
class AttentionModelWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    
    def __init__(self, input_dim=32000, embed_dim=512, num_heads=8, num_layers=3, dropout=0.1, epochs=100, batch_size=32, lr=1e-5, token_level=10, device=None, model_type='lr', early_stopping=True, patience=10):
        super().__init__()
        # Initialize the PyTorch model parameters
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.early_stopping = early_stopping
        self.patience = patience
        # Set device
        if device == None: 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {device}")
        self.device = device
        # Instantiate the model
        self.model_type = model_type
        if 'log_reg' in model_type:
            self.model = LogisticRegressionModel(input_dim)
        else:
            self.model = AttentionModel(input_dim, embed_dim, num_heads, num_layers, dropout)
        
        self.model.to(device)
        # Loss function and optimizer
        self.criterion = torch.nn.BCELoss()  # Assuming binary classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.token_level = token_level
        

        

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Reshape x_train and x_val
        # print(X.shape)
        x_train_full = X[:, :self.token_level, :]
        x_train_token = x_train_full.reshape(-1, x_train_full.shape[2])
        y_train_token = np.repeat(y, self.token_level)
        
        if self.early_stopping:
            # Split into train and validation
            x_train_token, x_val_token, y_train_token, y_val_token = train_test_split(
                x_train_token, y_train_token, test_size=0.2, random_state=42, stratify=y_train_token
            )
            val_dataset = BinaryDataset(x_val_token, y_val_token)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        train_dataset = BinaryDataset(x_train_token, y_train_token)
        # Create dataloaders
        batch_size = self.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # num_epochs = 50
        best_val_loss = float('inf')
        patience_counter = 0
        # Initialize model
        # input_dim = 32000 # x_train.shape[-1]  # 32000

        self.model.train()
        for epoch in tqdm(range(self.epochs), desc="Epochs", unit="epoch"):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                # inputs = inputs[:,0,:].to(device)  # Shape: (batch_size, 32000)
                inputs = inputs.to(self.device)  # Shape: (batch_size, 32000)
                labels = labels.to(self.device)  # Shape: (batch_size, 1)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs, att_weights = self.model(inputs)  # outputs: (batch_size, 1)
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                # # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
            
            if self.early_stopping:
                # Validate
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        outputs, _ = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                self.model.train()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                if epoch % 5 == 0:
                    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")

        return self

    def predict(self, X):
        # Make predictions
        X = torch.tensor(X,device=self.device)
        batch_log_sum_1 = torch.zeros(X.shape[0], device=self.device)  # Assuming preds are per sample
        batch_log_sum_2 = torch.zeros(X.shape[0], device=self.device)
        self.model.eval()
        epsilon = 1e-10
        old_preds = None
        with torch.no_grad():
            # for loc in range(upto+1):
            for loc in range(self.token_level):
                preds, _ = self.model(X[:,loc,:])
                # Mask for non-zero vectors in the inputs
                non_zero_mask = (X[:, loc, :].norm(dim=1) > epsilon)  # Check if the vector is non-zero
                # Update the log sums, applying the mask
                batch_log_sum_1 += torch.log(preds[:, 0]) * non_zero_mask
                batch_log_sum_2 += torch.log(1 - preds[:, 0]) * non_zero_mask
        
            output = batch_log_sum_1 - batch_log_sum_2

            return (output.cpu().squeeze().numpy() > 0.5).astype(int)  # Convert to binary labels

    def predict_v2(self, X, y_val=None): #Only difference from predict is it displays confidence shifts.
        # Make predictions
        # print(X.shape)
        epsilon = 1e-10
        X = torch.tensor(X,device=self.device)
        val_dataset = BinaryDataset(X, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        device = self.device
        self.model.eval()
        epsilon = 1e-10
        old_preds = None
        for upto in range(10):
            with torch.no_grad():
                val_preds = []
                val_labels = []
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    batch_log_sum_1 = torch.zeros(inputs.size(0), device=device)  # Assuming preds are per sample
                    batch_log_sum_2 = torch.zeros(inputs.size(0), device=device)
                    
                    for loc in range(upto+1):
                        preds, _ = self.model(inputs[:,loc,:])
                        # Mask for non-zero vectors in the inputs
                        non_zero_mask = (inputs[:, loc, :].norm(dim=1) > epsilon)  # Check if the vector is non-zero
                        # Update the log sums, applying the mask
                        batch_log_sum_1 += torch.log(preds[:, 0]) * non_zero_mask
                        batch_log_sum_2 += torch.log(1 - preds[:, 0]) * non_zero_mask

                    
                    outputs = batch_log_sum_1 - batch_log_sum_2
                        
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                if old_preds == None:
                    old_preds = val_preds
                else:
                    diff = torch.mean(torch.abs(torch.tensor(val_preds) - torch.tensor(old_preds)))
                    print(f'Change in confidence from Loc [{upto}] to Loc [{upto+1}]: {diff.item():.6f}')

                    old_preds = val_preds
                # Compute validation accuracy
                val_preds_binary = (np.array(val_preds) > 0).astype(int)
                val_accuracy = accuracy_score(val_labels, val_preds_binary)
                acc, f1, auroc = evaluate(np.concatenate(val_labels), val_preds, show=False)
                # print(f"Loc [{upto+1}], Val Accuracy: {val_accuracy:.4f}, AUC: {auroc:.4f}, F1: {f1:.4f}")
                print(f"Loc [{upto+1}], Val Accuracy: {val_accuracy:.4f}, AUC: {auroc:.4f}, F1: {f1:.4f}")


    def predict_proba(self, X):
        # Return probabilities (i.e., the raw output from the sigmoid)
        # Make predictions
        X = torch.tensor(X,device=self.device)
        batch_log_sum_1 = torch.zeros(X.shape[0], device=self.device)  # Assuming preds are per sample
        batch_log_sum_2 = torch.zeros(X.shape[0], device=self.device)
        self.model.eval()
        with torch.no_grad():
            for token_loc in range(self.token_level):
                X_tensor = X[:,token_loc,:]

                preds, _ = self.model(X_tensor)
                probs = preds
                batch_log_sum_1 = batch_log_sum_1 + torch.log(probs)[:,0]
                batch_log_sum_2 = batch_log_sum_2 + torch.log(1-probs)[:,0]
            
            output = batch_log_sum_1 - batch_log_sum_2

            # Apply sigmoid to get class 1 probability
            prob_class1 = torch.sigmoid(output)

            # Class 0 probability is 1 - class 1
            prob_class0 = 1 - prob_class1

            # Reshape to ensure they are 2D (shape: 20, 1) before concatenation
            prob_class0 = prob_class0.view(-1, 1)
            prob_class1 = prob_class1.view(-1, 1)

            # Concatenate to get a 2-class probability output (shape: 20, 2)
            prob = torch.cat([prob_class0, prob_class1], dim=1)
            
            # Return as a numpy array with shape (20, 2)
            return prob.cpu().numpy()
        


class AlphaModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=32000, embed_dim=512, num_heads=8, num_layers=3, dropout=0.1, epochs=100, batch_size=32, lr=1e-5, token_level=10, device=None, model_type='lr'):
        # Initialize the PyTorch model parameters
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_type = model_type
        # Set device
        if device == None: 
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"Using device: {device}")
        self.device = device
        # Instantiate the model
        self.model = AttentionModel(input_dim, embed_dim, num_heads, num_layers, dropout)
        self.model.to(device)
        # Loss function and optimizer
        self.criterion = torch.nn.BCELoss()  # Assuming binary classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.token_level = token_level
        self.cal = None
        self.config_dict = {
            'input_dim': input_dim,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'token_level': token_level,
            'device': device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'model_type':model_type
        }


    def fit(self, X, y):
        self.classes_ = np.unique(y)
        cal = calib_model_train(self.config_dict['input_dim'],
               X, y, self.config_dict, 2)  
       
        self.cal = cal

        return self

    def predict(self, X):
        # Make predictions
        X = torch.tensor(X,device=self.device)
        scores, tokens_used = self.cal.predict_with_head(X, np.zeros(len(X)))
        preds = (scores <= 0).astype(int)
        print("Scores Summary:")
        print(f"Min: {scores.min():.4f}")
        print(f"Max: {scores.max():.4f}")
        print(f"Mean: {scores.mean():.4f}")
        print(f"Median: {np.median(scores):.4f}")
        print(f"Std: {scores.std():.4f}")
        return preds
       

    def predict_proba(self, X):
        # Return probabilities (i.e., the raw output from the sigmoid)
        # Make predictions
        epsilon = 1e-10
        X = torch.tensor(X,device=self.device)
        batch_log_sum_1 = torch.zeros(X.shape[0], device=self.device)  # Assuming preds are per sample
        batch_log_sum_2 = torch.zeros(X.shape[0], device=self.device)
        self.model.eval()
        scores, tokens_used = self.cal.predict_with_head(X, np.zeros(len(X)))

        tokens_used = torch.tensor(tokens_used, device=X.device)
        max_token_len = X.shape[1]
        batch_size = X.shape[0]

        with torch.no_grad():
            for token_loc in range(max_token_len):
                X_tensor = X[:, token_loc, :]  # Shape: (batch_size, feature_dim)

                valid_token_mask = (tokens_used > token_loc).to(X_tensor.device)

                if valid_token_mask.sum() == 0:
                    break  # No more valid tokens in any sample

                preds, _ = self.cal.models_[0](X_tensor)

                # Mask out invalid samples
                non_zero_mask = (X_tensor.norm(dim=1) > epsilon) & valid_token_mask

                probs = preds
                batch_log_sum_1 = batch_log_sum_1 + torch.log(probs[:, 0]) * non_zero_mask
                batch_log_sum_2 = batch_log_sum_2 + torch.log(1 - probs[:, 0]) * non_zero_mask

            output = batch_log_sum_1 - batch_log_sum_2

            prob_class1 = torch.sigmoid(output)
            prob_class0 = 1 - prob_class1

            prob_class0 = prob_class0.view(-1, 1)
            prob_class1 = prob_class1.view(-1, 1)
            prob = torch.cat([prob_class0, prob_class1], dim=1)

            return prob.cpu().numpy()

import sys
class MTRELRWrapper(BaseEstimator, ClassifierMixin):
    """
    MTRE-Tau-LR: Multi-Token Reasoning Engine with Logistic Regression + Calibration.
    
    Two-stage pipeline:
    1. Train sklearn LogisticRegression on flattened tokens
    2. Use SequentialReliabilityCalibrator with Ray-sampled A/B/lam for early-exit
    
    Key differences from AlphaModelWrapper:
    - Uses sklearn LR (not PyTorch AttentionModel)
    - Ray samples A, B, lam and passes them as SearchConfig grids (single values)
    - LR is trained directly, not via calib_model_train
    """
    _estimator_type = "classifier"
    
    def __init__(self, input_dim=32000, token_level=10, A=None, B=None, lam=None):
        """
        Initialize MTRE-Tau-LR wrapper.
        
        Args:
            input_dim (int): Feature dimension for each token
            token_level (int): Maximum number of tokens to process
            A (float): Positive log-odds threshold (from Ray)
            B (float): Negative log-odds threshold (from Ray)
            lam (float): Exponential decay factor (from Ray)
        """
        self.input_dim = input_dim
        self.token_level = token_level
        self.A = A
        self.B = B
        self.lam = lam
        
        # sklearn LR - trained on flattened tokens
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression( #
            #max_iter=1000,
            random_state=42,
            #solver='saga',
            #n_jobs=60
        )
        
        self.cal = None  # SequentialReliabilityCalibrator
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Two-stage fitting:
        1. Train sklearn LR on flattened tokens
        2. Generate OOF per-token logs via 5-fold CV
        3. Fit SequentialReliabilityCalibrator with A/B/lam via fit_from_predictions
        
        Args:
            X (ndarray): Shape (N, token_level, feature_dim)
            y (ndarray): Shape (N,) - binary labels
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from non_linear_notebooks.model_archs.calib import SequentialReliabilityCalibrator, SearchConfig
        
        self.classes_ = np.unique(y)
        
        # STAGE 1: Train sklearn LR on flattened tokens
        X_flat = X[:, :self.token_level, :].reshape(-1, X.shape[2])
        print(X_flat.dtype)
        #sys.exit()
        y_flat = np.repeat(y, self.token_level)
        self.model.fit(X_flat, y_flat)
        
        # STAGE 2: Generate OOF per-token log-odds via 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_logs = np.zeros((len(y), self.token_level))
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train LR fold on training split
            X_train_flat = X_train[:, :self.token_level, :].reshape(-1, X.shape[2])
            y_train_flat = np.repeat(y_train, self.token_level)
            lr_fold = LogisticRegression(random_state=42) #max_iter=1000, random_state=42, solver='saga', n_jobs=60)
            lr_fold.fit(X_train_flat, y_train_flat)
            
            # Get per-token predictions on validation split
            for token_loc in range(self.token_level):
                X_val_token = X_val[:, token_loc, :]
                probs = lr_fold.predict_proba(X_val_token)[:, 1]
                # Convert to log-odds
                epsilon = 1e-10
                log_odds = np.log(probs + epsilon) - np.log(1 - probs + epsilon)
                oof_logs[val_idx, token_loc] = log_odds
        
        # STAGE 3: Fit SequentialReliabilityCalibrator with Ray's A/B/lam
        # Create SearchConfig with Ray-sampled A, B, lam as single-element grids
        # Kmax_grid ranges from 1 to token_level (at least 1 token must be read)
        search_config = SearchConfig(
            A_grid=tuple([self.A]) if self.A is not None else np.arange(0.001, 0.02, 0.001),
            B_grid=tuple([self.B]) if self.B is not None else np.arange(0.001, 0.02, 0.001),
            Kmax_grid=tuple(range(1, self.token_level + 1)),  # Range from 1 to token_level
            lam_grid=tuple([self.lam]) if self.lam is not None else (0.0, 0.1),
            metric="roc_auc",
            target_fpr=0.05
        )
        
        self.cal = SequentialReliabilityCalibrator(
            n_folds=1,
            search=search_config,
            seed=42
        )
        
        # Fit calibrator with OOF logs
        self.cal.fit_from_predictions(oof_logs, y)
        
        return self
    
    def predict(self, X):
        """
        Predict using calibrated LR model.
        
        Args:
            X (ndarray): Shape (N, token_level, feature_dim)
            
        Returns:
            ndarray: Shape (N,) - binary predictions {0, 1}
        """
        # Get per-token log-odds from LR
        logs = np.zeros((X.shape[0], self.token_level))
        
        for token_loc in range(self.token_level):
            X_token = X[:, token_loc, :]
            probs = self.model.predict_proba(X_token)[:, 1]
            epsilon = 1e-10
            log_odds = np.log(probs + epsilon) - np.log(1 - probs + epsilon)
            logs[:, token_loc] = log_odds
        
        # Use calibrator to get final scores
        scores, _ = self.cal.predict_scores(logs)
        preds = (scores <= 0).astype(int)
        
        return preds
    
    def predict_proba(self, X):
        """
        Return probability estimates.
        
        Args:
            X (ndarray): Shape (N, token_level, feature_dim)
            
        Returns:
            ndarray: Shape (N, 2) - [P(class 0), P(class 1)]
        """
        # Get per-token log-odds from LR
        logs = np.zeros((X.shape[0], self.token_level))
        
        for token_loc in range(self.token_level):
            X_token = X[:, token_loc, :]
            probs = self.model.predict_proba(X_token)[:, 1]
            epsilon = 1e-10
            log_odds = np.log(probs + epsilon) - np.log(1 - probs + epsilon)
            logs[:, token_loc] = log_odds
        
        # Use calibrator to get final scores
        scores, _ = self.cal.predict_scores(logs)
        
        # Convert scores to probabilities
        prob_class_1 = 1 / (1 + np.exp(-scores))
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict_proba(self, X):
        """
        Return probability estimates using sequential token processing.
        
        Process tokens 0 to token_level-1 sequentially:
        1. For each token, get probability from LR
        2. Accumulate log-odds
        3. Return sigmoid of accumulated log-odds as probability for class 1
        
        Args:
            X (ndarray): Shape (N, token_level, feature_dim)
            
        Returns:
            ndarray: Shape (N, 2) - [P(class 0), P(class 1)]
        """
        batch_log_sum_1 = np.zeros(X.shape[0])
        batch_log_sum_2 = np.zeros(X.shape[0])
        epsilon = 1e-10
        
        # Process tokens sequentially
        for token_loc in range(self.token_level):
            X_token = X[:, token_loc, :]  # Shape: (batch_size, feature_dim)
            
            # Get probability predictions
            probs_both = self.model.predict_proba(X_token)  # Shape: (N, 2)
            probs = probs_both[:, 1]  # P(class 1)
            
            # Accumulate log-odds
            batch_log_sum_1 += np.log(probs + epsilon)
            batch_log_sum_2 += np.log(1 - probs + epsilon)
        
        # Accumulated log-odds
        output = batch_log_sum_1 - batch_log_sum_2
        
        # Apply sigmoid
        prob_class_1 = 1 / (1 + np.exp(-output))
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])


# # # Example dataset (assuming X and y are your features and labels)
# X = np.random.randn(100, 10, 32000).astype(np.float32)  # Convert to float32 # 100 samples, 20 features
# y = np.random.randint(0, 2, size=100)  # Binary labels

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print('Running attention model..')
# # Your AttentionModel
# model = AttentionModel(input_dim=32000, embed_dim=128, num_heads=8, num_layers=3).to(device)

# # Run benchmark
# results = benchmark_model(model, input_dim=32000, device=device)

# # print(res['test_roc_auc'])
# # print(res['test_accuracy'])
# # print(res['test_f1'])

# print(f"AUROC: {np.mean(res['test_roc_auc'])*100:.2f}")
# print(f"ACC: {np.mean(res['test_accuracy'])*100:.2f}")
# print(f"F1: {np.mean(res['test_f1'])*100:.2f}")