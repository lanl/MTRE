import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.metric import evaluate
from utils.func import reshape_data, create_data_loaders
from non_linear_notebooks.model_archs.helpers import BinaryDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import logging
import os

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
class MLPModel(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=3, dropout=0.1):
        """
        Simple Logistic Regression Model with optional hidden layers.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of hidden layer.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        
        # First fully connected layer
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(embed_dim, embed_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer (logistic regression)
        self.output_layer = nn.Linear(embed_dim // 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.token_weights = nn.Parameter(torch.zeros(10))
        
    def forward(self, x):
        """
        Forward pass for the Logistic Regression Model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output predictions of shape (batch_size, 1).
        """
        # # First fully connected layer with ReLU activation and dropout
        x = self.fc1(x)  # (batch_size, hidden_dim)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Second fully connected layer with ReLU activation and dropout
        x = self.fc2(x)  # (batch_size, hidden_dim // 2)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Output layer with sigmoid activation for binary classification
        x = self.output_layer(x)  # (batch_size, 1)
        x = self.sigmoid(x)  # (batch_size, 1)
        
        return x, 0

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.array(x)

def logit_from_prob(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------- Metrics -------------------------

def roc_auc(scores: np.ndarray, y: np.ndarray) -> float:
    # Simple, dependency-free ROC AUC
    # Sort by score descending
    order = np.argsort(-scores)
    y = y[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Mann–Whitney U = sum of ranks of positives - n_pos*(n_pos+1)/2
    ranks = np.arange(1, len(y) + 1)
    pos_ranks_sum = ranks[y == 1].sum()
    U = pos_ranks_sum - n_pos * (n_pos + 1) / 2.0
    return U / (n_pos * n_neg)

def pr_auc(scores: np.ndarray, y: np.ndarray) -> float:
    # PR curve via sorted thresholds
    order = np.argsort(-scores)
    y = y[order]
    tp = 0
    fp = 0
    P = y.sum()
    if P == 0:
        return float("nan")
    precisions = []
    recalls = []
    last_score = None
    for i, yi in enumerate(y):
        if yi == 1:
            tp += 1
        else:
            fp += 1
        # record only when score changes or at last point
        if i == len(y) - 1 or scores[order[i+1]] != scores[order[i]]:
            precision = tp / (tp + fp)
            recall = tp / P
            precisions.append(precision)
            recalls.append(recall)
    # Integrate PR curve (trapezoidal in recall space)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    # Ensure starting at recall=0 with precision=P/(P+N)
    recalls = np.concatenate([[0.0], recalls])
    precisions = np.concatenate([[precisions[0]], precisions])
    area = 0.0
    for i in range(1, len(recalls)):
        area += (recalls[i] - recalls[i-1]) * precisions[i]
    return area

def f1_at_fpr(scores: np.ndarray, y: np.ndarray, target_fpr: float) -> Tuple[float, float, float]:
    # Choose threshold achieving FPR <= target_fpr, maximize F1.
    order = np.argsort(-scores)
    y_sorted = y[order]
    scores_sorted = scores[order]
    P = y_sorted.sum()
    N = len(y_sorted) - P
    tp = 0
    fp = 0
    best_f1 = -1.0
    best_th = None
    # scan thresholds at unique score levels
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / max(N, 1)
        if fpr <= target_fpr:
            prec = tp / max(tp + fp, 1)
            rec = tp / max(P, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            # threshold is current score
            th = scores_sorted[i]
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
    return best_f1 if best_f1 >= 0 else float("nan"), best_th if best_th is not None else float("nan"), target_fpr


# ------------------------- Early-exit aggregator -------------------------

def sequential_sum_llr(z: torch.Tensor, A: float, B: float, Kmax: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    z: [N, K] calibrated per-token log-odds
    Returns:
      final_score: [N] accumulated LLR at stopping or Kmax
      tokens_used: [N] number of tokens consumed
    """
    N, K = z.shape
    if Kmax is None or Kmax > K:
        Kmax = K
    L = torch.zeros(N, device=z.device)
    decided = torch.zeros(N, dtype=torch.bool, device=z.device)
    final = torch.zeros(N, device=z.device)
    used = torch.zeros(N, dtype=torch.long, device=z.device)
    for k in range(Kmax):
        L = L + z[:, k]
        pos = (L >= A) & ~decided
        neg = (L <= -B) & ~decided
        anyd = pos | neg
        final[anyd] = L[anyd]
        used[anyd] = k + 1
        decided = decided | anyd
    undec = ~decided
    final[undec] = L[undec]
    used[undec] = Kmax
    return final, used

def apply_decay(z: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Exponential decay over token index: w_l ∝ exp(-lam * (l-1))
    Returns weighted per-token log-odds: [N, K]
    """
    N, K = z.shape
    w = torch.exp(-lam * torch.arange(K, device=z.device, dtype=z.dtype))
    w = w / w.sum()
    return z * w


# ------------------------- Temperature scaling -------------------------

class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.T = nn.Parameter(torch.tensor([init_T], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 500, lr: float = 0.05):
        """
        Fit temperature on per-token logits (log-odds) by minimizing BCE.
        logits: [N, K]
        labels: [N] or [N, K] (broadcastable to logits)
        """
        self.train()
        opt = optim.LBFGS([self.T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
        bce = nn.BCEWithLogitsLoss(reduction='mean')
        y = labels.float().unsqueeze(1).expand_as(logits)

        def closure():
            opt.zero_grad()
            loss = bce(self.forward(logits), y)
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.T.detach().cpu().item())


# ------------------------- Config and main class -------------------------

@dataclass
class SearchConfig:
    A_grid: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
    B_grid: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
    Kmax_grid: Tuple[int, ...] = (5, 8, 10)
    lam_grid: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)  # 0.0 means no decay effect
    metric: str = "roc_auc"  # 'pr_auc' | 'roc_auc' | 'f1_at_fpr'
    target_fpr: float = 0.05

class SequentialReliabilityCalibrator:
    """
    Orchestrates:
      - Cross-fitting on Train (200) to get out-of-fold per-token predictions
      - Temperature scaling (on out-of-fold logits)
      - Grid search for (A, B, Kmax, lam) on out-of-fold preds
      - Final retrain on all Train and inference on Test with frozen params
    """

    def __init__(
        self,
        fit_head_fn: Optional[Callable[[object, Sequence[int], int], object]] = None,
        predict_head_fn: Optional[Callable[[object, object, Sequence[int]], np.ndarray]] = None,
        n_folds: int = 2,
        n_repeats: int = 1,
        bag_models: int = 1,
        device: str = "cpu",
        search: SearchConfig = SearchConfig(),
        seed: int = 42,
        config_dict: dict = {}
    ):
        """
        If you already have per-token probabilities/logits, you can skip fit_head_fn/predict_head_fn
        and call fit_from_predictions(...).
        """
        self.fit_head_fn = fit_head_fn
        self.predict_head_fn = predict_head_fn
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.bag_models = bag_models
        self.device = device
        self.search = search
        self.seed = seed
        self.config_dict = config_dict

        # Learned parameters
        self.temperature_: float = 1.0
        self.A_: float = 2.0
        self.B_: float = 2.0
        self.Kmax_: int = 8
        self.lam_: float = 0.0

        # Final heads (bag)
        self.models_: List[object] = []

    # --------------- core helpers ---------------

    @staticmethod
    def _evaluate_scores(scores: np.ndarray, y: np.ndarray, metric: str, target_fpr: float) -> float:
        if metric == "roc_auc":
            return roc_auc(scores, y)
        elif metric == "pr_auc":
            return pr_auc(scores, y)
        elif metric == "f1_at_fpr":
            f1, _, _ = f1_at_fpr(scores, y, target_fpr)
            return f1
        else:
            raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def _bootstrap_ci(scores: np.ndarray, y: np.ndarray, metric: str, target_fpr: float, B: int = 1000, seed: int = 0) -> Tuple[float, Tuple[float, float]]:
        rng = np.random.default_rng(seed)
        n = len(y)
        stats = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            stats.append(SequentialReliabilityCalibrator._evaluate_scores(scores[idx], y[idx], metric, target_fpr))
        stats = np.array(stats)
        return float(np.nanmean(stats)), (float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5)))

    # --------------- public API ---------------
    def fit_with_head(self, X, y):
        """
        Cross-fit head on the training loader's dataset to get OOF predictions for calibration,
        tune parameters, then retrain a bag of heads on all data.
        """
        assert self.fit_head_fn is not None and self.predict_head_fn is not None, \
            "fit_head_fn and predict_head_fn must be provided."

        # Access the underlying dataset
        # base_dataset = train_loader.dataset
        n_train = len(X)
        train_idx = list(range(n_train))

        all_oof_probs = []
        for rep in range(self.n_repeats):
            # Stratified split
            pos_idx = [i for i in train_idx if y[i] == 1]
            neg_idx = [i for i in train_idx if y[i] == 0]
            random.shuffle(pos_idx)
            random.shuffle(neg_idx)
            pos_folds = np.array_split(pos_idx, self.n_folds)
            neg_folds = np.array_split(neg_idx, self.n_folds)

            oof_probs = np.zeros((n_train, self.config_dict['token_level']), dtype=np.float32)

            for fold in range(self.n_folds):
                logging.info(f'Training on fold: {fold}')
                val_idx = np.concatenate([pos_folds[fold], neg_folds[fold]])
                tr_idx = [i for i in train_idx if i not in val_idx]

                # Split fold data
                x_train_fold, x_val_fold = X[tr_idx], X[val_idx]
                y_train_fold, y_val_fold = y[tr_idx], y[val_idx]
                x_train_tok, y_train_tok, x_val_tok, y_val_tok = reshape_data(
                    x_train_fold, y_train_fold, x_val_fold, y_val_fold, self.config_dict['token_level']
                )
                train_loader, val_loader = create_data_loaders(
                    x_train_tok, y_train_tok, x_val_tok, y_val_tok,
                    batch_size=512, shuffle_train=True
                )
                
                model = self.fit_head_fn(train_loader, self.config_dict, seed=self.seed + rep * 100 + fold)
                probs_val = self.predict_head_fn(model, x_val_fold, y_val_fold, self.config_dict)
                oof_probs[val_idx, :] = probs_val

            all_oof_probs.append(oof_probs)

        # Average repeated OOF predictions
        oof_probs = np.mean(all_oof_probs, axis=0)

        # Fit calibration parameters
        self.fit_from_predictions(oof_probs, y)

        # Retrain heads on full dataset with bagging
        x_train_tok, y_train_tok, x_val_tok, y_val_tok = reshape_data(
                    X, y, X, y, self.config_dict['token_level']
        )
        train_loader, val_loader = create_data_loaders(
            x_train_tok, y_train_tok, x_val_tok, y_val_tok,
                    batch_size=512, shuffle_train=True
        )
        self.models_ = [
            self.fit_head_fn(train_loader, self.config_dict, seed=self.seed + 999 + b)
            for b in range(self.bag_models)
        ]

        return self

    def fit_from_predictions(self, oof_logs: np.ndarray, y_train: np.ndarray):
        """
        Use when you already have out-of-fold per-token logs for the Train set.
        Tunes temperature and early-exit/decay parameters.
        """
        # Temperature scaling on per-token logits
        with torch.no_grad():
            z_raw = -torch.tensor(oof_logs, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)

        temp = TemperatureScaler(init_T=1.0)
        self.temperature_ = temp.fit(z_raw, y_t)

        # Calibrated logits
        z_cal = z_raw / self.temperature_

        # Grid search for (A, B, Kmax, lam)
        best_score = -float("inf")
        best_tuple = None

        for A in self.search.A_grid:
            for B in self.search.B_grid:
                for Kmax in self.search.Kmax_grid:
                    for lam in self.search.lam_grid:
                        z_use = z_cal
                        if lam > 0:
                            z_use = apply_decay(z_use, lam)
                        scores_t, _ = sequential_sum_llr(z_use, A=A, B=B, Kmax=Kmax)
                        scores = to_numpy(scores_t)
                        s = self._evaluate_scores(scores, y_train, self.search.metric, self.search.target_fpr)
                        if np.isnan(s):
                            continue
                        if s > best_score:
                            best_score = s
                            best_tuple = (A, B, Kmax, lam)
                            logging.info(f'Best Score: {s}')
                            logging.info(f'Best tuple: {best_tuple}')
                            print(f'Best Score: {s}')
                            print(f'Best tuple: {best_tuple}')


        if best_tuple is None:
            raise RuntimeError("Grid search failed (metric undefined). Check label balance and Kmax grid.")

        self.A_, self.B_, self.Kmax_, self.lam_ = best_tuple
        return self

    def predict_scores(self, logs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert per-token logs to final sequential scores with the learned parameters.
        Returns:
          scores: [N] signed evidence (sum of calibrated log-odds at stop)
          tokens_used: [N]
        """
        with torch.no_grad():
            z = torch.tensor(logs, dtype=torch.float32)
            z = -z / float(self.temperature_) # Figure out why it does better when negative
            if self.lam_ > 0:
                z = apply_decay(z, self.lam_)
            scores_t, used_t = sequential_sum_llr(z, A=self.A_, B=self.B_, Kmax=self.Kmax_)
        return to_numpy(scores_t), to_numpy(used_t)

    def predict_with_head(self, x_val, y_val) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses the bagged final models to produce averaged per-token probabilities and then
        applies the calibrated early-exit to return final scores and tokens_used.
        """
        assert self.models_, "No trained models found. Call fit_with_head(...) first."
        # Average probabilities from bag
        bag_probs = []
        for m in self.models_:
            logs = self.predict_head_fn(m, x_val, y_val)  # [N, K], logs
            bag_probs.append(logs)
        logs = np.mean(np.stack(bag_probs, axis=0), axis=0)
        return self.predict_scores(logs)
    
    def predict_with_head_proba(self, x_val, y_val) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses the bagged final models to produce averaged per-token logs and then
        applies the calibrated early-exit to return final scores and tokens_used.
        """
        assert self.models_, "No trained models found. Call fit_with_head(...) first."
        # Average probabilities from bag
        bag_probs = []
        for m in self.models_:
            probs = self.predict_head_fn(m, x_val, y_val)  # [N, K], probs
            bag_probs.append(probs)
        probs = np.mean(np.stack(bag_probs, axis=0), axis=0)
        return probs

    def evaluate(self, scores: np.ndarray, y: np.ndarray, with_ci: bool = True, B: int = 1000, seed: int = 0) -> Dict[str, float]:
        logging.info(f"Scores passed to evaluate: {scores[:20]}")
        out = {}
        preds = (scores <= 0).astype(int)
        acc = accuracy_score(y, preds)
        out["accuracy"] = acc
        out["pr_auc"] = pr_auc(scores, y)
        f1, th, fpr = f1_at_fpr(scores, y, self.search.target_fpr)
        # out[f"f1_at_fpr_{self.search.target_fpr:.3f}"] = f1
        # out[f"threshold_at_fpr_{self.search.target_fpr:.3f}"] = th
        out["f1_score"] = f1_score(y, preds)
        out["roc_auc"] = roc_auc(scores, y)

        logging.info("Scores Summary:")
        logging.info(f"Min: {scores.min():.4f}")
        logging.info(f"Max: {scores.max():.4f}")
        logging.info(f"Mean: {scores.mean():.4f}")
        logging.info(f"Median: {np.median(scores):.4f}")
        logging.info(f"Std: {scores.std():.4f}")

        return out

#How to use it (end-to-end with your head)
def calib_model(input_dim, train_loader, val_loader,
                x_val, y_val, x_train, y_train, config_dict, csv_file, folds=2):
    # You implement these two functions for YOUR model and data.
    def fit_head_fn(loader, config_dict, seed=None):
        # Extract hyperparameters
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        patience = 30
        min_delta = 0.0005
        epochs = 1000
        input_dim = 32000
        
        # Initialize model
        if 'log_reg' in config_dict and config_dict['log_reg']:
            model = LogisticRegressionModel(input_dim, **{k: config_dict[k] for k in ['embed_dim', 'num_heads', 'num_layers', 'dropout']}).to('cuda')
        else:
            model = AttentionModel(input_dim, **{k: config_dict[k] for k in ['embed_dim', 'num_heads', 'num_layers', 'dropout']}).to('cuda') # Can be Logistic Regression as well.

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

        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0.0
            
            for inputs, labels in loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
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
                    break

        return model

    def predict_head_fn(model, x_val, y_val, config_dict=None) -> np.ndarray:
            """
            Run the trained head on dataset[idx] and return per-token logs of shape [N, K].
            """
            epsilon = 1e-10
            batch_size = 128
            val_loader = DataLoader(BinaryDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
            best_auc = 0
            best_upto = 0

            if config_dict:
                token_level = config_dict.get('token_level') if config_dict.get('token_level') else 10
            
            def forward_with_mask(inputs, upto):
                """Forward pass accumulating predictions up to a given location."""
                batch_log_sum_1 = torch.zeros(inputs.size(0), device='cuda')
                batch_log_sum_2 = torch.zeros(inputs.size(0), device='cuda')
                
                # for loc in range(upto + 1):
                preds, _ = model(inputs[:, upto, :])
                non_zero_mask = (inputs[:, upto, :].norm(dim=1) > epsilon)
                batch_log_sum_1 += torch.log(preds[:, 0]) * non_zero_mask
                batch_log_sum_2 += torch.log(1 - preds[:, 0]) * non_zero_mask
                
                return batch_log_sum_1 - batch_log_sum_2

            old_preds, old_preds_binary = None, None
            aggregated_acc, aggregated_auc = [], []

            model.eval()
            total_preds = []
            for upto in range(token_level):
                val_preds, val_labels = [], []

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        outputs = forward_with_mask(inputs, upto)
                        val_preds.extend(outputs.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())

                val_preds = np.array(val_preds)
                val_labels = np.array(val_labels)

                total_preds.append(np.array(val_preds))

            # Evaluate model in no-grad mode, stack outputs into [N, K]
            return np.array(total_preds).T  # float32 numpy array, values in [0,1]

    cal = SequentialReliabilityCalibrator(
        fit_head_fn=fit_head_fn,
        predict_head_fn=predict_head_fn,
        n_folds=folds,
        n_repeats=1,
        bag_models=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        search=SearchConfig(
            A_grid=np.arange(0.001, 0.1, 0.001),
            B_grid=np.arange(0.001, 0.1, 0.001),
            Kmax_grid=list(range(1, config_dict['token_level'], 1)),
            lam_grid=np.arange(0.0, .1, .1),
            metric="roc_auc",   # or try "roc_auc" if labels are well-balanced
            target_fpr=0.05       # or try [0.01, 0.10] for robustness
        ),
        seed=42,
    )
    cal.config_dict = config_dict

    # 1) Fit (cross-fit on Train, tune T/A/B/Kmax/lam, bag final heads)
    cal.fit_with_head(x_train, y_train)

    # 2) Inference on the blind Test set
    scores, tokens_used = cal.predict_with_head(x_val, y_val)

    # 3) Evaluate (if you have labels for the test set)
    report = cal.evaluate(scores, y_val, with_ci=True, B=1000)
    logging.info(f'FINAL REPORT: {report}')
     # Save to CSV
    import os 
    df = pd.DataFrame([report])  # Make it a list of dicts to create a DataFrame with one row
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)



#How to use it (end-to-end with your head)
def calib_model_train(input_dim, x_train, y_train, config_dict, folds=2):
    def fit_head_fn(loader, config_dict, seed=None):
        # Extract hyperparameters
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        patience = 30
        min_delta = 0.0005
        epochs = config_dict['epochs'] #1000
        input_dim = config_dict['input_dim'] #32000
        
        # Initialize model
        if 'lr' in config_dict['model_type']:
            model = LogisticRegressionModel(input_dim, **{k: config_dict[k] for k in ['embed_dim', 'num_heads', 'num_layers', 'dropout']}).to('cuda')
        else:
            print('Running attention')
            model = AttentionModel(input_dim, **{k: config_dict[k] for k in ['embed_dim', 'num_heads', 'num_layers', 'dropout']}).to('cuda')

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

        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0.0
            
            for inputs, labels in loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                inputs, labels = filter_non_zero(inputs, labels)

                optimizer.zero_grad()
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)

            epoch_loss = total_loss / len(loader.dataset)

            # Early stopping check
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        return model

    def predict_head_fn(model, x_val, y_val, config_dict=None) -> np.ndarray:
        """
        Run the trained head on dataset[idx] and return per-token logs of shape [N, K].
        """
        epsilon = 1e-10
        batch_size = 128
        val_loader = DataLoader(BinaryDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
        best_auc = 0
        best_upto = 0

        # Extract token_level from config_dict, defaulting to 10
        token_level = 10
        if config_dict:
            token_level = config_dict.get('token_level') if config_dict.get('token_level') else 10
        
        def forward_with_mask(inputs, upto):
            """Forward pass accumulating predictions up to a given location."""
            batch_log_sum_1 = torch.zeros(inputs.size(0), device='cuda')
            batch_log_sum_2 = torch.zeros(inputs.size(0), device='cuda')
            
            # for loc in range(upto + 1):
            preds, _ = model(inputs[:, upto, :])
            non_zero_mask = (inputs[:, upto, :].norm(dim=1) > epsilon)
            batch_log_sum_1 += torch.log(preds[:, 0]) * non_zero_mask
            batch_log_sum_2 += torch.log(1 - preds[:, 0]) * non_zero_mask
            
            return batch_log_sum_1 - batch_log_sum_2

        old_preds, old_preds_binary = None, None
        aggregated_acc, aggregated_auc = [], []

        model.eval()
        total_preds = []
        
        # Use actual input dimension (number of tokens), not config token_level
        # This handles cases where X was sliced to fewer tokens
        actual_token_count = x_val.shape[1] if len(x_val.shape) > 1 else token_level
        
        for upto in range(actual_token_count):
            val_preds, val_labels = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    outputs = forward_with_mask(inputs, upto)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_preds = np.array(val_preds)
            val_labels = np.array(val_labels)

            total_preds.append(np.array(val_preds))

        # Evaluate model in no-grad mode, stack outputs into [N, K]
        return np.array(total_preds).T  # float32 numpy array, values in [0,1]
    if config_dict.get('token_level'):
        print('Token Level: ', config_dict['token_level'])
    else:
        print('Token Level not set defaulting to 10.')

    # Check if Ray has provided specific calibrator parameters, otherwise use grid search
    use_ray_params = all(k in config_dict for k in ['A', 'B', 'lam'])
    
    if use_ray_params:
        # Ray-tuned calibrator parameters: skip grid search, use provided values directly
        search = SearchConfig(
            A_grid=(config_dict['A'],),  # Single value (no grid search)
            B_grid=(config_dict['B'],),  # Single value (no grid search)
            Kmax_grid=(config_dict.get('Kmax', config_dict.get('token_level', 10)),),  # Single value
            lam_grid=(config_dict['lam'],),  # Single value (no grid search)
            metric="roc_auc",
            target_fpr=0.05
        )
    else:
        # Fallback: grid search for all combinations (original behavior)
        search = SearchConfig(
            A_grid=np.arange(0.001, .02, 0.001),
            B_grid=np.arange(0.001, .02, 0.001),
            Kmax_grid=list(range(1, config_dict.get('token_level') if config_dict.get('token_level') else 10, 1)),
            lam_grid=np.arange(0.0, .1, .1),
            metric="roc_auc",
            target_fpr=0.05
        )
    
    cal = SequentialReliabilityCalibrator(
        fit_head_fn=fit_head_fn,
        predict_head_fn=predict_head_fn,
        n_folds=folds,
        n_repeats=1,
        bag_models=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        search=search,
        seed=42,
    )
    cal.config_dict = config_dict

    # 1) Fit (cross-fit on Train, tune T/A/B/Kmax/lam, bag final heads)
    cal.fit_with_head(x_train, y_train)
    return cal


