import os
import sys
import atexit

# Completely block tensorboardX from being imported
# This forces Ray to treat it as not installed, preventing any thread creation
sys.modules['tensorboardX'] = None

# Disable TensorBoard env vars as backup
os.environ['RAY_AIR_DISABLE_TENSORBOARD_PLUGIN'] = '1'
os.environ['TENSORBOARD_PLUGIN_DIRNAMES'] = ''
os.environ['RAY_DISABLE_MEMORY_MONITOR'] = '1'
os.environ['RAY_TUNE_DISABLE_TENSORBOARD'] = '1'
os.environ['RAY_memory_monitor_refresh_ms'] = '0'
os.environ['RAY_memory_usage_threshold'] = '1.0'

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import json
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress verbose logging from libraries
logging.getLogger('ray').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
#from utils.func import read_jsonl
#from utils.parse import parse_multi_choice_response, parse_open_response, eval_multi_choice, eval_open
#from non_linear_notebooks.cross_validation_tune import AttentionModelWrapper, AlphaModelWrapper
import csv
import pickle
import torch
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
#from non_linear_notebooks.model_archs.calib import calib_model
import re
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def coerce_binary_labels(y_raw):
    """
    Accepts labels that might be {0,1}, {False,True}, {-1,1}, strings, or NaN.
    Returns float32 array in {0.,1.}, dropping invalid rows and a mask.
    """
    y = np.asarray(y_raw)

    # map common variants to {0,1}
    # - strings like "0","1","True","False"
    if y.dtype.kind in ("U", "S", "O"):
        def to_num(v):
            if v is None: return np.nan
            s = str(v).strip().lower()
            if s in {"1","true","yes","y"}:  return 1.0
            if s in {"0","false","no","n"}:  return 0.0
            try:  return float(s)
            except: return np.nan
            y = np.vectorize(to_num, otypes=[float])(y)

    # convert {-1,1} → {0,1}
    y = np.where(y == -1, 0.0, y)
    # cast to float
    y = y.astype(np.float32, copy=False)

    # valid in [0,1] and finite
    mask = np.isfinite(y) & (y >= 0.0) & (y <= 1.0)
    return y[mask], mask

def apply_mask_to_X(X, mask):
    if isinstance(X, np.ndarray):
        return X[mask]
    # handle dicts like {"x": ..., "x_first": ...}
    if isinstance(X, dict):
        return {k: v[mask] for k, v in X.items()}
    return X

from tqdm import tqdm
import numpy as np
import os

def _load_first_k_from_ins(ins, ref_key, inline_key, k=10):
    """
    Returns a (k, V) float32 array from logits pointer or inline logits.
    Squeezes (T,1,V)->(T,V) and pads to k rows if needed.
    Returns None if neither path is available.
    """
    arr = None
    ref = ins.get(ref_key)
    if ref and isinstance(ref, dict) and "npy" in ref:
        npy = ref["npy"]
        if os.path.exists(npy):
            try:
                mm = np.load(npy, mmap_mode="r")  # (T,V) or (T,1,V)
                total = min(k, mm.shape[0])
                data = []
                for i in tqdm(range(total), desc=f"Loading {ref_key}", ncols=80):
                    data.append(mm[i])
                arr = np.array(data)
            except Exception as e:
                print(f"Error loading {npy}: {e}")
                arr = None

    if arr is None and inline_key in ins and ins[inline_key] is not None:
        print(f"Loading inline logits for {inline_key}...")
        arr = np.asarray(ins[inline_key])[:k]

    if arr is None:
        return None

    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    t, V = arr.shape
    if t < k:
        pad = np.zeros((k - t, V), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return arr.astype(np.float32, copy=False)

from tqdm.auto import tqdm
from glob import glob
import os, json, numpy as np

def build_dataset_from_all_judged(
    judged_dir: str,
    k_first_steps: int = 10,
    max_files: int | None = None,
    max_rows_per_file: int | None = None,
    task: str = 'mmmu',
    show_progress: bool = True,
):
    """
    Aggregate a dataset from every '*.jsonl_judged.jsonl' file in judged_dir.
    Adds progress bars over files and per-file bytes read when show_progress=True.
    Loads ONLY one type: 'logits' OR 'logits_se', depending on whether '2' is in task.
    """
    if 'mme' in task:
        pattern = os.path.join(judged_dir, "mme.jsonl_judged.jsonl")
    else:
        pattern = os.path.join(judged_dir, "*_validation.jsonl_judged.jsonl")

    files = sorted(glob(pattern))
    if max_files is not None:
        files = files[:max_files]

    # Determine which type to load
    load_se = "2" in task
    ref_key = "logits_ref_se" if load_se else "logits_ref"
    inline_key = "logits_se" if load_se else "logits"

    X_list, Y_list = [], []
    counts_by_file = {}

    file_iter = tqdm(files, desc="Files", total=len(files), ncols=90) if show_progress else files

    for fp in file_iter:
        n_kept = 0
        bytes_bar = None
        if show_progress:
            try:
                total_bytes = os.path.getsize(fp)
                bytes_bar = tqdm(total=total_bytes, unit="B", unit_scale=True,
                                desc=os.path.basename(fp), ncols=90, leave=False)
            except OSError:
                bytes_bar = tqdm(desc=os.path.basename(fp), ncols=90, leave=False)

        with open(fp, "r") as f:
            for i, raw in enumerate(f):
                if max_rows_per_file is not None and i >= max_rows_per_file:
                    break

                if bytes_bar is not None:
                    try:
                        bytes_bar.update(len(raw.encode(errors="ignore")))
                    except Exception:
                        pass

                line = raw.strip()
                if not line:
                    continue
                ins = json.loads(line)

                # load the selected type only
                x = _load_first_k_from_ins(ins, ref_key=ref_key, inline_key=inline_key, k=k_first_steps)
                if x is None:
                    continue

                y_key = "y_type2" if load_se else "y_type1"
                y = int(ins.get(y_key, 0))

                X_list.append(x)
                Y_list.append(y)
                n_kept += 1

        if bytes_bar is not None:
            bytes_bar.close()

        counts_by_file[os.path.basename(fp)] = n_kept

    if not X_list:
        raise RuntimeError(f"No usable judged validation samples found in {judged_dir}")

    X = np.stack(X_list, axis=0)  # (N, k, V)
    Y = np.asarray(Y_list, dtype=np.int64)

    type_name = "type2" if load_se else "type1"
    out = {
        type_name: {"x": X, "x_first": X[:, 0, :], "y": Y},
        "meta": {"files": [os.path.basename(f) for f in files], "counts_by_file": counts_by_file},
    }
    return out

def evaluate_model(model, X, y):
    """Evaluate a model using 4-fold cross-validation on TEST set.
    
    CRITICAL: Creates fresh model instance for each fold to avoid weight contamination.
    Each fold trains on fold training data and evaluates on fold test data.
    """
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    accuracies = []
    f1s = []
    aurocs = []
    
    fold_num = 0
    for train_idx, test_idx in cv.split(X, y):
        fold_num += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # CRITICAL: Create fresh model for this fold
        # Get the model's initialization parameters
        model_params = {
            'input_dim': model.input_dim,
            'epochs': model.epochs,
            'embed_dim': model.embed_dim,
            'batch_size': model.batch_size,
            'lr': model.lr,
            'model_type': model.model_type,
            'token_level': model.token_level,
        }
        
        # Add method-specific parameters
        if hasattr(model, 'num_layers'):
            model_params['num_layers'] = model.num_layers
            model_params['dropout'] = model.dropout
            model_params['early_stopping'] = model.early_stopping
            model_params['patience'] = model.patience
        
        # Recreate model (fresh weights for this fold)
        if 'tau' in model.model_type.lower():
            from non_linear_notebooks.cross_validation_tune import AlphaModelWrapper
            fold_model = AlphaModelWrapper(**model_params)
        else:
            from non_linear_notebooks.cross_validation_tune import AttentionModelWrapper
            fold_model = AttentionModelWrapper(**model_params)
        
        # Fit on fold training data
        fold_model.fit(X_train, y_train)
        
        # Predict on fold TEST data (held-out)
        y_pred = fold_model.predict(X_test)
        y_proba = fold_model.predict_proba(X_test)[:, 1]  # Get class 1 probability
        
        # Calculate metrics on test fold
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        auroc = roc_auc_score(y_test, y_proba)
        
        accuracies.append(acc)
        f1s.append(f1)
        aurocs.append(auroc)
    
    # Return metrics averaged across folds
    metrics = (np.mean(accuracies), np.mean(f1s), np.mean(aurocs))
    
    return metrics

def evaluate_model_lr(model, X, y):
    """Evaluate MTRE-LR model using 4-fold cross-validation on TEST set.
    
    MTRE-LR (Multi-Token Reasoning Engine - Logistic Regression):
    - Processes tokens SEQUENTIALLY from 0 to token_level-1
    - Accumulates log-odds: log(p) - log(1-p) across tokens
    - Uses sklearn LogisticRegression (no epochs/learning_rate/dropout complexity)
    - Fresh model per fold (no weight contamination)
    """
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    accuracies = []
    f1s = []
    aurocs = []
    
    fold_num = 0
    for train_idx, test_idx in cv.split(X, y):
        fold_num += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # CRITICAL: Create fresh model for this fold
        from non_linear_notebooks.cross_validation_tune import MTRELRWrapper
        fold_model = MTRELRWrapper(
            input_dim=model.input_dim,
            token_level=model.token_level
        )
        
        # Fit on fold training data
        fold_model.fit(X_train, y_train)
        
        # Predict on fold TEST data (held-out)
        y_pred = fold_model.predict(X_test)
        y_proba = fold_model.predict_proba(X_test)[:, 1]  # Get class 1 probability
        
        # Calculate metrics on test fold
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        auroc = roc_auc_score(y_test, y_proba)
        
        accuracies.append(acc)
        f1s.append(f1)
        aurocs.append(auroc)
    
    # Return metrics averaged across folds
    metrics = (np.mean(accuracies), np.mean(f1s), np.mean(aurocs))
    
    return metrics

def trainable(config, ds=None, model_name=None, method_name=None, task=None, early_stopping=None, patience=None):
    # Block tensorboardX on workers too
    import sys
    sys.modules['tensorboardX'] = None
    
    # Suppress verbose progress bars and logging in workers
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    # Set up paths for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        sys.path.append(current_dir)
    
    # Now import inside the function
    from utils.func import read_jsonl
    from utils.parse import parse_multi_choice_response, parse_open_response, eval_multi_choice, eval_open
    from non_linear_notebooks.cross_validation_tune import AttentionModelWrapper, MTRELRWrapper
    from non_linear_notebooks.model_archs.calib import calib_model

    if '2' in task:
        # Type-2 (SE logits):
        X_0_attention = ds["type2"]["x"]        # (N, 10, V)
        X_0 = ds["type2"]["x_first"]  # (N, V)
        y_0 = ds["type2"]["y"]        # (N,)
    else: 
        # Type-1 (original logits) for your attention model:
        X_0_attention = ds["type1"]["x"]        # (N, 10, V)
        X_0 = ds["type1"]["x_first"]  # (N, V)
        y_0 = ds["type1"]["y"]        # (N,)
    
    # Slice X_0_attention to the current token_level from config
    # The dataset was loaded with max tokens (10), so we slice it here
    current_k = config["token_level"]
    if X_0_attention.shape[1] > current_k:
        X_0_attention = X_0_attention[:, :current_k, :]
    
    y_0, m1 = coerce_binary_labels(y_0)
    X_0_attention = X_0_attention[m1]
    X_0 = X_0[m1]

    input_dim = X_0_attention.shape[-1]

    if 'mtre_lr' in method_name:
        # MTRE-LR: sklearn LR with sequential token processing (NO calibration)
        model = MTRELRWrapper(
            input_dim=input_dim,
            token_level=config["token_level"]
        )
        metrics = evaluate_model_lr(model, X_0_attention, y_0)
    else:
        # MTRE-Attention: AttentionModelWrapper with sequential token processing (NO calibration)
        model = AttentionModelWrapper(
            input_dim=input_dim, 
            epochs=100, 
            embed_dim=config["embed_dim"], 
            num_layers=config["num_layers"], 
            batch_size=config["batch_size"], 
            lr=config["lr"], 
            model_type=method_name, 
            token_level=config["token_level"], 
            dropout=0.5, 
            early_stopping=early_stopping, 
            patience=patience
        )
        metrics = evaluate_model(model, X_0_attention, y_0)
    
    # Report to Ray Tune
    session.report({"accuracy": metrics[0], "f1": metrics[1], "auroc": metrics[2]})

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6:
        print("Usage: python MMMU_tune_ray.py <model_name> <method> <task> <early_stopping> <patience>")
        print("Example: python MMMU_tune_ray.py model_name method task True 10")
    else:
        model_name = sys.argv[1]
        method_name = sys.argv[2]
        task = sys.argv[3]
        early_stopping = sys.argv[4].lower() == 'true'
        patience = int(sys.argv[5])

        # Load dataset ONCE in the driver
        print(f"Loading dataset for {model_name} (task={task}) into shared memory...")
        # Output directory: set OUTPUT_DIR env var or use default ./outputs/
        output_dir = os.environ.get("OUTPUT_DIR", "./outputs")
        judged_dir = f"{output_dir}/judged_raw/{model_name}"
        # Load with k=10 (max possible tokens) so we can slice it later
        ds = build_dataset_from_all_judged(
            judged_dir=judged_dir,
            k_first_steps=10,
            max_files=None,
            max_rows_per_file=None,
            task=task
        )
        print("Dataset loaded successfully.")

        # Define search space based on method
        if 'mtre_lr' in method_name:
            # MTRE-LR: only tune token_level (no early-exit, simple sequential processing)
            config = {
               "token_level": tune.grid_search(list(range(1, 11))),
            }
        else:
            # Attention/Tau methods: full hyperparameter space
            config = {
                "embed_dim": tune.choice([64,128,256, 512,]),
                "lr": tune.loguniform(1e-6, 1e-2),
                "num_layers": tune.choice([1, 2, 3, 4,5,6]),
                "batch_size": tune.choice([32,64,128,256,512 ]), #512, 700]),
                "token_level": tune.choice([1, 2, 3, 4,5,6,7,8,9,10]),
            }

        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=10,
            grace_period=1,
            reduction_factor=2
        )

        reporter = CLIReporter(
            metric_columns=["accuracy", "f1", "auroc"],
            print_intermediate_tables=True
        )

        # Use tune.with_parameters to put the dataset in the object store
        # This allows all workers on a node to share the memory (zero-copy)
        result = tune.run(
            tune.with_parameters(
                trainable, 
                ds=ds,
                model_name=model_name, 
                method_name=method_name, 
                task=task, 
                early_stopping=early_stopping, 
                patience=patience,
                #local_dir = "/lustre/scratch5/ceodspspectrum/ray_tmp2/"
            ),
            resources_per_trial={"cpu":8, "gpu": 1},
            config=config,
            storage_path=os.environ.get("RAY_STORAGE_PATH", "./ray_results"),
            resume="AUTO",
            num_samples=10 if 'mtre_lr' in method_name else 216,  # mtre_lr: 10 samples (1 hyperparam), Attention/Tau: 216 samples (5+ hyperparams)
            scheduler=scheduler,
            progress_reporter=reporter,
            verbose=1,
            max_failures=3
        )

        best_trial = result.get_best_trial("auroc", "max", "last")
        print("\n" + "="*80)
        print("TUNING COMPLETE - RESULTS")
        print("="*80)
        print(f"\nBest trial config:")
        for key, value in best_trial.config.items():
            print(f"  {key}: {value}")
        print(f"\nBest trial final metrics:")
        print(f"  Accuracy: {best_trial.last_result['accuracy']:.4f}")
        print(f"  F1 Score: {best_trial.last_result['f1']:.4f}")
        print(f"  AUROC:    {best_trial.last_result['auroc']:.4f}")
        print("="*80 + "\n")
        
        # Properly shutdown Ray to avoid tensorboardX thread issues at shutdown
        ray.shutdown()
