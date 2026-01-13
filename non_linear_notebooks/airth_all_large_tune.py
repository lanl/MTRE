"""
Unified Ray Tune for both Attention and Tau methods on Arithmetic datasets.

This script provides a single interface to tune:
- Attention: AttentionModelWrapper with standard model parameters
- Tau: AlphaModelWrapper with model + calibrator parameters

Both use identical 4-fold cross-validation evaluation scheme (test set metrics).
Loads data from local JSONL format (output_raw, not judged_raw).
"""

import os
import sys
import atexit

# Completely block tensorboardX from being imported
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging
import warnings
from functools import partial

# Suppress all warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='.*lbfgs failed to converge.*')
warnings.filterwarnings('ignore', message='.*STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.*')

# Suppress verbose logging from libraries
logging.getLogger('ray').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)

import csv
import pickle
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re

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


def _load_first_k_from_ins(ins, ref_key, inline_key, k=10):
    """Returns a (k, V) float32 array from logits pointer or inline logits."""
    arr = None
    ref = ins.get(ref_key)
    if ref and isinstance(ref, dict) and "npy" in ref:
        npy = ref["npy"]
        if os.path.exists(npy):
            try:
                mm = np.load(npy, mmap_mode="r")
                arr = np.array(mm[:k])
            except Exception:
                arr = None

    if arr is None and inline_key in ins and ins[inline_key] is not None:
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


def _parse_spoken_numbers(text: str):
    """Extract integers from 'One, Two, Three...' style response."""
    NUMBER_WORDS = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
        "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
        "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,
        "nineteen":19,"twenty":20,"twenty-one":21,"twenty two":22,"twenty-two":22,
        "twenty three":23,"twenty-three":23,"twenty four":24,"twenty-four":24,
        "twenty five":25,"twenty-five":25,"twenty six":26,"twenty-six":26,
        "twenty seven":27,"twenty-seven":27,"twenty eight":28,"twenty-eight":28,
        "twenty nine":29,"twenty-nine":29,"thirty":30
    }
    
    if not text:
        return []
    if ',' in text:
        raw = text.split(',')
    else:
        import re
        raw = re.findall(r'\b(?:and\s+)?([A-Za-z-]+|\d+)\b', text)
    out = []
    for tok in raw:
        t = tok.strip().lower()
        if t.isdigit():
            out.append(int(t))
        elif t in NUMBER_WORDS:
            out.append(NUMBER_WORDS[t])
    return out


def _is_correct_count(response: str, label: int) -> bool:
    """Check if response counts correctly (1 if correct, 0 if not)."""
    seq = _parse_spoken_numbers(response)
    if not seq:
        return False
    return len(seq) == int(label) and all(v == i+1 for i, v in enumerate(seq))


def build_dataset_from_jsonl(
    jsonl_path: str,
    k_first_steps: int = 10,
    ref_key: str = "logits_ref",
    inline_key: str = "logits",
):
    """Load dataset from a single JSONL file (output_raw format).
    
    For arithmetic datasets: y = 1 if model counted correctly, else 0.
    Uses response and label to determine correctness.
    """
    if not os.path.exists(jsonl_path):
        raise RuntimeError(f"JSONL file not found: {jsonl_path}")

    X_list, Y_list = [], []

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ins = json.loads(line)

            # Load logits
            x = _load_first_k_from_ins(ins, ref_key=ref_key, inline_key=inline_key, k=k_first_steps)
            if x is None:
                continue

            # Load label and response
            label = ins.get("label")
            if label is None:
                continue
            response = ins.get("response", "")
            
            # y = 1 if model counted correctly, else 0
            y = 1 if _is_correct_count(response, int(label)) else 0

            X_list.append(x)
            Y_list.append(y)

    if not X_list:
        raise RuntimeError(f"No usable samples found in {jsonl_path}")

    X = np.stack(X_list, axis=0)  # (N, k, V)
    Y = np.asarray(Y_list, dtype=np.int64)

    out = {
        "x": X,
        "x_first": X[:, 0, :],
        "y": Y,
        "meta": {"file": jsonl_path, "count": len(Y)},
    }
    return out


def evaluate_model_with_cv(model, X, y, method_type='attention'):
    """
    Evaluate model using 4-fold cross-validation on TEST set.
    
    Both Attention and Tau use this same evaluation scheme:
    - For each fold: train on fold training data, evaluate on fold test data
    - For Tau: internal 5-fold CV for calibration happens during fit() on training data
    - Returns: tuple of (accuracy, f1, auroc) averaged across 4 folds
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
        
        print(f"\n--- Fold {fold_num}/4 ({method_type.upper()}) ---")
        print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        
        # Fit on fold training data
        if method_type == 'tau':
            print("Training tau model with internal calibration (5-fold CV on training data)...")
        else:
            print("Training attention model...")
        model.fit(X_train, y_train)
        
        # Evaluate on fold TEST data (held-out)
        print("Generating predictions on held-out test data...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics on test fold
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        auroc = roc_auc_score(y_test, y_proba)
        
        accuracies.append(acc)
        f1s.append(f1)
        aurocs.append(auroc)
        
        print(f"Fold {fold_num} metrics: Acc={acc:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}")
    
    # Average across folds
    avg_acc = np.mean(accuracies)
    avg_f1 = np.mean(f1s)
    avg_auroc = np.mean(aurocs)
    
    print(f"\n=== Cross-Fold Average (4 folds) ===")
    print(f"  Accuracy: {avg_acc:.4f} ± {np.std(accuracies):.4f}")
    print(f"  F1 Score: {avg_f1:.4f} ± {np.std(f1s):.4f}")
    print(f"  AUROC:    {avg_auroc:.4f} ± {np.std(aurocs):.4f}")
    
    return (avg_acc, avg_f1, avg_auroc)


def trainable_unified(config, ds=None, model_name=None, dataset_name=None, method_name=None):
    """
    Ray trainable function for unified tuning of both Attention and Tau.
    
    Args:
        config: Ray config dict with:
                - Common: embed_dim, batch_size, lr, token_level
                - Attention: num_layers, dropout
                - Tau: A, B, lam (calibrator parameters)
        method_name: 'attention', 'tau', or other variant
        All other args: dataset, model name, dataset name
    """
    # Block tensorboardX on workers
    import sys
    sys.modules['tensorboardX'] = None
    
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
    # Set up paths for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
        sys.path.append(current_dir)
    
    # Import inside function to avoid worker serialization issues
    from non_linear_notebooks.cross_validation import AttentionModelWrapper, AlphaModelWrapper
    
    # Load and prepare data
    X_0_attention = ds["x"]
    y_0 = ds["y"]
    
    # Slice to current token_level
    current_k = config["token_level"]
    if X_0_attention.shape[1] > current_k:
        X_0_attention = X_0_attention[:, :current_k, :]
    
    y_0, m1 = coerce_binary_labels(y_0)
    X_0_attention = X_0_attention[m1]
    
    input_dim = X_0_attention.shape[-1]
    
    # Create model based on method type
    if 'tau' in method_name.lower():
        model = AlphaModelWrapper(
            input_dim=input_dim, 
            epochs=100,
            embed_dim=config["embed_dim"], 
            batch_size=config["batch_size"], 
            lr=config["lr"], 
            model_type=method_name, 
            token_level=config["token_level"]
        )
    else:  # attention or other variants
        model = AttentionModelWrapper(
            input_dim=input_dim, 
            epochs=100, 
            embed_dim=config["embed_dim"], 
            num_layers=config["num_layers"], 
            batch_size=config["batch_size"], 
            lr=config["lr"], 
            model_type=method_name, 
            token_level=config["token_level"], 
            dropout=config.get("dropout", 0.5)
        )
    
    # Evaluate using 4-fold CV on test set
    metrics = evaluate_model_with_cv(model, X_0_attention, y_0, method_type=method_name)
    
    # Report to Ray Tune
    print(f"\nReporting Trial Metrics: Accuracy={metrics[0]:.4f}, F1={metrics[1]:.4f}, AUROC={metrics[2]:.4f}")
    session.report({
        "accuracy": float(metrics[0]),
        "f1": float(metrics[1]),
        "auroc": float(metrics[2]),
        "training_iteration": 1
    })


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print("Usage: python arith_all_large_tune.py <model_name> <dataset_name> <method> <num_samples> [early_stopping] [patience]")
        print("\nMethods:")
        print("  attention        - Attention-based model")
        print("  tau              - Tau (AlphaModelWrapper) with calibration")
        print("\nExamples:")
        print("  python arith_all_large_tune.py Intern_VL_3_5 math23k attention 216")
        print("  python arith_all_large_tune.py Intern_VL_3_5 math23k tau 108")
    else:
        model_name = sys.argv[1]
        dataset_name = sys.argv[2]
        method_name = sys.argv[3]
        num_samples = int(sys.argv[4])

        # Load dataset ONCE in the driver
        print(f"Loading dataset {dataset_name} for {model_name} into shared memory...")
        # Output directory: set OUTPUT_DIR env var or use default ./outputs/
        output_dir = os.environ.get("OUTPUT_DIR", "./outputs")
        jsonl_path = f"{output_dir}/{model_name}/{dataset_name}.jsonl"
        ds = build_dataset_from_jsonl(
            jsonl_path,
            k_first_steps=10,
            ref_key="logits_ref",
            inline_key="logits"
        )
        print(f"Dataset loaded successfully. Shape: {ds['x'].shape}, Labels: {ds['y'].shape}")

        # Define search space based on method
        if 'tau' in method_name.lower():
            print(f"\n{'='*80}")
            print(f"TUNING TAU METHOD WITH CALIBRATOR PARAMETERS")
            print(f"{'='*80}\n")
            
            config = {
                # Model parameters
                "embed_dim": tune.choice([256, 512, 1024]),
                "batch_size": tune.choice([64, 128, 256, 512]),
                "lr": tune.loguniform(1e-5, 1e-3),
                "token_level": tune.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]),
                
                # Calibrator parameters
                "A": tune.loguniform(0.001, 0.1),
                "B": tune.loguniform(0.001, 0.1),
                "lam": tune.choice([0.0, 0.05, 0.1, 0.2]),
            }
            
            scheduler = ASHAScheduler(
                metric="auroc",
                mode="max",
                max_t=4,
                grace_period=1,
                reduction_factor=2
            )
            
            result_name = f"tau_best_config_{model_name}_{dataset_name}"
        else:  # attention
            print(f"\n{'='*80}")
            print(f"TUNING ATTENTION METHOD")
            print(f"{'='*80}\n")
            
            config = {
                # Model parameters
                "embed_dim": tune.choice([64, 128, 256, 512, 1024]),
                "batch_size": tune.choice([32, 64, 128, 256, 512, 700]),
                "lr": tune.loguniform(1e-6, 1e-2),
                "num_layers": tune.choice([1, 2, 3, 4, 5, 6]),
                "token_level": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                "dropout": tune.choice([0.3, 0.5, 0.7]),
            }
            
            scheduler = ASHAScheduler(
                metric="auroc",
                mode="max",
                max_t=10,
                grace_period=1,
                reduction_factor=2
            )
            
            result_name = f"attention_best_config_{model_name}_{dataset_name}"

        reporter = CLIReporter(
            metric_columns=["accuracy", "f1", "auroc"],
            print_intermediate_tables=True
        )

        result = tune.run(
            tune.with_parameters(
                trainable_unified, 
                ds=ds,
                model_name=model_name, 
                dataset_name=dataset_name,
                method_name=method_name
            ),
            resources_per_trial={"cpu": 4, "gpu": 1},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            verbose=1,
            max_failures=3,
            stop={"training_iteration": 1}
        )

        # Print results
        best_trial = result.get_best_trial("auroc", "max", "last")
        print("\n" + "="*80)
        print(f"TUNING COMPLETE - {method_name.upper()} RESULTS")
        print("="*80)
        print(f"\nBest trial config:")
        for key, value in best_trial.config.items():
            print(f"  {key}: {value}")
        print(f"\nBest trial final metrics:")
        print(f"  Accuracy: {best_trial.last_result['accuracy']:.4f}")
        print(f"  F1 Score: {best_trial.last_result['f1']:.4f}")
        print(f"  AUROC:    {best_trial.last_result['auroc']:.4f}")
        print("="*80 + "\n")
        
        # Save best config to file
        os.makedirs("./scratch_results", exist_ok=True)
        best_config_file = f"./scratch_results/{result_name}.json"
        with open(best_config_file, 'w') as f:
            json.dump({
                'method': method_name,
                'model': model_name,
                'dataset': dataset_name,
                'config': best_trial.config,
                'metrics': {
                    'accuracy': float(best_trial.last_result['accuracy']),
                    'f1': float(best_trial.last_result['f1']),
                    'auroc': float(best_trial.last_result['auroc']),
                }
            }, f, indent=2)
        print(f"Best config saved to {best_config_file}")
        
        # Properly shutdown Ray
        ray.shutdown()
