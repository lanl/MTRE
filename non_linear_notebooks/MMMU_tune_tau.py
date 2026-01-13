"""
Ray Tune for Tau (AlphaModelWrapper) with Calibrator Parameter Search.

This script extends the standard hyperparameter tuning to include:
1. Model parameters: embed_dim, lr, batch_size, token_level
2. Calibrator parameters: A, B, Kmax, lam

Separate from MMMU_tune_ray.py to avoid conflicts with Attention model tuning.
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
from sklearn.model_selection import cross_val_score
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


from tqdm import tqdm
from glob import glob

def _load_first_k_from_ins(ins, ref_key, inline_key, k=10):
    """Returns a (k, V) float32 array from logits pointer or inline logits."""
    arr = None
    ref = ins.get(ref_key)
    if ref and isinstance(ref, dict) and "npy" in ref:
        npy = ref["npy"]
        if os.path.exists(npy):
            try:
                mm = np.load(npy, mmap_mode="r")
                total = min(k, mm.shape[0])
                data = []
                for i in tqdm(range(total), desc=f"Loading {ref_key}", ncols=80):
                    data.append(mm[i])
                arr = np.array(data)
            except Exception as e:
                print(f"Error loading {npy}: {e}")
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


def build_dataset_from_all_judged(
    judged_dir: str,
    k_first_steps: int = 10,
    max_files: int | None = None,
    max_rows_per_file: int | None = None,
    task: str = 'mmmu',
    show_progress: bool = True,
):
    """Aggregate dataset from judged files."""
    if 'mme' in task:
        pattern = os.path.join(judged_dir, "mme.jsonl_judged.jsonl")
    else:
        pattern = os.path.join(judged_dir, "*_validation.jsonl_judged.jsonl")

    files = sorted(glob(pattern))
    if max_files is not None:
        files = files[:max_files]

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

    X = np.stack(X_list, axis=0)
    Y = np.asarray(Y_list, dtype=np.int64)

    type_name = "type2" if load_se else "type1"
    out = {
        type_name: {"x": X, "x_first": X[:, 0, :], "y": Y},
        "meta": {"files": [os.path.basename(f) for f in files], "counts_by_file": counts_by_file},
    }
    return out


def evaluate_model_tau(model, X, y):
    """
    Evaluate tau model using 4-fold cross-validation (matching MMMU_tune_ray.py for Attention).
    
    CRITICAL: Creates fresh model instance for each fold to avoid weight contamination.
    Each fold trains on fold training data and evaluates on fold test data.
    
    For each fold:
    - Creates fresh AlphaModelWrapper with same hyperparameters
    - model.fit(X_train, y_train) calls AlphaModelWrapper.fit()
      - Which internally does 5-fold cross-fitting for NN training and calibration
      - Trains on X_train, calibrator parameters tuned on OOF from X_train only
    - Evaluate on X_test (held-out fold data)
    
    Returns: tuple of (accuracy, f1, auroc) averaged across 4 folds
    Each metric is the MEAN across folds.
    """
    # Import here to ensure availability in worker context
    from non_linear_notebooks.cross_validation_tune import AlphaModelWrapper, MTRELRWrapper
    
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    accuracies = []
    f1s = []
    aurocs = []
    
    fold_num = 0
    for train_idx, test_idx in cv.split(X, y):
        fold_num += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"\n--- Fold {fold_num}/4 ---")
        print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        
        # CRITICAL: Create FRESH model for this fold to avoid weight contamination
        # Route based on model type
        if isinstance(model, MTRELRWrapper):
            # LR-based tau: simple initialization
            fold_model = MTRELRWrapper(
                input_dim=model.input_dim,
                token_level=model.token_level,
                A=model.A,
                B=model.B,
                lam=model.lam
            )
        else:
            # Attention-based tau: full hyperparameter initialization
            fold_model = AlphaModelWrapper(
                input_dim=model.input_dim,
                epochs=model.epochs,
                embed_dim=model.embed_dim,
                batch_size=model.batch_size,
                lr=model.lr,
                model_type=model.model_type,
                token_level=model.token_level
            )
        
        # Fit on fold training data
        print("Training tau model with internal calibration...")
        fold_model.fit(X_train, y_train)
        
        # Evaluate on fold TEST data (held-out)
        print("Generating predictions on held-out fold test data...")
        y_pred = fold_model.predict(X_test)
        y_proba = fold_model.predict_proba(X_test)[:, 1]
        
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


def trainable_tau(config, ds=None, model_name=None, method_name=None, task=None, early_stopping=None, patience=None):
    """
    Ray trainable function for tau hyperparameter tuning with unified calibration.
    
    Supports both:
    - Attention-based tau: AlphaModelWrapper
    - LR-based tau: MTRELRWrapper  
    
    Both methods use the SAME calibration strategy (A/B/lam via SequentialReliabilityCalibrator).
    Only the base model architecture differs.
    
    Tunes shared params:
    - token_level: max tokens to read
    - A: positive log-odds threshold for early exit
    - B: negative log-odds threshold for early exit
    - lam: exponential decay factor over tokens
    
    Method-specific params:
    - For 'tau' (Attention): embed_dim, batch_size, lr
    - For 'mtre_tau_lr' (LR): none (all params are shared)
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
    from non_linear_notebooks.cross_validation_tune import AlphaModelWrapper, MTRELRWrapper
    
    # Load and prepare data
    if '2' in task:
        X_0_attention = ds["type2"]["x"]
        y_0 = ds["type2"]["y"]
    else:
        X_0_attention = ds["type1"]["x"]
        y_0 = ds["type1"]["y"]
    
    # Slice to current token_level
    current_k = config["token_level"]
    if X_0_attention.shape[1] > current_k:
        X_0_attention = X_0_attention[:, :current_k, :]
    
    y_0, m1 = coerce_binary_labels(y_0)
    X_0_attention = X_0_attention[m1]
    
    input_dim = X_0_attention.shape[-1]
    
    # Route based on method_name
    if 'mtre_tau_lr' in method_name:
        # LR-based tau: Pass A, B, lam sampled from Ray to SearchConfig
        model = MTRELRWrapper(
            input_dim=input_dim,
            token_level=config["token_level"],
            A=config.get("A", 0.01),      # Ray-sampled threshold
            B=config.get("B", 0.01),      # Ray-sampled threshold
            lam=config.get("lam", 0.0),   # Ray-sampled decay
        )
        print(f"\nTraining MTRE-Tau-LR (sklearn LR + SequentialReliabilityCalibrator)...")
        print(f"  A={config.get('A', 0.01):.6f}, B={config.get('B', 0.01):.6f}, lam={config.get('lam', 0.0):.6f}")
    else:
        # Attention-based tau: full config with model hyperparams
        # Build config_dict with ALL parameters (model + calibrator)
        config_dict = {
            'input_dim': input_dim,
            'embed_dim': config["embed_dim"],
            'num_heads': 8,  # Fixed
            'num_layers': 3,  # Fixed for tau
            'dropout': 0.5,
            'epochs': 100,
            'batch_size': config["batch_size"],
            'lr': config["lr"],
            'token_level': config["token_level"],
            'model_type': 'tau',
            # Calibrator parameters
            'A': config.get("A", 0.01),
            'B': config.get("B", 0.01),
            'Kmax': config["token_level"],  # Constrain Kmax to available tokens
            'lam': config.get("lam", 0.0),
        }
        
        model = AlphaModelWrapper(
            input_dim=input_dim, 
            epochs=config_dict['epochs'],
            embed_dim=config["embed_dim"], 
            batch_size=config["batch_size"], 
            lr=config["lr"], 
            model_type='tau', 
            token_level=config["token_level"]
        )
        print(f"\nTraining Attention-based tau (with SequentialReliabilityCalibrator)...")
    
    # Unified evaluation for both methods
    metrics = evaluate_model_tau(model, X_0_attention, y_0)
    
    # Report metrics averaged across 4 test folds to Ray Tune
    # metrics is a tuple: (avg_acc, avg_f1, avg_auroc) from 4-fold CV
    # Each value is the MEAN across the 4 held-out test folds
    # The scheduler optimizes the "auroc" metric (averaged across folds)
    print(f"\nReporting Trial Metrics: Accuracy={metrics[0]:.4f}, F1={metrics[1]:.4f}, AUROC={metrics[2]:.4f}")
    session.report({
        "accuracy": float(metrics[0]),    # 4-fold CV test set accuracy (averaged)
        "f1": float(metrics[1]),          # 4-fold CV test set F1 (averaged)
        "auroc": float(metrics[2]),       # 4-fold CV test set AUROC (averaged, scheduler optimizes this)
        "training_iteration": 1           # Indicates this trial is complete
    })


if __name__ == "__main__":
        import sys

        if len(sys.argv) != 7:
            print("Usage: python MMMU_tune_tau.py <model_name> <task> <method_name> <num_samples> <early_stopping> <patience>")
            print("Example: python MMMU_tune_tau.py Intern_VL_3_5 mmmu mtre_tau_lr 108 True 5")
            sys.exit(1)

        def str2bool(s):
            return str(s).strip().lower() in {"true","1","t","yes","y"}

        model_name     = sys.argv[1]
        task           = sys.argv[2]
        method_name    = sys.argv[3]
        num_samples    = int(sys.argv[4])          # <- was [3]
        early_stopping = str2bool(sys.argv[5])     # <- was using wrong guard
        patience       = int(sys.argv[6])          # <- was using wrong guard

        # Load dataset ONCE in the driver
        print(f"Loading dataset for {model_name} (task={task}) into shared memory...")
        # Output directory: set OUTPUT_DIR env var or use default ./outputs/
        output_dir = os.environ.get("OUTPUT_DIR", "./outputs")
        judged_dir = f"{output_dir}/judged_raw/{model_name}"
        ds = build_dataset_from_all_judged(
            judged_dir=judged_dir,
            k_first_steps=10,
            max_files=None,
            max_rows_per_file=None,
            task=task
        )
        print("Dataset loaded successfully.")

        # ========== TAU-SPECIFIC SEARCH SPACE ==========
        # Model parameters
        config = {
            "embed_dim": tune.choice([128,256, 512, 1024]),
            "batch_size": tune.choice([64, 128, 256, 512]),
            "lr": tune.loguniform(1e-5, 1e-3),
            "token_level": tune.choice([1,2, 3, 4, 5, 6, 7, 8, 9, 10]),
            
            # Calibrator parameters: Early-exit thresholds and decay
            "A": tune.loguniform(0.001, 0.1),      # Positive threshold (log-odds)
            "B": tune.loguniform(0.001, 0.1),      # Negative threshold (log-odds)
            "lam": tune.choice([0.0, 0.05, 0.1, 0.2]),  # Exponential decay factor over tokens
            # Note: Kmax (max tokens to read) is constrained by token_level during training
        }

        scheduler = ASHAScheduler(
            metric="auroc",  # Optimize for AUROC
            mode="max",
            max_t=4,  # Allow 4 checkpoints for early stopping to work
            grace_period=1,  # Allow at least 1 checkpoint before pausing
            reduction_factor=2  # Eliminate bottom 50% of trials at each pause point
        )

        reporter = CLIReporter(
            metric_columns=["accuracy", "f1", "auroc"],
            print_intermediate_tables=True
        )

        result = tune.run(
            tune.with_parameters(
                trainable_tau, 
                ds=ds,
                model_name=model_name,
                method_name=method_name,  # Default to Attention-based tau
                task=task,
                early_stopping=early_stopping,
                patience=patience
            ),
            resources_per_trial={"cpu": 60, "gpu": 2},
            config=config,
            storage_path=os.environ.get("RAY_STORAGE_PATH", "./ray_results"),
            resume="AUTO",
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            verbose=1,
            max_failures=3,
            stop={"training_iteration": 1}  # Stop after 1 iteration (full evaluation)
        )

        # Print results
        best_trial = result.get_best_trial("auroc", "max", "last")
        print("\n" + "="*80)
        print("TAU TUNING COMPLETE - RESULTS")
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
        best_config_file = f"./scratch_results/tau_best_config_{model_name}.json"
        os.makedirs("./scratch_results", exist_ok=True)
        with open(best_config_file, 'w') as f:
            json.dump({
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
