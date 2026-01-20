

def run_analysis(model_name, method_name, task):
    import os
    import json
    import sys
    import numpy as np
    from sklearn.model_selection import cross_validate, StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    import logging
    from utils.func import read_jsonl
    from utils.parse import parse_multi_choice_response, parse_open_response, eval_multi_choice, eval_open
    from non_linear_notebooks.cross_validation import AttentionModelWrapper, AlphaModelWrapper
    import csv
    import pickle
    import torch
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from non_linear_notebooks.model_archs.calib import calib_model
    import re

    # Regex that matches things like: "labeled D", "(D)", "option D", "answer: D", "final answer is D"
    import os, json
    import numpy as np 
    from glob import glob

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
                    # for i in tqdm(range(total), desc=f"Loading {ref_key}", ncols=80):
                    for i in range(total):
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

    print(model_name)
    # print(dataset_name)
    # Output directory: set OUTPUT_DIR env var or use default ./outputs/
    output_dir = os.environ.get("OUTPUT_DIR", "./outputs")
    judged_dir = f"{output_dir}/judged_raw/{model_name}"
    token_level = 10
    ds = build_dataset_from_all_judged(
        judged_dir=judged_dir,
        k_first_steps=token_level,          # first 10 steps
        max_files=None,            # or an int to cap
        max_rows_per_file=None,    # or an int to cap
        task=task
    )
    if '2' in task:
         # Type-2 (SE logits):
         X_0_attention = ds["type2"]["x"]        # (N, 10, V)
         X_0 = ds["type2"]["x_first"]  # (N, V)
         y_0 = ds["type2"]["y"]        # (N,)
    else: 
        # Type-1 (original logits) for your attention model:
        X_0_attention = ds["type1"]["x"]        # (N, token_level, V)
        X_0 = ds["type1"]["x_first"]  # (N, V)
        y_0 = ds["type1"]["y"]        # (N,)

    y_0, m1 = coerce_binary_labels(y_0)
    X_0_attention = X_0_attention[m1]
    X_0 = X_0[m1]

    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    def evaluate_model_old(model, X, y, model_name, save_path=None):
        """Evaluate a model using cross-validation and log results."""
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        res = cross_validate(model, X, y, cv=cv, scoring=('roc_auc', 'accuracy', 'f1'))
        
        metrics = (
            np.mean(res['test_accuracy']), 
            np.mean(res['test_f1']), 
            np.mean(res['test_roc_auc'])
        )
        
        print(f"{model_name}:")
        print(f"Accuracy: {metrics[0]*100:.2f}")
        print(f"F1-Score: {metrics[1]*100:.2f}")
        print(f"AUROC: {metrics[2]*100:.2f}")

        metrics_std = (
            np.std(res['test_accuracy']), 
            np.std(res['test_f1']), 
            np.std(res['test_roc_auc'])
        )

        print(f"Accuracy Standard Deviation: {metrics_std[0]*100:.2f}")
        print(f"F1-Score Standard Deviation: {metrics_std[1]*100:.2f}")
        print(f"AUROC Standard Deviation: {metrics_std[2]*100:.2f}")
        
        return metrics, metrics_std

    def evaluate_model(model, X, y, model_name, save_path=None):
        """Evaluate a model using manual cross-validation.
        
        CRITICAL: Creates fresh model instance for each fold to avoid weight contamination.
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
            
            print(f"\nFold {fold_num}/4 - Train size: {len(y_train)}, Test size: {len(y_test)}")
            
            # Fit fresh model on this fold's training data
            fold_model.fit(X_train, y_train)
            
            # Predict on this fold's test data
            y_pred = fold_model.predict(X_test)
            y_proba = fold_model.predict_proba(X_test)[:, 1]  # Get class 1 probability
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary')
            auroc = roc_auc_score(y_test, y_proba)
            
            accuracies.append(acc)
            f1s.append(f1)
            aurocs.append(auroc)
            
            print(f"Fold {fold_num}: Acc={acc:.4f}, F1={f1:.4f}, AUROC={auroc:.4f}")
        
        metrics = (np.mean(accuracies), np.mean(f1s), np.mean(aurocs))
        
        print(f"\n{model_name}:")
        print(f"Accuracy: {metrics[0]*100:.2f} ± {np.std(accuracies)*100:.2f}")
        print(f"F1-Score: {metrics[1]*100:.2f} ± {np.std(f1s)*100:.2f}")
        print(f"AUROC: {metrics[2]*100:.2f} ± {np.std(aurocs)*100:.2f}")
        
        metrics_std = (np.std(accuracies), np.std(f1s), np.std(aurocs))
        
        return metrics, metrics_std
    # def evaluate_model(model, X, y):
    #     """Evaluate a model using manual cross-validation and return metrics."""
    #     cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        
    #     accuracies = []
    #     f1s = []
    #     aurocs = []
        
    #     for train_idx, test_idx in cv.split(X, y):
    #         X_train, X_test = X[train_idx], X[test_idx]
    #         y_train, y_test = y[train_idx], y[test_idx]
            
    #         # Fit on train
    #         model.fit(X_train, y_train)
            
    #         # Predict on test
    #         y_pred = model.predict(X_test)
    #         y_proba = model.predict_proba(X_test)[:, 1]  # Get class 1 probability
            
    #         # Calculate metrics
    #         acc = accuracy_score(y_test, y_pred)
    #         f1 = f1_score(y_test, y_pred, average='binary')
    #         auroc = roc_auc_score(y_test, y_proba)
            
    #         accuracies.append(acc)
    #         f1s.append(f1)
    #         aurocs.append(auroc)
        
    #     metrics = (np.mean(accuracies), np.mean(f1s), np.mean(aurocs))
        
    #     return metrics
    
    #Other Models
    input_dim=X_0_attention.shape[-1]
    # print(X_0_attention[0])
    print(sum(y_0))

    epochs=100
    batch_size=64
    embed_dim=1024
    # lr=0.0000005
    num_layers=6
    # lr=0.0005
    lr= 0.00010235526079237716
    header = [
        "Epochs",
        "Accuracy", "Accuracy Std",
        "F1", "F1 Std",
        "AUROC", "AUROC Std"
    ]

    csv_filename = f"./scratch_results/mmmu_{model_name}_{method_name}.csv"
    # Extract metrics in required order: accuracy, f1, roc_auc
    accuracy_idx = 0
    f1_idx = 1
    roc_auc_idx = 2
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        if 'first_token' in method_name:
            logreg = LogisticRegression(max_iter=100)
            metrics_mean, metrics_std = evaluate_model(logreg, X_0, y_0, 'Log_Reg')
            writer.writerow([
                epochs,
                metrics_mean[accuracy_idx], metrics_std[accuracy_idx],
                metrics_mean[f1_idx], metrics_std[f1_idx],
                metrics_mean[roc_auc_idx], metrics_std[roc_auc_idx],
            ])
        else:
            for epochs in [100]:#[50, 75, 100, 150, 200, 250, 300, 350, 375, 400, 450, 500, 550, 600, 700, 1000]:
                print(f"\nEvaluating model with {epochs} epochs...")
                if 'tau' in method_name:
                    model = AlphaModelWrapper(input_dim=input_dim, epochs=epochs, embed_dim=embed_dim, batch_size=batch_size, lr=lr, model_type=method_name, token_level=token_level)
                else:
                    model = AttentionModelWrapper(input_dim=input_dim, epochs=epochs, embed_dim=embed_dim, num_layers=num_layers, batch_size=batch_size, lr=lr, model_type=method_name, token_level=token_level, dropout=.5)
                metrics_mean, metrics_std = evaluate_model(model, X_0_attention, y_0, model_name, 'Log_Reg')
                # metrics_mean = evaluate_model(model, X_0_attention, y_0)
                print(metrics_mean)

                writer.writerow([
                    epochs,
                    metrics_mean[accuracy_idx], metrics_std[accuracy_idx],
                    metrics_mean[f1_idx], metrics_std[f1_idx],
                    metrics_mean[roc_auc_idx], metrics_std[roc_auc_idx],
                ])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python MMMU_eval.py <model_name> <method> <task>")
    else:
        run_analysis(sys.argv[1],sys.argv[2],sys.argv[3])
