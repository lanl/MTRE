def run_analysis(model_name, dataset_name, method_name):
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


    # ---- reuse the loader from before (slightly trimmed for this script) ----
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
                    mm = np.load(npy, mmap_mode="r")   # (T,V) or (T,1,V)
                    arr = np.array(mm[:k])             # materialize only k rows
                except Exception:
                    arr = None

        if arr is None and inline_key in ins and ins[inline_key] is not None:
            arr = np.asarray(ins[inline_key])[:k]

        if arr is None:
            return None

        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        t, V = arr.shape[0], arr.shape[1]
        if t < k:
            pad = np.zeros((k - t, V), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        return arr.astype(np.float32, copy=False)
    
    # --- helper maps for parsing "One, Two, Three..." ---
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

    def _normalize_token(token: str) -> str:
        t = token.strip()
        t = re.sub(r'^\s*and\s+', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s+and\s*$', '', t, flags=re.IGNORECASE)
        t = re.sub(r'[.\s]+$', '', t)
        t = re.sub(r'\s+', ' ', t.lower())
        return t

    def _token_to_int(token: str):
        t = _normalize_token(token)
        if t.isdigit():
            return int(t)
        return NUMBER_WORDS.get(t)

    def _parse_spoken_numbers(text: str):
        """Extract integers from 'One, Two, Three...' style response."""
        if not text:
            return []
        if ',' in text:
            raw = text.split(',')
        else:
            raw = re.findall(r'\b(?:and\s+)?([A-Za-z-]+|\d+)\b', text)
        out = []
        for tok in raw:
            n = _token_to_int(tok)
            if n is not None:
                out.append(n)
        return out

    def _is_correct_count(response: str, label: int) -> bool:
        seq = _parse_spoken_numbers(response)
        if not seq:
            return False
        return len(seq) == int(label) and all(v == i+1 for i, v in enumerate(seq))
    # ---------- Build dataset ----------
    def build_dataset_from_jsonl(
        jsonl_path: str,
        k_first_steps: int = 10,
        ref_key: str = "logits_ref",
        inline_key: str = "logits",
    ):
        """
        Build dataset from a JSONL file.
        y = 1 if model counted correctly, else 0.
        Uses the same _load_first_k_from_ins logic as run_analysis.
        """
        if 'Squares' in jsonl_path:
            end_point = 600
        else:
            end_point = 1080

        X_list, Y_list, ids = [], [], []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                ins = json.loads(line)
                label = ins.get("label")
                if label is None:
                    continue
                response = ins.get("response", "")
                y = 1 if _is_correct_count(response, int(label)) else 0

                x = _load_first_k_from_ins(ins, ref_key=ref_key, inline_key=inline_key, k=k_first_steps)
                if x is None:
                    continue

                X_list.append(x)
                Y_list.append(y)
                ids.append(ins.get("id", f"row_{i}"))
                if i>end_point:
                    break


        if not X_list:
            raise RuntimeError(f"No usable samples found in {jsonl_path}")

        X = np.stack(X_list, axis=0)          # (N, k, V)
        Y = np.asarray(Y_list, dtype=np.int64)
        meta = {"ids": ids, "n_samples": len(Y)}

        return {"x": X, "x_first": X[:, 0, :], "y": Y, "meta": meta}
    




           
    answers = {
        0: 'zero',
        1: 'one',
        2: 'two',
        3: 'Three',
        4: 'Four',
        5: 'Five',
        6: 'Six',
        7: 'Seven',
        8: 'Eight',
        9: 'Nine',
        10: 'Ten',
        11: 'Eleven',
        12: 'Twelve',
        13: 'Thirteen',
        14: 'Fourteen',
        15: 'Fifteen',
        16: 'Sixteen',
        17: 'Seventeen',
        18: 'Eighteen',
        19: 'Nineteen',
        20: 'Twenty'
    }
    print(model_name)
    # print(dataset_name)
    # X_0_attention, X_0, y_0 = get_dataset(model_name, dataset_name)
    # Output directory: set OUTPUT_DIR env var or use default ./outputs/
    output_dir = os.environ.get("OUTPUT_DIR", "./outputs")
    jsonl_path = f"{output_dir}/{model_name}/{dataset_name}.jsonl"
    token_level = 10
    ds = build_dataset_from_jsonl(
        jsonl_path,
        k_first_steps=token_level,
        ref_key="logits_ref",
        inline_key= "logits",
    )
    print(ds["x"].shape, ds["y"].shape)

    # Type-1 (original logits) for your attention model:
    X_0_attention = ds["x"]        # (N, 10, V)
    X_0= ds["x_first"]  # (N, V)
    y_0 = ds["y"]        # (N,)

    y_0, m1 = coerce_binary_labels(y_0)
    X_0_attention = X_0_attention[m1]
    X_0 = X_0[m1]
    print(y_0)
    print(X_0.shape)





    # print(ds["meta"]["files"])
    # print(ds["meta"]["counts_by_file"])
    # Count each unique label
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    def eval_lr_delta(X_0, y_0):
        # Initialize the logistic regression model
        logreg = LogisticRegression(max_iter=100)

        # Initialize KFold (5-fold cross-validation in this case)
        kf = KFold(n_splits=4)#, shuffle=True, random_state=42)

        # To track performance across folds
        all_misclassified = []
        accuracies = []
        auc_scores = []
        precisions = []
        f1s = []

        # Perform manual cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_0)):
            # Train and test split for each fold
            X_train, X_test = X_0[train_idx], X_0[test_idx]
            y_train, y_test = y_0[train_idx], y_0[test_idx]
            
            # Fit model on the training set
            logreg.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = logreg.predict(X_test)
            y_prob = logreg.predict_proba(X_test)[:, 1]  # For AUC, we need the probabilities
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred, average='binary')  # Calculate F1 score
            
            # Store metrics for each fold
            accuracies.append(accuracy)
            auc_scores.append(auc)
            f1s.append(f1)
            
        # Print overall performance metrics
        print("\nCross-validation performance across all folds:")
        print(f"Mean Accuracy: {np.mean(accuracies)*100}")
        print(f"Mean AUC: {np.mean(auc_scores)*100}")
        print(f"Mean F1: {np.mean(f1s)*100}")

        print(f"Standard Deviation of Accuracy: {np.std(accuracies)*100}")
        print(f"Standard Deviation of AUC: {np.std(auc_scores)*100}")
        print(f"Standard Deviation of F1: {np.std(f1s)*100}")

        metrics = (
            np.mean(accuracies), 
            np.mean(f1s), 
            np.mean(auc_scores)
        )

        metrics_std = (
            np.std(accuracies), 
            np.std(f1s), 
            np.std(auc_scores)
        )

        return metrics, metrics_std

    def evaluate_model(model, X, y, model_name, save_path=None):
        """Evaluate a model using cross-validation and log results."""
        # cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        res = cross_validate(model, X, y, cv=4, scoring=('roc_auc', 'accuracy', 'f1'))
        
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
        
    #Other Models
    input_dim=X_0_attention.shape[-1]
    # print(X_0_attention[0])
    # print(y_0)

    epochs=350
    batch_size=700
    embed_dim=512
    # lr=0.0000005
    num_layers=3
    # lr=0.0005
    lr=0.000005
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
            metrics_mean, metrics_std = eval_lr_delta(X_0, y_0)
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
                    model = AlphaModelWrapper(input_dim=input_dim, epochs=epochs, embed_dim=embed_dim, batch_size=batch_size, lr=lr, model_type=method_name)
                else:
                    model = AttentionModelWrapper(input_dim=input_dim, epochs=epochs, embed_dim=embed_dim, num_layers=num_layers, batch_size=batch_size, lr=lr, model_type=method_name, token_level=token_level, dropout=.5)
                metrics_mean, metrics_std = evaluate_model(model, X_0_attention, y_0, model_name)

                writer.writerow([
                    epochs,
                    metrics_mean[accuracy_idx], metrics_std[accuracy_idx],
                    metrics_mean[f1_idx], metrics_std[f1_idx],
                    metrics_mean[roc_auc_idx], metrics_std[roc_auc_idx],
                ])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python arith_all_large.py <model_name> <dataset_name> <method>")
    else:
        run_analysis(sys.argv[1],sys.argv[2],sys.argv[3])

