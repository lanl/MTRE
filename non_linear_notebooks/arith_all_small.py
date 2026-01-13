

def run_analysis(model_name, dataset_name, method_name):
    import os
    import json
    import sys
    import numpy as np
    from sklearn.model_selection import cross_validate
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    import logging
    from utils.func import read_jsonl
    from non_linear_notebooks.cross_validation import AttentionModelWrapper, AlphaModelWrapper
    import csv
    import pickle
    import torch
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from non_linear_notebooks.model_archs.calib import calib_model

    def get_dataset(model_name, dataset_name):
        answers = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
                6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', 11: 'Eleven', 12: 'Twelve'}
        
        if dataset_name in ['Squares', 'Lines', 'Triangle', 'OlympicLikeLogo']:
            logits = read_jsonl(f"./output/{model_name}/{dataset_name}.jsonl")

            if dataset_name =='Squares':
                x_train = np.array([ins['logits'] for ins in logits[:600]]).astype(np.float32) #USED FOR ATTENTION MODEL
                x_train_first = np.array([ins['logits'][0] for ins in logits[:600]]).astype(np.float32) #USED FOR FIRST TOKEN MODEL
                labels = np.array([ins['label'] for ins in logits[:600]])
                if model_name == 'MiniGPT4':
                    preds = np.array([ins['response'] for ins in logits[:600]])
                else:
                    preds = np.array([ins['response'].split('.')[0] for ins in logits[:600]])
            else:
                x_train = np.array([ins['logits'] for ins in logits[:]]).astype(np.float32) #USED FOR ATTENTION MODEL
                x_train_first = np.array([ins['logits'][0] for ins in logits[:]]).astype(np.float32) #USED FOR FIRST TOKEN MODEL
                labels = np.array([ins['label'] for ins in logits[:]])
                preds = np.array([ins['response'].split('.')[0] for ins in logits[:]])

            #NOTE THE PARSING OF RESPONSES DEPENDS ON YOUR MODEL OUTPUTS:
            if dataset_name == 'Squares': 
                if model_name in ['LLaVA-7B', 'LLaMA_Adapter']:#, 'mPLUG-Owl']:
                    y_0 = np.array([
                        1 if str(answers[labels[i]]).lower() in preds[i].lower() else 0
                        for i in range(len(preds))
                    ])
                elif model_name in ['mPLUG-Owl']:
                    y_0 = np.array([
                        1 if str(answers[labels[i]]).lower() in preds[i].lower() or str(labels[i]) in preds[i] else 0
                        for i in range(len(preds))
                    ])
                elif model_name in ['MiniGPT4']:
                    y_0 = np.array([
                        1 if str(labels[i]) in preds[i] else 0
                        for i in range(len(preds))
                    ])
            elif dataset_name == 'OlympicLikeLogo':
                if model_name in ['LLaVA-7B']:#, 'mPLUG-Owl']:
                    y_0 = np.array([
                    1 if preds[i].split(',')[-1].strip().lower() in answers[labels[i]].lower() else 0
                    for i in range(len(preds))
                    ])
                elif model_name in ['LLaMA_Adapter','mPLUG-Owl']:
                    y_0 = np.array([
                        1 if answers[labels[i]].lower() in preds[i] or str(labels[i]).lower() in preds[i]  else 0
                        for i in range(len(preds))
                    ])
                elif model_name in ['MiniGPT4']:
                    y_0 = np.array([
                        1 if str(labels[i]).lower() in preds[i] else 0
                        for i in range(len(preds))
                    ])
            elif dataset_name == 'Triangle':
                if model_name in ['LLaVA-7B', 'mPLUG-Owl']:
                    y_0 = np.array([
                        1 if preds[i].split(',')[-1].strip().lower() in answers[labels[i]].lower() else 0
                        for i in range(len(preds))
                    ])
                    print([preds[i] for i in range(len(preds)) if y_0[i] == 1])
                elif model_name in ['LLaMA_Adapter']:
                    y_0 = np.array([
                        1 if answers[labels[i]].lower() in preds[i] or str(labels[i]).lower() in preds[i]  else 0
                        for i in range(len(preds))
                    ])
                elif model_name in ['MiniGPT4']:
                    y_0 = np.array([
                        1 if str(labels[i]).lower() in preds[i] else 0
                        for i in range(len(preds))
                    ])
            elif dataset_name =='Lines':
                if model_name in ['LLaVA-7B', 'MiniGPT4', 'LLaMA_Adapter', 'mPLUG-Owl']:#, 'mPLUG-Owl']:
                    y_0 = np.array([
                        1 if str(answers[labels[i]]).lower() in preds[i].lower() else 0
                        for i in range(len(preds))
                    ])
            return x_train, x_train_first, y_0
        elif 'MathV' in dataset_name:
            data = json.load(open(f"./output/{model_name}/MathV_output.json"))
            logits = read_jsonl(f"./output/{model_name}/MathV.jsonl")
            pred_list = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'].lower() else 0 for ins in logits]
            if '2' in dataset_name:
                x_train = np.array([ins['logits_se'] for ins in logits]).astype(np.float32)
                x_train_first = np.array([ins['logits_se'][0] for ins in logits]).astype(np.float32)
                y_0 = np.array([1 if data[str(i)]["true_false"] else 0 for i in range(1, 1001)])
                y_train = np.array([1 if y_i==p_i else 0 for y_i, p_i in zip(y_0, pred_list)])
                # Filter for label == 1
                return x_train, x_train_first, y_train
            else:
                x_train = np.array([ins['logits'] for ins in logits]).astype(np.float32)
                x_train_first = np.array([ins['logits'][0] for ins in logits]).astype(np.float32)
                y_train = np.array([1 if data[str(i)]["true_false"] else 0 for i in range(1, 1001)])
            return (x_train, x_train_first, y_train)
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
    print(dataset_name)
    X_0_attention, X_0, y_0 = get_dataset(model_name, dataset_name)
    # Count each unique label
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    def evaluate_model(model, X, y, model_name, save_path=None):
        """Evaluate a model using cross-validation and log results."""
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
    input_dim=32000 
    epochs=350
    batch_size=700
    lr=0.0000005
    header = [
        "Epochs",
        "Accuracy", "Accuracy Std",
        "F1", "F1 Std",
        "AUROC", "AUROC Std"
    ]

    csv_filename = f"./scratch_results/{dataset_name}_{model_name}_{method_name}.csv"
    # Extract metrics in required order: accuracy, f1, roc_auc
    accuracy_idx = 0
    f1_idx = 1
    roc_auc_idx = 2
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for epochs in [50, 75, 100, 150, 200, 250, 300, 350, 375, 400, 450, 500, 550, 600, 700, 1000]:
            print(f"\nEvaluating model with {epochs} epochs...")
            if 'tau' in method_name:
                model = AlphaModelWrapper(input_dim=input_dim, epochs=epochs, batch_size=batch_size, lr=lr, model_type=method_name)
            else:
                model = AttentionModelWrapper(input_dim=input_dim, epochs=epochs, batch_size=batch_size, lr=lr, model_type=method_name)
            metrics_mean, metrics_std = evaluate_model(model, X_0_attention, 1 - y_0, model_name)

            writer.writerow([
                epochs,
                metrics_mean[accuracy_idx], metrics_std[accuracy_idx],
                metrics_mean[f1_idx], metrics_std[f1_idx],
                metrics_mean[roc_auc_idx], metrics_std[roc_auc_idx],
            ])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python arith_all.py <model_name> <dataset_name> <method>")
    else:
        run_analysis(sys.argv[1],sys.argv[2],sys.argv[3])

