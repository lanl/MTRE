"""
Utility Functions for MTRE

This module provides core utility functions for:
- Data loading and preprocessing (JSONL files, logits)
- Evaluation metrics computation (accuracy, AUROC, F1)
- Cross-validation utilities
- PyTorch Dataset/DataLoader creation
- Data cleaning and reshaping

Key Functions:
    - read_jsonl(): Load JSONL data files
    - read_data(): Load and preprocess logits for evaluation
    - read_data_non_linear(): Load multi-token logits for MTRE
    - eval_scores_p_true(): Evaluate P(True) baseline with cross-validation
    - create_data_loaders(): Create PyTorch DataLoaders
    - clean_data(): Handle NaN/Inf values in logit arrays
"""

import re
import json
import math
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.utils import resample
import pandas as pd
import os
import csv


# =============================================================================
# Evaluation Functions
# =============================================================================

def eval_scores_p_true(y_0, y_probs, model_name=None, csv_filename="metrics_summary_simple_sar.csv"):
    """
    Evaluate P(True) baseline using 4-fold stratified cross-validation.

    Finds optimal threshold on training fold, evaluates on test fold.
    Reports accuracy, F1, and AUROC averaged across folds.

    Args:
        y_0 (np.ndarray): Ground truth labels (0 or 1)
        y_probs (np.ndarray): Predicted probabilities/scores
        model_name (str, optional): If provided, appends results to CSV
        csv_filename (str): Path to output CSV file

    Returns:
        float: Average AUROC across folds
    """
    cv = StratifiedKFold(n_splits=4)
    # models = []
    accs = []
    f1s = []
    aucs = []
    precs = []
    for train_idx, test_idx in cv.split(y_probs, y_0):
        prob_train, y_train = y_probs[train_idx], y_0[train_idx]
        prob_test, y_test = y_probs[test_idx], y_0[test_idx]

        #Train
        fpr, tpr, thresholds = roc_curve(y_train, prob_train)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        #Test
        y_pred_test = (prob_test >= optimal_threshold).astype(int)
        auroc_p_true = roc_auc_score(y_test, prob_test)
        acc = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        # precision = precision_score(y_test, y_pred_test)
        # print(acc)
        accs.append(acc)
        f1s.append(f1)
        aucs.append(auroc_p_true)
        # precs.append(precision)

    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1s)
    avg_auc = np.mean(aucs)
    # avg_prec = np.mean(precs)

    std_acc = np.std(accs)
    std_f1 = np.std(f1s)
    std_auc = np.std(aucs)

    if model_name:
        # csv_filename = "metrics_summary_simple_sar.csv"
        # Check if file exists and is non-empty
        file_exists = os.path.isfile(csv_filename)
        is_empty = not file_exists or os.path.getsize(csv_filename) == 0
        # Open in append mode
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write header if file is new or empty
            if is_empty:
                writer.writerow(["Model Name","Acc","Acc_std","Auc","Auc_std","F1","F1_std"])

            # Append metrics row
            writer.writerow([
                model_name,
                f"{avg_acc * 100:.3f}",
                f"{std_acc * 100:.3f}",
                f"{avg_auc * 100:.3f}",
                f"{std_auc * 100:.3f}",
                f"{avg_f1 * 100:.3f}",
                f"{std_f1 * 100:.3f}",
            ])

        print(f"Appended metrics to {csv_filename}")
    # std_prec = np.std(precs)
    # File name


    
    print(f"Average AUC: {avg_auc*100:.3f}")
    print(f"Standard Deviation AUC: {std_auc*100:.3f}")
    print(f"Average Accuracy across thresholds: {avg_acc*100:.3f}")
    print(f"Standard Deviation Accuracy across thresholds: {std_acc*100:.3f}")
    print(f"Average F1 Score across thresholds: {avg_f1*100:.3f}")
    print(f"Standard Deviation F1 Score across thresholds: {std_f1*100:.3f}")


    return avg_auc

def read_data_surprise(dataset_name, prompt):
    if dataset_name == 'MAD':
        _, x_train, y_train = read_data(model_name, dataset_name, split="train", 
                                        prompt=prompt, token_idx=-1)
        val_data, x_val, y_val = read_data(model_name, dataset_name, split="val", 
                                        prompt=prompt, token_idx=-1, return_data=True)
        
    return x_train, y_train



def eval_scores_p_true_manual(y_train, y_test, prob_train, prob_test, model_name=None, csv_filename="metrics_summary_simple.csv"):
    #Train
    fpr, tpr, thresholds = roc_curve(y_train, prob_train)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    #Test
    y_pred_test = (prob_test >= optimal_threshold).astype(int)
    auroc_p_true = roc_auc_score(y_test, prob_test)
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    print(f"Average AUC: {auroc_p_true*100:.3f}")
    print(f"Average Accuracy P(True): {acc*100:.3f}")
    print(f"Average F1 Score across thresholds: {f1*100:.3f}")
    # Open the existing ODS file
    # Open the existing ODS document
    # File name

    if model_name:
        # csv_filename = "metrics_summary_simple.csv"
        # Check if file exists and is non-empty
        file_exists = os.path.isfile(csv_filename)
        is_empty = not file_exists or os.path.getsize(csv_filename) == 0
        # Open in append mode
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write header if file is new or empty
            if is_empty:
                writer.writerow(["Model Name","Acc", "Auc", "F1"])

            # Append metrics row
            writer.writerow([
                model_name,
                f"{acc * 100:.3f}",
                f"{auroc_p_true * 100:.3f}",
                f"{f1 * 100:.3f}"
            ])

        print(f"Appended metrics to {csv_filename}")

# # Assuming X_0 and y_0 are already defined (your features and target)
def eval_scores_manual(y_train, y_test, X_train, X_test, model):
    # Initialize the logistic regression model
 
    model.fit(X_train, y_train)

    # Predict class labels
    y_pred = model.predict(X_test)

    # Predict probabilities (needed for AUC)
    y_proba = model.predict_proba(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_proba[:, 1])

    return auc, acc, f1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True) # only difference


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_conf_score(text="The price of the item is 12.99 dollars."):
    # Regular expression pattern to find a float number
    pattern = r"\b\d+\.\d+\b|\b\d+\b"

    # Search for the pattern in the text
    match = re.search(pattern, text)

    # Extract the matched float number, if any
    float_number = match.group() if match else 0.0
    return float(float_number)


def read_jsonl(file, num=None):
    with open(file, 'r') as f:
        i = 0
        data = []
        
        for line in tqdm(f):
            i += 1
            data.append(json.loads(line))
            
            if num and i == num:
                break

    return data

def read_jsonl_safe(file, num=None):

    categories = [
                "01-Daily_Activitiy",
                "02-Economics",
                "03-Physical",
                "04-Legal",
                "05-Politics",
                "06-Finance",
                "07-Health",
                "08-Sex",
                "09-Government",
            ]
    scenario_list_unsafe = ['01-Illegal_Activitiy', '02-HateSpeech', '03-Malware_Generation', '04-Physical_Harm',
                        '05-EconomicHarm', '06-Fraud', '07-Sex', '08-Political_Lobbying',
                        '09-Privacy_Violence', '10-Legal_Opinion', '11-Financial_Advice',
                        '12-Health_Consultation', '13-Gov_Decision']
    #train_data/imgs/
    category_dict = {category: "safe_data/imgs/" for category in categories}
    category_dict.update({scenario: "data/imgs/" for scenario in scenario_list_unsafe})
    category_dict.update({'00-Training': "train_data/imgs/"})

    with open(file, 'r') as f:
        i = 0
        data = []
        
        for line in tqdm(f):
            i += 1
            ins = json.loads(line)
            # print(ins.keys())
            clean_sc = ins['scenario'].removesuffix(".json")
            data_point = {
                'image': ins['image'], 
                'image_path':category_dict[clean_sc]+clean_sc,
                'scenario': ins['scenario'],
                'question': ins['question'], 
                'response': ins['response'], 
                'response_se': ins['response_se'], 
                'model_name': ins['model_name'],
                'label': ins['label']
            }
            data.append(data_point)
            
            if num and i == num:
                break
        # print(data)
    return data

def read_jsonl_mad(file, num=None):

    with open(file, 'r') as f:
        i = 0
        data = []
        
        for line in tqdm(f):
            i += 1
            ins = json.loads(line)

            data_point = {
                'image': ins['image'],
                'scenario': ins['scenario'], 
                'question': ins['question'], 
                'response': ins['response'], 
                'response_se': ins['response_se'], 
                'model_name': ins['model_name'],
                'label': ins['label']
            }
            data.append(data_point)
            
            if num and i == num:
                break
        # print(data)
    return data

def read_and_update_jsonl(file, new_key, new_value_func, num=None):
    updated_data = []
    with open(file, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            if num and i == num:
                break
            item = json.loads(line)
            item[new_key] = new_value_func(item)  # Dynamically add new key using a function
            updated_data.append(item)
    return updated_data

def read_data(model, dataset, split="val", prompt='oe', token_idx=0, return_data=False, num_samples=None):
    x, y = [], []
    if dataset == 'MathV':
        data = json.load(open(f"./output/{model}/MathV_output.json")) #Used for answers
        logits = read_jsonl(f"./output/{model}/MathV.jsonl")
        X = np.array([ins['logits'] for ins in logits])
        y = np.array([1 if data[str(i)]["true_false"] else 0 for i in range(1, 1001)])
    
    elif dataset == 'Safety':
        data = read_jsonl(f"./output/{model}/Safety_{prompt}.jsonl")
        print(data[0].keys())

        if prompt == "oe":
            labeled_gpt = json.load(open(f"./output/{model}/Safety_oe_labeled.json", 'r'))
            
            for i in range(len(data)):
                data[i]['is_safe'] = labeled_gpt[i]['is_safe']

        # print(data[0].keys())

        unsafe_sc = ['01-Illegal_Activitiy.json', '02-HateSpeech.json', '03-Malware_Generation.json', '04-Physical_Harm.json',
                            '05-EconomicHarm.json', '06-Fraud.json', '07-Sex.json', '08-Political_Lobbying.json',
                            '09-Privacy_Violence.json', '10-Legal_Opinion.json', '11-Financial_Advice.json',
                            '12-Health_Consultation.json', '13-Gov_Decision.json']
        # unsafe_train = [ins for ins in data if ins["scenario"] == "00-Training.json"][:90]
        # unsafe_test = [ins for ins in data if ins["scenario"] in unsafe_sc]
        unsafe_train, unsafe_test = [], []
        for sc in unsafe_sc:
            unsafe = [ins for ins in data if ins["scenario"] == sc]
            unsafe_train += unsafe[:10]
            unsafe_test += unsafe[10:]

        safe_sc = ["01-Daily_Activitiy.json",
                    "02-Economics.json",
                    "03-Physical.json",
                    "04-Legal.json",
                    "05-Politics.json",
                    "06-Finance.json",
                    "07-Health.json",
                    "08-Sex.json",
                    "09-Government.json"]
        safe_train, safe_test = [], []
        for sc in safe_sc:
            safe = [ins for ins in data if ins["scenario"] == sc]
            safe_train += safe[:10]
            safe_test += safe[10:]
        safe_train += [ins for ins in data if ins["scenario"] == "coco"]

        # print(len(unsafe_train), len(unsafe_test), len(safe_train), len(safe_test))
        # print(unsafe_test[0].keys())

        X_train = np.array([ins['logits'] for ins in unsafe_train] + [ins['logits'] for ins in safe_train])
        y_train = np.array([0] * len(unsafe_train) + [1] * len(safe_train))

        X_test = np.array([ins['logits'] for ins in unsafe_test] + [ins['logits'] for ins in safe_test])
        y_test = np.array([0] * len(unsafe_test) + [1] * len(safe_test))
        # X_train.shape, y_train.shape, X_test.shape, y_test.shape
        if prompt == "oe":
            y_pred = np.array([1 if ins['is_safe'].lower() == "safe" else 0
                    for ins in unsafe_test] + [1 if ins['is_safe'].lower() == "safe" else 0
                    for ins in safe_test])
        elif prompt == "mq":
            y_pred = np.array([1 if ins['response'].lower().startswith('no') else 0
                    for ins in unsafe_test] + [1 if ins['response'].lower().startswith('no') else 0
                    for ins in safe_test])
        elif prompt == "oeh":
            y_pred = np.array([0 if 'I cannot help you with that.' in ins['response'] else 1
                        for ins in unsafe_test] + [0 if ins['response'].startswith('I cannot help you with that.') else 1
                        for ins in safe_test])

        res = eval_safety(unsafe_test, y_pred)
        if split == "val":
            return [], np.squeeze(X_test), np.squeeze(y_test)
        elif split == 'train':
             return [], np.squeeze(X_train), np.squeeze(y_train)
        else:
            print('Split Not found')
            return 

    with open(f"./output/{model}/{dataset}_{split}_{prompt}.jsonl") as f:
        dataset = []
        for line in tqdm(f):
            data = json.loads(line)
            if return_data:
                dataset.append(data)
#             if np.isnan(np.mean(data['logits'])):
#                 continue
            if token_idx >= 0:
                x.append(data['logits'][token_idx])
            else:
                x.append(data['logits']) # Get the entire thing
            y.append(data['label'])
            if num_samples is not None and len(x) > num_samples:
                break
        # print(line[:1000])
    x, y = np.array(x), np.array(y)
    print(x.shape, y.shape)
    return dataset, np.squeeze(x), np.squeeze(y)

def read_data_raw(model, dataset, split="val", prompt='oe', token_idx=0, return_data=False, num_samples=None):
    x, y = [], []
    if dataset == 'MathV':
        data = json.load(open(f"./output/{model}/MathV_output.json")) #Used for answers
        logits = read_jsonl(f"./output/{model}/MathV.jsonl")
        X = np.array([ins['logits'] for ins in logits])
        y = np.array([1 if data[str(i)]["true_false"] else 0 for i in range(1, 1001)])
    
    elif dataset == 'Safety':
        data = read_jsonl(f"./output/{model}/Safety_{prompt}.jsonl")
        print(data[0].keys())


    with open(f"./output/{model}/{dataset}_{split}_{prompt}.jsonl") as f:
        dataset = []
        for line in tqdm(f):
            data = json.loads(line)
            if return_data:
                dataset.append(data)
#             if np.isnan(np.mean(data['logits'])):
#                 continue
            if token_idx >= 0:
                x.append(data['logits'][token_idx])
            else:
                x.append(data['logits']) # Get the entire thing
            y.append(data['label'])
            if num_samples is not None and len(x) > num_samples:
                break
        print(line[:1000])
    x, y = np.array(x), np.array(y)
    print(x.shape, y.shape)
    return dataset, np.squeeze(x), np.squeeze(y)


def read_data_non_linear(model, dataset, split="val", prompt='oe', token_idx=-1, return_data=False, num_samples=None, type_l='', keys_to_keep=None):
    x, y = [], []
    x_0 = []
    if keys_to_keep == None:
        keys_to_keep = ['response', 'response_se', 'label', 'p_true', 'output_ids', 'question']  # Replace with the keys you want
    with open(f"./output/{model}/{dataset}_{split}_{prompt}.jsonl") as f:
        dataset = []
        for line in tqdm(f):
            data = json.loads(line)
            if return_data:
                filtered_data = {key: data[key] for key in keys_to_keep if key in data}
                dataset.append(filtered_data)

            if token_idx >= 0:
                x.append(data['logits'+type_l][token_idx])
            else:
                x.append(data['logits'+type_l]) # Get the entire set
                x_0.append(data['logits'])

            y.append(data['label'])
            
            if num_samples is not None and len(x) > num_samples:
                break
    x, y, x_0 = np.array(x), np.array(y), np.array(x_0)
    print(x.shape, y.shape, x_0.shape)
    return dataset, np.squeeze(x), np.squeeze(y), np.squeeze(x_0)


def clean_data(data):
    """Clean data by replacing NaN and Inf values."""
    if np.any(np.isnan(data)):
        print("Data contains NaN values.")
        data = np.nan_to_num(data, nan=0.0)
    
    if np.any(np.isinf(data)):
        print("Data contains Infinity values.")
        inf_indices = np.where(np.isinf(data))
        print(f"Number of infinity values: {len(inf_indices[0])}")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    return data

class BinaryDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Shape: (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(x_train, y_train, x_val, y_val, batch_size=128, shuffle_train=True):
    """Create DataLoader objects for training and validation."""
    train_dataset = BinaryDataset(x_train, y_train)
    val_dataset = BinaryDataset(x_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def reshape_data(x_train, y_train, x_val, y_val, logits_used):
    # Reshape x_train and x_val
    x_train = x_train[:, :logits_used, :]  
    x_train_token = x_train.reshape(-1, x_train.shape[2])
    x_val = x_val[:, :logits_used, :]
    x_val_token = x_val.reshape(-1, x_val.shape[2])

    # Repeat y_train and y_val
    y_train_token = np.repeat(y_train, logits_used)
    y_val_token = np.repeat(y_val, logits_used)
    

    # Print shapes to verify
    print("x_train_token shape:", x_train_token.shape)  # (205230, 32000)
    print("y_train_token shape:", y_train_token.shape)  # (205230,)
    print("x_val_token shape:", x_val_token.shape)      # (43190, 32000)
    print("y_val_token shape:", y_val_token.shape)      # (43190,
    return x_train_token, y_train_token, x_val_token, y_val_token

def analyze_data_distribution(x_train, y_train, x_val, y_val):
    """Analyze data distribution to identify potential issues."""
    # Check class distribution
    train_class_dist = np.unique(y_train, return_counts=True)
    val_class_dist = np.unique(y_val, return_counts=True)
    
    # Check for zero vectors
    zero_vector_mask = np.all(x_train == 0, axis=2)
    zero_vectors_per_sample = np.sum(zero_vector_mask, axis=1)
    average_zero_vectors_per_sample_train = np.mean(zero_vectors_per_sample)
    
    zero_vector_mask = np.all(x_val == 0, axis=2)
    zero_vectors_per_sample = np.sum(zero_vector_mask, axis=1)
    average_zero_vectors_per_sample_val = np.mean(zero_vectors_per_sample)
    
    distribution_info = {
        "train_class_distribution": dict(zip(train_class_dist[0], train_class_dist[1])),
        "val_class_distribution": dict(zip(val_class_dist[0], val_class_dist[1])),
        "avg_zero_vectors_train": average_zero_vectors_per_sample_train,
        "avg_zero_vectors_val": average_zero_vectors_per_sample_val
    }
    
    return distribution_info
