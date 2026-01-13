
import os
import json
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.func import read_data
import logging
from non_linear_notebooks.run_eval_type_clean import run_full_attention_eval 
import re
import csv


def change_logging_config(logging_path):
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Reconfigure logging
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO (use DEBUG for more verbosity)
        format='%(message)s',
        handlers=[
            logging.StreamHandler(),  # Print to console
            logging.FileHandler(logging_path, mode='a')  # Write to a file (append mode)
        ]
    )
    
def evaluate_mad_1(model_name="LLaVA-7B", prompt="oe", tau=False, logits_used=10, trained_model_path=None, config_list=None):
    dataset_name = 'MAD'
    logging_path = f'./attention_results/{model_name}_{prompt}_{dataset_name}_type_I.log'
    change_logging_config(logging_path)
    
    csv_file = f'./attention_results/{model_name}_{prompt}_{dataset_name}_type_I.csv'
    train_data, x_train, y_train = read_data(model_name, dataset_name, split="train", 
                                    prompt=prompt, token_idx=-1, return_data=True)
    val_data, x_val, y_val = read_data(model_name, dataset_name, split="val", 
                                    prompt=prompt, token_idx=-1, return_data=True)

    #VLM performance:
    if prompt == "oe":
        val_labeled = json.load(open(f'./output/{model_name}/MAD_val_oe_labeled.jsonl'))
        train_labeled = json.load(open(f'./output/{model_name}/MAD_train_oe_labeled.jsonl'))
        pred_list = np.array([0 if ins["is_answer"].lower() == 'no' else 1 for ins in val_labeled])
        pred_list_train = np.array([0 if ins["is_answer"].lower() == 'no' else 1 for ins in train_labeled])
    elif prompt == "oeh":
        pred_list = np.array([0 if ins['response'].startswith("Sorry, I cannot answer your question") else 1 for ins in val_data])
        pred_list_train = np.array([0 if ins['response'].startswith("Sorry, I cannot answer your question") else 1 for ins in train_data])
    elif prompt == "mq":
        pred_list = np.array([0 if "no" in ins['response'].lower() else 1 for ins in val_data])
        pred_list_train = np.array([0 if "no" in ins['response'].lower() else 1 for ins in train_data])
        
    y_val = np.array([1 if label==pred else 0 for label, pred in zip(y_val,pred_list)]) #Can we check if the vlm is correct
    y_train = np.array([1 if label==pred else 0 for label, pred in zip(y_train,pred_list_train)]) # Can we check if the vlm is correct

    y_pred=pred_list
    run_full_attention_eval(x_train, 1 - y_train, x_val, 1 - y_val, model_name, dataset_name, prompt, csv_file, 1 - y_pred, tau=tau, x_0_train=None, 
                            x_0_val=None, type_num='1', logits_used=logits_used, trained_model_path=trained_model_path, config_list=config_list)
   
# OPTIONALY: Get the file path passed as an argument
# temp_file_path = sys.argv[1]
# def process_csv_data(file_path):
#     configs = []
#     with open(file_path, newline='') as f:
#         reader = csv.DictReader(f)   # automatically skips the header
#         for row in reader:
#             configs.append({
#                 'embed_dim':  int(row['embed_dim']),
#                 'num_heads':  int(row['num_heads']),
#                 'num_layers': int(row['num_layers']),
#                 'dropout':    float(row['dropout']),
#                 'lr':         float(row['lr']),
#                 'csv_file':   os.path.basename(file_path)
#             })
#     prompt = configs[0]['csv_file'].split('_')[-4]
#     csv_file = configs[0]['csv_file']
#     if "mPLUG-Owl" in csv_file:
#         model_name = 'mPLUG-Owl'
#     elif "LLaVA-7B" in csv_file:
#         model_name = "LLaVA-7B"
#     elif "MiniGPT4" in csv_file:
#         model_name = 'MiniGPT4'
#     elif "LLaMA_Adapter" in csv_file:
#         model_name = "LLaMA_Adapter"
#     else:
#         model_name = None
#     return configs, prompt, model_name

# config_list, prompt, model_name = process_csv_data(temp_file_path)
config_list = [{'embed_dim':1024, 'num_heads':64, 'num_layers':2, 'dropout':0.5, 'epochs':100, 'lr':5e-7, 'log_reg':False}]
prompt = 'oe'
model_name = 'LLaVA-7B'
evaluate_mad_1(model_name=model_name, prompt=prompt, tau=False, logits_used=10, trained_model_path=None, config_list=config_list)
