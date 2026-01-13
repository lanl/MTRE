
import os
import json
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.func import read_data_non_linear
from non_linear_notebooks.run_eval_type_clean import run_full_attention_eval 
import logging
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


def evaluate_mad_2(model_name="LLaVA-7B", prompt="oe", tau=False, logits_used=10, trained_model_path=None, config_list=None):
    dataset_name = "MAD"
    logging_path = f'./attention_results/{model_name}_{prompt}_{dataset_name}_type_II.log'
    csv_file = f'./attention_results/{model_name}_{prompt}_{dataset_name}_type_II.csv'
    if os.path.exists(logging_path):
        os.remove(logging_path)
    change_logging_config(logging_path)

    train_data, x_train, y_0_train, x_0_train = read_data_non_linear(model_name, dataset_name, split="train", 
                                    prompt=prompt, token_idx=-1, return_data=True, type_l='_se') # Returns keys_from_dict, x_II, label_I, x_I_first_logit

    val_data, x_val, y_0_val, x_0_val = read_data_non_linear(model_name, dataset_name, split="val", 
                                    prompt=prompt, token_idx=-1, return_data=True,type_l='_se', num_samples=1800) #_se = selfeval

    if prompt == "oe":
        val_labeled = json.load(open(f'./output/{model_name}/MAD_val_oe_labeled.jsonl')) #answers to first question
        train_labeled = json.load(open(f'./output/{model_name}/MAD_train_oe_labeled.jsonl'))
        pred_list_val = [0 if ins["is_answer"] == 'no' else 1 for ins in val_labeled]
        pred_list_train = [0 if ins["is_answer"] == 'no' else 1 for ins in train_labeled]
    elif prompt == "oeh":
        pred_list_val = [0 if ins['response'].startswith("Sorry, I cannot answer your question") else 1 for ins in val_data]
        pred_list_train = [0 if ins['response'].startswith("Sorry, I cannot answer your question") else 1 for ins in train_data]
    elif prompt == "mq":
        pred_list_val = [0 if "no" in ins['response'].lower() else 1 for ins in val_data]
        pred_list_train = [0 if "no" in ins['response'].lower() else 1 for ins in train_data]

    label_list_val = [ins['label'] for ins in val_data]
    label_list_train = [ins['label'] for ins in train_data]
    correctness_list_val = [1 if label==pred else 0 for label, pred in zip(label_list_val,pred_list_val)]
    correctness_list_train = [1 if label==pred else 0 for label, pred in zip(label_list_train,pred_list_train)]


    pred_list_val_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] else 0 for ins in val_data]
    pred_list_train_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] else 0 for ins in train_data]

    label_list_val_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_val,pred_list_val_2)]
    label_list_train_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_train,pred_list_train_2)]

    y_val = np.array(label_list_val_2) #Type II labels Was the model correct in determining if it was right or wrong?
    y_train = np.array(label_list_train_2) #Type II labels
    
    y_pred = np.array(pred_list_val_2)
    run_full_attention_eval(x_train, 1 - y_train, x_val, 1 - y_val, model_name, dataset_name, prompt, csv_file, 1 - y_pred, tau=tau, x_0_train=x_0_train, 
                            x_0_val=x_0_val, type_num='2', logits_used=logits_used, trained_model_path=trained_model_path, config_list=config_list)
   

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
model_name = 'MiniGPT4'
prompt = 'oe'
evaluate_mad_2(model_name=model_name, prompt=prompt, tau=False, logits_used=10, trained_model_path=None, config_list=config_list)