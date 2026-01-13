
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.func import clean_data, analyze_data_distribution, create_data_loaders, reshape_data
from utils.metric import evaluate, save_metrics_to_csv
from non_linear_notebooks.model_archs.attention_models_experimental import run_model_train_eval
import logging
import numpy as np

def run_full_attention_eval(
    x_train, y_train, x_val, y_val,
    model_name, dataset_name, prompt, csv_file, y_pred,
    tau=False, x_0_train=None, x_0_val=None,
    type_num='', logits_used=10, trained_model_path=None, config_list=None
):
    output_dir = './attention_results'

    # --- Helper Functions ---
    def preprocess(x_train_data, x_val_data, response_label=''):
        """Clean data in case of any corruption"""
        x_train_clean = clean_data(x_train_data)
        x_val_clean = clean_data(x_val_data)
        logging.info(analyze_data_distribution(x_train_clean, y_train, x_val_clean, y_val))
        return x_train_clean, x_val_clean

    def train_attention(x_train_data, x_val_data, response_label=''):
        """Reshape data, create loaders, and run attention training."""
        x_train_tok, y_train_tok, x_val_tok, y_val_tok = reshape_data(
            x_train_data, y_train, x_val_data, y_val, logits_used
        )
        train_loader, val_loader = create_data_loaders(
            x_train_tok, y_train_tok, x_val_tok, y_val_tok,
            batch_size=512, shuffle_train=True
        )
        input_dim = x_train_data.shape[-1]
        
        run_model_train_eval(
            input_dim, train_loader, val_loader,
            x_val_data, y_val, x_train_data, y_train,
            model_name, dataset_name, prompt, csv_file,
            config_list, device='cuda', early_stopping=logits_used,
            trained_model_path=trained_model_path,
            type_num=type_num, response=response_label, tau=tau
        )

    # --- Step 1: Initial Evaluation ---
    # acc, f1, auroc = evaluate(y_val, y_pred, show=True, save_location=logit_dir)
    # save_metrics_to_csv(csv_file, model_name, {'acc': acc, 'ap': ap, 'f1': f1, 'auroc': auroc})
    logging.info(f"Shapes -> Train: {x_train.shape}, Val: {x_val.shape}")

    # --- Step 2: Matched types ---
    x_train, x_val = preprocess(x_train, x_val)
    train_attention(x_train, x_val)

    # -- First Response if wanted. --
    # if x_0_train is not None:
    #     logging.info("##### Running First Response Attention #####")
    #     x_0_train, x_0_val = preprocess_and_train_baseline(x_0_train, x_0_val, response_label='first')
    #     train_attention(x_0_train, x_0_val, response_label='first')