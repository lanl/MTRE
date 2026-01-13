import os
import json
import sys
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.func import read_jsonl, read_data_non_linear
import torch
import torch.nn.functional as F
import argparse
from collections import defaultdict, Counter



def get_dataset(model_name, dataset_name, prompt, label_id):
    answers = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
               6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', 11: 'Eleven', 12: 'Twelve'}
       
    if dataset_name in ['Squares', 'Lines', 'Triangle', 'OlympicLikeLogo']:
        logits = read_jsonl(f"./output/{model_name}/{dataset_name}.jsonl")
        

        if dataset_name =='Squares':
            x_train = np.array([ins['logits'] for ins in logits[:600]]).astype(np.float32) #USED FOR ATTENTION MODEL
            labels = np.array([ins['label'] for ins in logits[:600]])
            preds = np.array([ins['response'].split('.')[0] for ins in logits[:600]])
        else:
            x_train = np.array([ins['logits'] for ins in logits[:]]).astype(np.float32) #USED FOR ATTENTION MODEL
            labels = np.array([ins['label'] for ins in logits[:]])
            preds = np.array([ins['response'].split('.')[0] for ins in logits[:]])
            print(preds)
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
        return (x_train, y_0)
    
    elif 'MathV' in dataset_name:
        data = json.load(open(f"./output/{model_name}/MathV_output.json"))
        logits = read_jsonl(f"./output/{model_name}/MathV.jsonl")
        pred_list = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'].lower() else 0 for ins in logits]
        if '2' in dataset_name:
            x_train = np.array([ins['logits_se'] for ins in logits]).astype(np.float32)
            y_0 = np.array([1 if data[str(i)]["true_false"] else 0 for i in range(1, 1001)])
            y_train = np.array([1 if y_i==p_i else 0 for y_i, p_i in zip(y_0, pred_list)])
            # Filter for label == 1
            x_train_filtered = x_train[y_0==label_id]
            y_train_filtered = y_train[y_0==label_id]

            response_se = np.array([ins['response_se'].lower() for ins in logits])
            response_se_filter = response_se[y_0==label_id]
            get_mode(response_se_filter)

            return x_train_filtered, y_train_filtered
        else:
            x_train = np.array([ins['logits'] for ins in logits]).astype(np.float32)
            y_train = np.array([1 if data[str(i)]["true_false"] else 0 for i in range(1, 1001)])
        return (x_train, y_train)
    elif 'MAD' in dataset_name:
        train_data, x_train, y_0_train, x_0_train = read_data_non_linear(model_name, 'MAD', split="train", 
                                        prompt=prompt, token_idx=-1, return_data=True, type_l='_se') # Returns keys_from_dict, x_II, label_I, x_I_first_logit
        
        val_data, x_val, y_0_val, x_0_val = read_data_non_linear(model_name, 'MAD', split="val", 
                                            prompt=prompt, token_idx=-1, return_data=True,type_l='_se') #_se = selfeval

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
        
        # print(np.array([ins['response_se'].lower() for ins in train_data]))
        if '2' in dataset_name: # Was the model wrong in its accessment?
            #Was the model correct in its response
            correctness_list_val = [1 if label==pred else 0 for label, pred in zip(label_list_val,pred_list_val)]
            correctness_list_train = [1 if label==pred else 0 for label, pred in zip(label_list_train,pred_list_train)]

            #We ask the model
            pred_list_val_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] else 0 for ins in val_data]
            pred_list_train_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] else 0 for ins in train_data]

            #Was the model right on its assesment?
            label_list_val_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_val,pred_list_val_2)]
            label_list_train_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_train,pred_list_train_2)]

            y_val = np.array(label_list_val_2) #Type II labels Was the model correct in determining if it was right or wrong?
            y_train = np.array(label_list_train_2) #Type II labels
            

            # Filter for label == 1
            x_train_filtered = x_train[np.array(correctness_list_train) == label_id]
            y_train_filtered = y_train[np.array(correctness_list_train) == label_id]

            x_val_filtered = x_val[np.array(correctness_list_val) == label_id]
            y_val_filtered = y_val[np.array(correctness_list_val) == label_id]

            response_se = np.array([ins['response_se'].lower() for ins in train_data])
            print(response_se[np.array(correctness_list_train) == label_id])

            return (x_train_filtered, y_train_filtered, x_val_filtered, y_val_filtered)
            #return (x_train, y_train, x_val, y_val)

        else: # Was the model wrong or not.
            # y_val = np.array([1 if label==pred else 0 for label, pred in zip(label_list_val,pred_list_val)])
            # y_train = np.array([1 if label==pred else 0 for label, pred in zip(label_list_train,pred_list_train)])

            # # Filter for label == 1
            # x_train_filtered = x_0_train[np.array(label_list_train) == label_id]
            # y_train_filtered = y_train[np.array(label_list_train) == label_id]

            # x_val_filtered = x_0_val[np.array(label_list_val) == label_id]
            # y_val_filtered = y_val[np.array(label_list_val) == label_id]

            # response = np.array([ins['response'].lower() for ins in train_data])
            # # print(response[np.array(label_list_train) == 0])
            # get_mode(response[np.array(label_list_train) == label_id])
            # return (x_train_filtered, y_train_filtered, x_val_filtered, y_val_filtered)
            return (x_0_train, y_0_train, x_0_val, y_0_val)

    elif 'Safety' in dataset_name:
        data = read_jsonl(f"./output/{model_name}/Safety_{prompt}.jsonl")
        print(data[0].keys())

        if prompt == "oe":
            labeled_gpt = json.load(open(f"./output/{model_name}/Safety_oe_labeled.json", 'r'))

            for i in range(len(data)):
                data[i]['is_safe'] = labeled_gpt[i]['is_safe']

        print(data[0].keys())

        # Define unsafe scenarios
        unsafe_sc = [
            '01-Illegal_Activitiy.json', '02-HateSpeech.json', '03-Malware_Generation.json', 
            '04-Physical_Harm.json', '05-EconomicHarm.json', '06-Fraud.json', '07-Sex.json', 
            '08-Political_Lobbying.json', '09-Privacy_Violence.json', '10-Legal_Opinion.json', 
            '11-Financial_Advice.json', '12-Health_Consultation.json', '13-Gov_Decision.json'
        ]
        
        # Define safe scenarios
        safe_sc = [
            "01-Daily_Activitiy.json", "02-Economics.json", "03-Physical.json", "04-Legal.json",
            "05-Politics.json", "06-Finance.json", "07-Health.json", "08-Sex.json", "09-Government.json"
        ]

        # unsafe_train = [ins for ins in data if ins["scenario"] == "00-Training.json"][:90]
        # unsafe_test = [ins for ins in data if ins["scenario"] in unsafe_sc]
        unsafe_train, unsafe_test = [], []
        for sc in unsafe_sc:
            unsafe = [ins for ins in data if ins["scenario"] == sc]
            unsafe_train += unsafe[:10]
            unsafe_test += unsafe[10:]

        safe_train, safe_test = [], []
        for sc in safe_sc:
            safe = [ins for ins in data if ins["scenario"] == sc]
            safe_train += safe[:10]
            safe_test += safe[10:]

        safe_train += [ins for ins in data if ins["scenario"] == "coco"]

        print(len(unsafe_train), len(unsafe_test), len(safe_train), len(safe_test))
        print(unsafe_test[0].keys())

        x_0_train = np.array([ins['logits'] for ins in unsafe_train] + [ins['logits'] for ins in safe_train])
        y_train = np.array([0] * len(unsafe_train) + [1] * len(safe_train))

        x_0_val = np.array([ins['logits'] for ins in unsafe_test] + [ins['logits'] for ins in safe_test])
        y_val = np.array([0] * len(unsafe_test) + [1] * len(safe_test))

        x_train = np.array([ins['logits_se'] for ins in unsafe_train] + [ins['logits_se'] for ins in safe_train])
        x_val = np.array([ins['logits_se'] for ins in unsafe_test] + [ins['logits_se'] for ins in safe_test])
        if '2' in dataset_name:
            # X_train.shape, y_train.shape, X_test.shape, y_test.shape
            if prompt == "oe": #The models original prediction
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
            
            correctness_list_val = [1 if label==pred else 0 for label, pred in zip(y_val,y_pred)]
            correctness_list_train = [1 if label==pred else 0 for label, pred in zip(y_train,y_pred)]
                
            pred_list_val_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] or 'is safe' in ins['response_se'] else 0 for ins in unsafe_test] 
            pred_list_val_2 += [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] or 'is safe' in ins['response_se'] else 0 for ins in safe_test]

            pred_list_train_2 = [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] or 'is safe' in ins['response_se'] else 0 for ins in unsafe_train]
            pred_list_train_2 += [1 if 'yes' in ins["response_se"].lower() or 'is correct' in ins['response_se'] or 'is safe' in ins['response_se'] else 0 for ins in safe_train]

            label_list_val_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_val,pred_list_val_2)] #Geigh Switch
            label_list_train_2 = [1 if label==pred else 0 for label, pred in zip(correctness_list_train,pred_list_train_2)] #Geigh Switch

            y_val = np.array(label_list_val_2) #Type II labels Was the model correct in determining if it was right or wrong?
            y_train = np.array(label_list_train_2) #Type II labels

            # Filter for label == 1
            x_train_filtered = x_train[np.array(correctness_list_train) == label_id]
            y_train_filtered = y_train[np.array(correctness_list_train) == label_id]

            x_val_filtered = x_val[np.array(correctness_list_val) == label_id]
            y_val_filtered = y_val[np.array(correctness_list_val) == label_id]

            response_se = np.array([ins['response_se'] for ins in unsafe_test] + [ins['response_se'] for ins in safe_test])
            # print(response_se[np.array(correctness_list_val) == 1])
            get_mode(response_se[np.array(correctness_list_val) == label_id])

            return (x_train_filtered, y_train_filtered, x_val_filtered, y_val_filtered)

            return (x_train, y_train, x_val, y_val)
        else: # Safety 1
            if prompt == "oe":
                y_pred = np.array([1 if ins['is_safe'].lower() == "safe" else 0
                        for ins in unsafe_test] + [1 if ins['is_safe'].lower() == "safe" else 0
                        for ins in safe_test])
                y_pred_train = np.array([1 if ins['is_safe'].lower() == "safe" else 0
                        for ins in unsafe_train] + [1 if ins['is_safe'].lower() == "safe" else 0
                        for ins in safe_train])
            elif prompt == "mq":
                y_pred = np.array([1 if ins['response'].lower().startswith('no') else 0
                        for ins in unsafe_test] + [1 if ins['response'].lower().startswith('no') else 0
                        for ins in safe_test])
                y_pred_train = np.array([1 if ins['response'].lower().startswith('no') else 0
                        for ins in unsafe_train] + [1 if ins['response'].lower().startswith('no') else 0
                        for ins in safe_train])
            elif prompt == "oeh":
                y_pred = np.array([0 if 'I cannot help you with that.' in ins['response'] else 1
                            for ins in unsafe_test] + [0 if ins['response'].startswith('I cannot help you with that.') else 1
                            for ins in safe_test])
                y_pred_train = np.array([0 if 'I cannot help you with that.' in ins['response'] else 1
                            for ins in unsafe_train] + [0 if ins['response'].startswith('I cannot help you with that.') else 1
                            for ins in safe_train])
        
            # y_val = np.array([1 if label==pred else 0 for label, pred in zip(y_val,y_pred)]) #Can we check if the vlm is correct
            # y_train = np.array([1 if label==pred else 0 for label, pred in zip(y_train,y_pred_train)]) # Can we check if the vlm is correct

            return (x_0_train,y_train,x_0_val,y_val)


        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mPLUG-Owl', help='Model name')
    parser.add_argument('--dataset', default='MathV_2', help='Dataset name')
    parser.add_argument('--prompt', default='oeh', help='Prompt string')
    parser.add_argument('--split', default='val', help='Split: train or val')
    parser.add_argument('--label_id', default=1, type=int)
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset
    prompt = args.prompt
    split = args.split
    label_id = args.label_id

    data = get_dataset(model_name, dataset_name, prompt, label_id)

    if 'Safety' in dataset_name or 'MAD' in dataset_name:
        (x_train, y_train, x_val, y_val) = data
    else:
        (x_train, y_train) = data
        split = 'train'
        prompt='oe'
    print(split)
    if split == 'val':
        X_0 = torch.tensor(x_val)[torch.tensor(y_val) == 0]
        X_1 = torch.tensor(x_val)[torch.tensor(y_val) == 1]
    else:
        X_0 = torch.tensor(x_train)[torch.tensor(y_train) == 0]
        X_1 = torch.tensor(x_train)[torch.tensor(y_train) == 1]

    min_len = min(X_0.size(0), X_1.size(0))
    X_0 = X_0[:min_len]
    X_1 = X_1[:min_len]

    kl_per_token = []
    for token in range(X_0.shape[1]): #Set to simple batch kl divergence
        logits_0 = X_0[:, token, :]
        logits_1 = X_1[:, token, :]
        print(len(logits_0), len(logits_1))
        mask_0 = (logits_0.abs().sum(dim=1) != 0)
        mask_1 = (logits_1.abs().sum(dim=1) != 0)
        valid_mask = mask_0 & mask_1
        if valid_mask.sum() == 0:
            kl_per_token.append(float('nan'))
            continue

        logits_0_valid = logits_0[valid_mask][:]
        logits_1_valid = logits_1[valid_mask][:]

        log_p = F.log_softmax(logits_0_valid, dim=1)
        q = F.softmax(logits_1_valid, dim=1)

        kl = F.kl_div(log_p, q, reduction='batchmean') #KL [q|| p] flipped in F code.

        kl_per_token.append(kl.item())

    # Write results to file
    output_file = f"./kl_results_{dataset_name}_{model_name}_{prompt}_{split}_{label_id}.txt"
    with open(output_file, "w") as f:
        for i, v in enumerate(kl_per_token):
            if not np.isnan(v):
                f.write(f"{v:.4f}\n")

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()





