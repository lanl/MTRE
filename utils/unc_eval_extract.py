import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from datasets import load_dataset
from utils.func import read_jsonl, softmax
from utils.metric import evaluate

# model_name = "MiniGPT4"
model_name = "LLaMA_Adapter"
data = read_jsonl(f"./output/{model_name}/MathV.jsonl")
print('Len of data',len(data))

dataset = load_dataset("./data/MathVista", split='testmini')
print('Length of dataset',len(dataset))

if not os.path.exists(f"./output/{model_name}/MathV_output.json"):
    res = {}
    for pid in range(len(dataset)):
        dic = dataset[pid]
        del dic['decoded_image']
        dic['response'] = data[pid]['response']
        res[pid+1] = dic

    json.dump(res, open(f"./output/{model_name}/MathV_output.json", 'w'))


print(f"""python extract_answer.py \\
    --output_dir "../../TowardsTrustworthy/output/{model_name}/" \\
    --output_file "MathV{fix}_output.json" \\
    --llm_engine "gpt-4-0125-preview" """)

print(f"""python calculate_score.py \\
    --output_dir "/data/qinyu/research/TowardsTrustworthy/output/{model_name}/" \\
    --output_file "MathV{fix}_output.json" """)