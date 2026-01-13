import os
import json

from datasets import load_dataset
class BaseDataset:
    def __init__(self): ...
#'Hint: Please answer the question requiring an answer and provide the correct response at the end.\nQuestion: How many intersection points are there? Zero, One, or Two?' MiniGPT4
class Lines(BaseDataset):
    def __init__(self, prompter, split="testmini", data_root="./data/Lines/", response_type='oe'):
        super(Lines, self).__init__()
        self.ann = json.load(open(f"./data/Lines/metadata.json"))
        self.img_root = data_root
        self.split = split
        self.prompter = prompter
        # self.true_split = true_split
        self.response_type = response_type
         
    def get_data(self):
        print(list(self.ann.values())[0])
        data = [
            {
                "img_path": os.path.join(self.img_root, f"{ins['pid']}.png"),
                "question": 'Question: There are _ intersecting lines.',#ins['query'],# "question": 'Hint: Please answer the question requiring an answer and provide the correct response at the end.\nQuestion: How many intersection points are there? Zero, One, or Two?',
                "label": ins['answer'],
                "original_question": ins['query'],
            }
            for ins in list(self.ann.values())  # Iterating over the values in the dictionary
        ]
        return data, ['pid']
