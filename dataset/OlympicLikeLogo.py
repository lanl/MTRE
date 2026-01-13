import os
import json

from datasets import load_dataset

class BaseDataset:
    def __init__(self): ...

class OlympicLikeLogo(BaseDataset):
    def __init__(self, prompter, split="testmini", data_root="./data/OlympicLikeLogo/", response_type='oe'):
        super(OlympicLikeLogo, self).__init__()
        self.ann = json.load(open(f"./data/OlympicLikeLogo/metadata.json"))
        self.img_root = data_root
        self.split = split
        self.prompter = prompter
        # self.true_split = true_split
        self.response_type = response_type
         
    def get_data(self):
        print(list(self.ann.values())[0])
        data = [
            {
                #"img_path": os.path.join(self.img_root, f'{ins['pid']}.png'),
                "img_path": os.path.join(self.img_root, f"{ins['pid']}.png"),
                "question":  ins['query'],
                "label": ins['answer'],
                "original_question":  "Count the number of circles in the image."#ins['query'],
            }
            for ins in list(self.ann.values())  # Iterating over the values in the dictionary
        ]
        return data, ['pid']
