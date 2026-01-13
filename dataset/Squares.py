import os
import json

from datasets import load_dataset

class BaseDataset:
    def __init__(self): ...
#mPLUG-Owl: 'Count the number of squares.'
#LLaMA_Adapter: 'Count the number of nested squares that you can see.'
#MiniGPT4: 'Count the number of nested squares that you can see, hint: there are at least 2 and no more than 5.'
#LLaVA-7B: 'How many nested squares are there?'
class Squares(BaseDataset):
    def __init__(self, prompter, split="testmini", data_root="./data/Squares/", response_type='oe'):
        super(Squares, self).__init__()
        self.ann = json.load(open(f"./data/Squares/metadata.json"))
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
                "question":  ins['query'],#'Count the number of nested squares that you can see.',
                "label": ins['answer'],
                "original_question": ins['query'] #'Count the number of nested squares that you can see.',
            }
            for ins in list(self.ann.values())  # Iterating over the values in the dictionary
        ]
        return data, ['pid']