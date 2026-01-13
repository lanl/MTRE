import os
import json

from datasets import load_dataset

class BaseDataset:
    def __init__(self): ...

class MathVista(BaseDataset):
    def __init__(self, prompter, split="testmini", data_root="/data/MathVista/", response_type='oe'):
        super(MathVista, self).__init__()
        true_split = split.split("_")[0]
        if split == "testmini":
            # self.ann = load_dataset("AI4Math/MathVista")['testmini']
            self.ann = load_dataset("./data/MathVista")['testmini']
        else:
            self.ann = json.load(open(f"./output/{true_split}/MathV_output.json"))
        self.img_root = data_root
        self.split = split
        self.prompter = prompter
        self.true_split = true_split
        self.response_type = response_type
         
    def get_data(self):
        if self.split == "testmini":
            data = [
                {
                    'pid': ins['pid'],
                    "img_path": os.path.join(self.img_root, ins['image']),
                    "question": self.prompter.build_prompt(ins['query']),
                    "label": ins['answer'],
                    "original_question": ins['query'],
                }
                for ins in self.ann
            ]
        else:
            # data = [
            #     {
            #         'pid': self.ann[str(i)]['pid'],
            #         "img_path": os.path.join(self.img_root, self.ann[str(i)]['image']),
            #         "question": f"Given the image,\nthe query '{self.ann[str(i)]['query']}',\nand an answer '{self.ann[str(i)]['response']}.\nIs the answer correct? Please respond with 'Yes' or 'No'.",
            #         "label": 1 if self.ann[str(i)]['true_false'] else 0
            #     }
            #     for i in range(1, 1001)
            # ]
            data = [
                {
                    'pid': self.ann[str(i)]['pid'],
                    "img_path": os.path.join(self.img_root, self.ann[str(i)]['image']),
                    "question": f"Given the image,\nthe query '{self.ann[str(i)]['query']}',\nand an answer '{self.ann[str(i)]['response']}.\nIs the answer correct? Please explain, restrict your answer to 20 words.",
                    "label": 1 if self.ann[str(i)]['true_false'] else 0
                }
                for i in range(1, 1001)
            ]
        return data, ['pid']