import os
import json

from datasets import load_dataset

class BaseDataset:
    def __init__(self): ...
# Count the triangles in this image. Respond by counting them out loud, in the format: One, Two, Three, etc
#Count the number of triangles in this image. Respond in the format: One, Two, etc. Restrict your response to only numbers.
#MiniGPT4: "How many triangles are in this image? 3 or 4?",#
class Triangle(BaseDataset):
    def __init__(self, prompter, split="testmini", data_root="./data/Triangle/", response_type='oe'):
        super(Triangle, self).__init__()
        self.ann = json.load(open(f"./data/Triangle/metadata.json"))
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
                "question": ins['query'],
                "label": ins['answer'],
                "original_question":  "How many triangles can you identify in this image? Are there 3, 4, or a different number?"#ins['query'],
            }
            for ins in list(self.ann.values())  # Iterating over the values in the dictionary
        ]
        return data, ['pid']
