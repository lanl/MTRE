import os
import json

from dataset.base import BaseDataset

class MADBench(BaseDataset):
    def __init__(self, prompter, split="val", data_root="./data/coco/val2017/", response_type='oe'):
        super(MADBench, self).__init__()
        true_split = split.split("_")[0]
        self.ann_root = "./data/MADBench/" #./data/MADBench/"
        self.img_root = data_root
        self.split = split
        self.prompter = prompter
        self.true_split = true_split
        self.response_type = response_type
         
    def get_data(self):
        if 'selfeval' not in self.split:
            ann = json.load(open(os.path.join(self.ann_root, "Normal.json"), 'r'))
            # print('Root :',self.img_root)
            # print(ins["file"])
            normal_data = [
                {
                    "img_path": ins["file"],
                    "question": self.prompter.build_prompt(ins["Question(GPT)"]),
                    "label": 1,
                    "scenario": "Normal",
                    "original_question":ins["Question(GPT)"]
                }
                for ins in ann
            ]
            if self.split == "train":
                data = normal_data[:100]
            else:
                data = normal_data[100:]
            
            val_phrases = []
            for sc in ["CountOfObject", "NonexistentObject", "ObjectAttribute", "SceneUnderstanding", "SpatialRelationship"]:
                ann = json.load(open(os.path.join(self.ann_root, f"{sc}.json"), 'r'))
                print(sc)
                sc_data = [
                    {
                        "img_path": ins["file"],
                        "question": self.prompter.build_prompt(ins["Question(GPT)"]),
                        "label": 0,
                        "scenario": sc,
                        "original_question":ins["Question(GPT)"]
                    }
                    for ins in ann
                ]
                if self.split == "train":
                    sc_data = sc_data[:20]
                else:
                    sc_data = sc_data[20:]
                data += sc_data

            return data, ["scenario"]
        else:
              
            
            self.ann = json.load(open(f"./output/LLaVA-7B/MAD_{self.true_split}_{self.response_type}_labeled.json", 'r'))
            data = [
                {
                    'pid': ins['image'],
                    "img_path": os.path.join(self.img_root, ins['image']),
                    "question": f"Given the image,\nthe query '{ins['question']}',\nand an answer '{ins['response']}.\nIs the answer correct? Please explain, restrict your answer to 20 words.",
                    "label": 1 if self.ann[i]['is_answer']=='yes' else 0
                }
                for i, ins in enumerate(self.ann)
            ]
            return data, ['pid']