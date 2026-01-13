import sys
###################################################
####### Set the path to the repository here #######
sys.path.append("")
###################################################

import torch
from PIL import Image

import llama
from model.base import LargeMultimodalModel

class LLaMA_Adapter(LargeMultimodalModel):
    def __init__(self, args):
        super(LLaMA_Adapter, self).__init__()
        llama_dir = "" #Set PATH HERE.

        # choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
        self.model, self.preprocess = llama.load("", llama_dir, llama_type="7B", device="cuda")
        self.model.eval()
        
        self.temperature = args.temperature
        self.top_p = args.top_p
        
    def forward_with_probs(self, image, prompt):
        prompt = llama.format_prompt(prompt)
        image = Image.fromarray(image).convert('RGB')
        img = self.preprocess(image).unsqueeze(0).to("cuda")

        response, output_ids, logits = self.model.generate(img, [prompt],
                temperature=self.temperature,
                top_p=self.top_p)
        return response[0], output_ids, logits, None
    

    def get_p_true(self, image, prompt, return_dict=False): #NOTE THIS IS DEPENDANT ON THE MODEL
        prompt += ' '
        prompt = llama.format_prompt(prompt)
        image = Image.fromarray(image).convert('RGB')
        img = self.preprocess(image).unsqueeze(0).to("cuda")

        with torch.inference_mode():
            model_output_true = self.model.p_true_forward(img, [prompt],
                temperature=self.temperature,
                top_p=self.top_p
            )

        loss_true = model_output_true

        return -loss_true.cpu().item()

    