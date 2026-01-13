"""
Model Module - Vision-Language Model Wrappers

This module provides unified interfaces for various Vision-Language Models (VLMs)
used in the MTRE hallucination detection pipeline.

Supported Models:
    - LLaVA-7B, LLaVA-13B: LLaVA family models
    - LLaVA_NeXT: Next-generation LLaVA
    - Intern_VL_3_5: InternVL 3.5
    - MiniGPT4: MiniGPT-4
    - mPLUG-Owl: mPLUG-Owl model
    - LLaMA_Adapter: LLaMA-Adapter V2

Usage:
    from model import build_model
    model = build_model(args)
    response, output_ids, logits, probs = model.forward_with_probs(image, prompt)
"""


def build_model(args):
    """
    Factory function to instantiate a VLM based on the model name.

    Args:
        args: Namespace object containing:
            - model_name (str): Name of the model to load
            - model_path (str): Path to model weights (for some models)
            - temperature (float): Sampling temperature
            - top_p (float): Nucleus sampling parameter
            - num_beams (int): Beam search width

    Returns:
        LargeMultimodalModel: Instantiated model wrapper, or None if not found.

    Example:
        >>> args.model_name = "LLaVA-7B"
        >>> args.model_path = "liuhaotian/llava-v1.5-7b"
        >>> model = build_model(args)
    """
    if args.model_name == "InstructBLIP":
        from .InstructBLIP import InstructBLIP
        model = InstructBLIP(args)
    elif args.model_name == "LLaVA-7B":
        from .LLaVA import LLaVA
        model = LLaVA(args)
        print('Model: ', model)
    elif args.model_name == "LLaVA-13B":
        from .LLaVA import LLaVA
        model = LLaVA(args)
    elif args.model_name == "LLaMA_Adapter":
        from .LLaMA_Adapter import LLaMA_Adapter
        model = LLaMA_Adapter(args)
    elif args.model_name == "MMGPT":
        from .MMGPT import MMGPT
        model = MMGPT(args)
    elif args.model_name == "GPT4V":
        from .GPT4V import GPTClient
        model = GPTClient()
    elif args.model_name == "MiniGPT4":
        from .MiniGPT4 import MiniGPT4
        model = MiniGPT4(args)
    elif args.model_name == "mPLUG-Owl":
        from .mPLUG_Owl import mPLUG_Owl
        model = mPLUG_Owl(args)
    elif args.model_name == 'LLaVA_NeXT':
        from .LLaVA_NeXT import LlavaNeXT
        model = LlavaNeXT(model_path=args.model_path)
    elif args.model_name == 'Intern_VL_3_5':
        # print(args.model_name)
        from .Intern_VL_3_5 import Intern_VL_3_5
        model = Intern_VL_3_5(model_path=args.model_path)
    else:
        print('Could not find: ',args.model_name)
        model = None
        
    return model
