"""
Dataset Module - Benchmark Dataset Loaders

This module provides dataset loaders for various VLM evaluation benchmarks
used in MTRE hallucination detection experiments.

Supported Datasets:
    - MAD (MADBench): Deceptive question detection
    - MMSafety (MM-SafetyBench): Jailbreak/safety detection
    - MathVista: Math problem uncertainty
    - POPE: Object hallucination detection
    - MME: General VLM evaluation
    - MMMU: Multimodal understanding
    - Counting tasks: Lines, Squares, Triangle, OlympicLikeLogo

Configuration:
    Update `dataset_roots` dictionary below to point to your local data paths.

Usage:
    from dataset import build_dataset
    data, extra_keys = build_dataset("MAD", "val", prompter, args)
"""

# Dataset root paths - UPDATE THESE to match your local setup
dataset_roots = {
    "MMSafety": "./data/MM-SafetyBench/", #
    "MAD": "./data/coco/val2017/",   # MADBench uses COCO images
    "MathVista": "./data/MathVista/",
    "POPE": "/data/coco/",  # POPE uses COCO images
    "OlympicLikeLogo": "./data/OlympicLikeLogo/",
    "Lines": "./data/Lines/",
    "Triangle": "./data/Triangle/",
    "Squares": "./data/Squares/",
    "MME": "./data/MME/",      # UPDATE: Set to your MME data path
    "MMMU": "./data/MMMU/"     # UPDATE: Set to your MMMU data path
}
class BaseDataset:
    """Base class for all dataset loaders (placeholder for unsupported datasets)."""

    def __init__(self):
        pass

    def get_data(self):
        """Return empty data and keys for unsupported datasets."""
        return [], []


def build_dataset(dataset_name, split, prompter, args):
    """
    Factory function to load and return dataset samples.

    Args:
        dataset_name (str): Name of the dataset to load (e.g., "MAD", "MMSafety")
        split (str): Dataset split ("train", "val", "test")
        prompter: Prompter object for formatting prompts
        args: Namespace with additional arguments (e.g., args.prompt for prompt style)

    Returns:
        tuple: (data, extra_keys)
            - data (list): List of sample dictionaries with keys:
                - 'img_path' (str): Path to image file
                - 'question' (str): Formatted prompt/question
                - 'label' (int/str): Ground truth label
            - extra_keys (list): Additional keys to preserve in output

    Example:
        >>> data, keys = build_dataset("MAD", "val", prompter, args)
        >>> print(data[0].keys())  # dict_keys(['img_path', 'question', 'label', ...])
    """
    if dataset_name == "MAD":
        from .MADBench import MADBench
        dataset = MADBench(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "MathVista":
        from .MathV import MathVista
        dataset = MathVista(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "MMSafety":
        from .MMSafety import MMSafetyBench
        dataset = MMSafetyBench(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "POPE":
        from .POPE import POPEDataset
        dataset = POPEDataset(split, dataset_roots[dataset_name])
    elif dataset_name == "OlympicLikeLogo":
        from .OlympicLikeLogo import OlympicLikeLogo
        dataset = OlympicLikeLogo(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "Lines":
        from .Lines import Lines
        dataset = Lines(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "Triangle":
        from .Triangle import Triangle
        dataset = Triangle(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "Squares":
        from .Squares import Squares
        dataset = Squares(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == 'MME':
        from .MME import MMEParquetDataset
        dataset = MMEParquetDataset(prompter=None, data_root=dataset_roots[dataset_name]+'data/', 
                                    image_root=dataset_roots[dataset_name]+'images/')
    elif 'MMMU' in dataset_name:
        from .MMMU import MMMUParquetDataset
        # subject = dataset_name.split('_')[-1]
        subject = '_'.join(dataset_name.split('_')[1:]) 
        print(dataset_name)
        print(subject)
        subject_path = dataset_roots['MMMU'] + subject
        cache_dir = subject_path+'/_img_cache/'
        dataset = MMMUParquetDataset(prompter=None, subject_dir=subject_path, image_mode="first", cache_dir=cache_dir,
                                     split=split)
    else:
        # from .base import BaseDataset
        dataset = BaseDataset()
        
    return dataset.get_data()

def get_dataset_obj(dataset_name, split, prompter, args):
    if dataset_name == "VizWiz":
        from .VizWiz import VizWizDataset
        dataset = VizWizDataset(prompter, split, dataset_roots[dataset_name], args.prompt, args.non_linear_response_folder)
    elif dataset_name == "MAD":
        from .MADBench import MADBench
        dataset = MADBench(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "ImageNet":
        from .ImageNet import ImageNetDataset
        dataset = ImageNetDataset(split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "MathVista":
        from .MathV import MathVista
        dataset = MathVista(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "MMSafety":
        from .MMSafety import MMSafetyBench
        dataset = MMSafetyBench(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "POPE":
        from .POPE import POPEDataset
        dataset = POPEDataset(split, dataset_roots[dataset_name])
    elif dataset_name == "OlympicLikeLogo":
        from .OlympicLikeLogo import OlympicLikeLogo
        dataset = OlympicLikeLogo(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "Pentagon":
        from .Pentagon import Pentagon
        dataset = Pentagon(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "Lines":
        from .Lines import Lines
        dataset = Lines(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "Triangle":
        from .Triangle import Triangle
        dataset = Triangle(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "Pattern":
        from .Pattern import Pattern
        dataset = Pattern(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "Alphabet":
        from .Alphabet import Alphabet
        dataset = Alphabet(prompter, split, dataset_roots[dataset_name], args.prompt)
    elif dataset_name == "Squares":
        from .Squares import Squares
        dataset = Squares(prompter, split, dataset_roots[dataset_name], args.prompt)
    else:
        # from .base import BaseDataset
        dataset = BaseDataset()
        
    return dataset
