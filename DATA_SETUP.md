# Data Setup Guide

This guide explains how to obtain and prepare datasets for MTRE experiments.

## Quick Start

```bash
# Create data directories
mkdir -p data/{Lines,Squares,Triangle,OlympicLikeLogo}
mkdir -p data/{coco,MM-SafetyBench,MathVista,MME,MMMU}
```

## Dataset Overview

| Dataset | Task | Source | Generation |
|---------|------|--------|------------|
| Lines | Counting intersections | Synthetic | `gen_line_data.py` |
| Squares | Counting squares | Synthetic | `gen_square_data.py` |
| Triangle | Counting triangles | Synthetic | `gen_triangle_data.py` |
| OlympicLikeLogo | Counting rings | Synthetic | `gen_olympic_data.py` |
| MAD-Bench | Deceptive questions | COCO + annotations | Download |
| MM-SafetyBench | Jailbreak detection | External | Download |
| MathVista | Math uncertainty | External | Download |
| POPE | Object hallucination | COCO | Download |
| MME | VLM evaluation | HuggingFace | Download + extract |
| MMMU | Multimodal understanding | HuggingFace | Download + extract |

---

## Synthetic Datasets (Generated)

These datasets can be fully regenerated from scripts:

### Lines (Intersection Counting)
```bash
mkdir -p ./data/Lines
python dataset/gen_scripts/gen_line_data.py
# Outputs: 600 images + metadata.json
```

### Squares (Square Counting)
```bash
mkdir -p ./data/Squares
python dataset/gen_scripts/gen_square_data.py
```

### Triangles
```bash
mkdir -p ./data/Triangle
python dataset/gen_scripts/gen_triangle_data.py
# For harder variant:
python dataset/gen_scripts/gen_triangle_data_hard.py
```

### Olympic-Like Logo (Ring Counting)
```bash
mkdir -p ./data/OlympicLikeLogo
python dataset/gen_scripts/gen_olympic_data.py
# For harder variant:
python dataset/gen_scripts/gen_olympic_data_hard.py
```

---

## External Datasets (Download Required)

### COCO Images (Required for MAD-Bench, POPE)

Download from [COCO Dataset](https://cocodataset.org/#download):
- `val2017.zip` (MAD-Bench)
- `train2014.zip` and `val2014.zip` (POPE)

```bash
cd data/coco
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2017.zip && unzip train2014.zip && unzip val2014.zip
```

Expected structure:
```
data/coco/
├── val2017/
├── train2014/
└── val2014/
```

### MM-SafetyBench

Download from [Google Drive](https://drive.google.com/file/d/16jULXndiNwFE8L6NzTz63StM9Njts3qS/view):
- Based on [The-First-to-Know](https://github.com/Qinyu-Allen-Zhao/LVLM-LP/) preprocessing

```bash
# After downloading, extract to:
unzip MM-SafetyBench.zip -d data/MM-SafetyBench/
```

### MathVista

Download testmini set from [MathVista](https://mathvista.github.io/):

```bash
cd data/MathVista
# Follow instructions at https://mathvista.github.io/
```

### MME (From Parquet)

MME data is distributed as parquet files. Extract images:

```bash
# 1. Download MME parquet files to data/MME/data/
# 2. Extract images:
python data/extractors/extract_mme.py --src ./data/MME/data --out ./data/MME/images
```

### MMMU (From Parquet)

Similar to MME, MMMU uses parquet format:

```bash
# Download MMMU parquet files to data/MMMU/{subject}/
# Images are extracted automatically by the dataset loader
```

---

## Annotations (Included)

The following annotation files are included in this repository:

```
data/
├── MADBench/
│   └── mad_annot.json          # MAD-Bench annotations
├── pope/
│   ├── coco_pope_adversarial.json
│   ├── coco_pope_popular.json
│   └── coco_pope_random.json   # POPE annotations
└── MM-SafetyBench/
    └── processed_questions/    # Question annotations
```

---

## Configuring Dataset Paths

After setting up data, update paths in `dataset/__init__.py`:

```python
dataset_roots = {
    "MMSafety": "./data/MM-SafetyBench/",
    "MAD": "./data/coco/val2017/",
    "MathVista": "./data/MathVista/",
    "POPE": "./data/coco/",
    "OlympicLikeLogo": "./data/OlympicLikeLogo/",
    "Lines": "./data/Lines/",
    "Triangle": "./data/Triangle/",
    "Squares": "./data/Squares/",
    "MME": "./data/MME/",
    "MMMU": "./data/MMMU/"
}
```

---

## Verification

Test that datasets load correctly:

```python
from dataset import build_dataset
from utils.prompt import Prompter

# Test MAD-Bench
prompter = Prompter("oe", "mad")
class Args: prompt = "oe"
data, keys = build_dataset("MAD", "val", prompter, Args())
print(f"MAD-Bench: {len(data)} samples")

# Test Lines
data, keys = build_dataset("Lines", "val", prompter, Args())
print(f"Lines: {len(data)} samples")
```

---

## Troubleshooting

**FileNotFoundError: Image not found**
- Check that `dataset_roots` paths are correct in `dataset/__init__.py`
- Verify images are extracted/downloaded to the expected locations

**Empty dataset returned**
- Some datasets require specific splits ("train", "val", "test")
- Check the dataset loader for supported splits

**Parquet read errors**
- Install pyarrow: `pip install pyarrow`
- Ensure parquet files are not corrupted (re-download if needed)
