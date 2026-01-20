# 🔬 MTRE: Multi-Token Reliability Estimation for Hallucination Detection in VLMs

[![arXiv](https://img.shields.io/badge/arXiv-2505.11741-b31b1b.svg)](https://arxiv.org/abs/2505.11741)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

**MTRE** is a lightweight, white-box method for detecting hallucinations in Vision-Language Models (VLMs). By analyzing the complete sequence of early token logits using multi-token log-likelihood ratios and self-attention, MTRE captures reliability dynamics that single-token methods miss.

*Geigh Zollicoffer, Minh Vu, Manish Bhattarai*

## 🌟 Key Features

- **Multi-Token Analysis**: Analyzes the first n tokens instead of just the first token, capturing subtle inconsistencies that accumulate over time
- **White-Box Detection**: Lightweight method using model logits without requiring fine-tuning or auxiliary networks
- **State-of-the-Art Performance**: 9.4% gain in Accuracy and 14.8% gain in AUROC over standard detection methods
- **Comprehensive Benchmarks**: Tested on MAD-Bench, MM-SafetyBench, MathVista, MMMU, MME, and four compositional-geometry tasks
- **Multiple VLM Support**: Compatible with LLaVA, MiniGPT-4, mPLUG-Owl, LLaMA-Adapter, and InternVL-3.5

### Usage
We evaluate the MTRE method on different tasks. For a specific task, you need to (i) prepare the datasets, (ii) prepare the vlm, (iii) run the model on the dataset to get logits of the first n tokens, and lastly, (iv) evaluate the performance of MTRE. The process may take some time, and may require minor adjustments to the models or packages downloaded.

## 📋 Table of Contents

- [Installation](#-installation)
- [Pipeline Overview](#-pipeline-overview)
- [Usage](#-usage)
  - [Data Preparation](#1-data-preparation)
  - [Model Setup](#2-model-setup)
  - [Extract Logits](#3-extract-logits)
  - [Evaluate MTRE](#4-evaluate-mtre)
- [Configuration](#%EF%B8%8F-configuration)
- [Supported Models](#-supported-models)
- [Supported Datasets](#-supported-datasets)
- [Citation](#-citation)
- [Methodology](#-methodology)
- [Troubleshooting](#-troubleshooting)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lanl/MTRE.git
cd MTRE
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
# or: conda activate mtre_env  # Conda
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and download models (required for all):
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT && pip install -e . && cd ..
```


### Command Line Usage

Extract logits from a VLM on a specific task (Make sure to set model path):

```bash
# Run LLaVA-7B on MM-SafetyBench
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/Safety/run_LLaVA_7B.sh

# Run on MAD-Bench
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/MAD/run_LLaVA_7B.sh

# Run on InternVL-3.5 on MMMU
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/MMMU/run_all_InternVL_3_5.sh
```

Evaluate MTRE on extracted logits:

```bash
# Evaluate on Type 1 datasets (MAD-Bench, MM-SafetyBench)
cd non_linear_notebooks
python nonlinear_attention_mad_1.py
python nonlinear_attention_safety_1.py

# Evaluate on counting/math tasks
python arith_all.py LLaVA-7B Squares MTRE

# Evaluate on larger models (MMMU type 2)
python MMMU_eval.py InternVL_3_5 MTRE MMMU_2
```

## 📊 Pipeline Overview

MTRE operates in two main stages:

### Stage 1: Logit Extraction
- Generate VLM responses for image-question pairs
- Extract token-level log-likelihoods for the first 10 tokens
- Store logits, embeddings, and metadata

### Stage 2: MTRE Evaluation
- Aggregate multi-token log-likelihood ratios
- Apply self-attention mechanism across token sequences
- Compute reliability scores using MTRE or MTRE-τ
- Detect hallucinations based on reliability thresholds

```
Image + Question → [VLM Generation] → First k Tokens
                         ↓
                   Token Logits & Embeddings
                         ↓
              [MTRE Analysis] → Multi-Token Ratios
                         ↓                    ↓
               Self-Attention       Log-Likelihood Aggregation
                         ↓                    ↓
                         → Reliability Score ←
                                ↓
                      Hallucination Detection
```

## 📖 Usage

### Project Structure

```
MTRE/
├── run_model.py           # Main script to extract logits from VLMs
├── collect_results.py     # Aggregate evaluation results
├── model/                 # VLM model wrappers (LLaVA, MiniGPT4, etc.)
│   ├── __init__.py
│   ├── base.py
│   ├── LLaVA.py
│   ├── LLaVA_NeXT.py
│   ├── Intern_VL_3_5.py
│   ├── MiniGPT4.py
│   ├── mPLUG_Owl.py
│   └── LLaMA_Adapter.py
├── dataset/               # Dataset loaders (MADBench, MMSafety, etc.)
│   ├── __init__.py
│   ├── MADBench.py
│   ├── MMSafety.py
│   ├── MathV.py
│   ├── POPE.py
│   ├── Lines.py
│   ├── Squares.py
│   ├── Triangle.py
│   ├── OlympicLikeLogo.py
│   └── gen_scripts/       # Data generation scripts
├── utils/                 # Utility functions (metrics, prompts, parsing)
│   ├── func.py
│   ├── metric.py
│   ├── prompt.py
│   ├── parse.py
│   └── unc_eval_extract.py
├── non_linear_notebooks/                    # MTRE model architectures, training, and 
|   ├── model_archs/                         # Core MTRE implementations
|   │   ├── calib.py                         # MTRE-τ architecture
|   │   └── attention_models_experimental.py # MTRE attention variants
|   ├── configs/                             # Hyperparameter configurations
|   ├── arith_all_small.py                   # Arithmetic eval (small)
|   ├── arith_all_large.py                   # Arithmetic eval (large)
|   ├── nonlinear_attention_mad_1.py         # MAD experiment (variant 1)
|   ├── nonlinear_attention_mad_2.py         # MAD experiment (variant 2)
|   ├── nonlinear_attention_safety_1.py      # Safety experiment (variant 1)
|   ├── nonlinear_attention_safety_2.py      # Safety experiment (variant 2)
|   ├── MMMU_eval.py                         # MMMU evaluation
|   ├── MMMU_eval_ray.py                     # Distributed MMMU evaluation
|   ├── MMMU_eval_tune.py                    # MMMU hyperparameter tuning
|   ├── MMMU_tune_tau.py                     # τ tuning for MMMU
|   ├── cross_validation.py                  # Cross-validation framework
|   ├── cross_validation_tune.py             # Cross-validation tuning framework
|   ├── run_eval_type_clean.py               # Unified evaluation runner
|   ├── surprise_scratch.py                  # Scratch experiments
|   └── scratch_results/                     # Experimental outputs

├── scripts/               # Shell scripts for running experiments
│   ├── Safety/
│   ├── MAD/
│   ├── MathV/
│   ├── Lines/
│   ├── Squares/
│   ├── Triangle/
|   ├── MMMU/
|   ├── MME/
│   └── OlympicLikeLogo/
└── data/                  # Dataset annotations (download separately)
    ├── MADBench/
    ├── Pope/     #Safety uses Pope
    ├── MathV/
    ├── Lines/
    ├── Squares/
    ├── Triangle/
    ├── MMMU/
    ├── MME/
    └── OlympicLikeLogo/
```

### 1. Data Preparation

**See [DATA_SETUP.md](DATA_SETUP.md) for complete instructions.**

After downloading datasets, modify dataset paths in [dataset/__init__.py](dataset/__init__.py).

#### Task 1: Defense against Jailbreak Attacks

Dataset: [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)

```bash
# Download generated safe/unsafe image-query pairs
# Data based on The-First-to-Know repo
wget https://drive.google.com/file/d/16jULXndiNwFE8L6NzTz63StM9Njts3qS/view?usp=sharing
unzip mm_safety_data.zip -d data/
```

#### Task 2: Identify Deceptive Questions

Dataset: [MAD-Bench](https://arxiv.org/abs/2402.13220)

```bash
# Download COCO val2017 for images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/

# MAD-Bench annotations are included in data/MADBench/
```

#### Task 3: Uncertainty in Math Solving

Dataset: [MathVista](https://mathvista.github.io/) testmini set

```bash
# Download MathVista testmini
git clone https://github.com/lupantech/MathVista.git
# Follow their instructions to download testmini split

# Note: Use utils/unc_eval_extract.py to parse VLM predictions
```

#### Task 4: Mitigate Hallucination (POPE)

Dataset: [POPE](https://github.com/AoiDragon/POPE) with [COCO](https://cocodataset.org/#home)

```bash
# Download COCO train2014 and val2014
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip -d data/pope/
unzip val2014.zip -d data/pope/

# POPE annotations are included in data/pope/
```

#### Task 5: Counting Tasks (Compositional Geometry)

Datasets: Lines, Squares, Triangle, OlympicLikeLogo

Based on [VLMs-ARE-BLIND](https://arxiv.org/abs/2407.06581)

```bash
# Generate datasets using provided scripts
cd dataset/gen_scripts

# Generate Lines dataset
python gen_line_data.py --output_dir ../../data/Lines
python gen_line_data_hard.py --output_dir ../../data/Lines

# Generate Squares dataset
python gen_square_data.py --output_dir ../../data/Squares

# Generate Triangle dataset
python gen_triangle_data.py --output_dir ../../data/Triangle
python gen_triangle_data_hard.py --output_dir ../../data/Triangle

# Generate OlympicLikeLogo dataset
python gen_olympic_data.py --output_dir ../../data/OlympicLikeLogo
python gen_olympic_data_hard.py --output_dir ../../data/OlympicLikeLogo

cd ../..
```

**Note**: See paper [Appendix A.1](https://arxiv.org/abs/2505.11741) for optimized prompts that improve VLM performance on counting tasks.

### 2. Model Setup

Prepare VLMs according to their original repositories:

#### LLaVA (Recommended for Quick Start)

```bash
# Clone LLaVA repository
git clone https://github.com/haotian-liu/LLaVA.git

# Download LLaVA-v1.5-7B checkpoint
# Models auto-download from HuggingFace on first use, or manually:
from transformers import LlavaForConditionalGeneration
model = LlavaForConditionalGeneration.from_pretrained("liuhaotian/llava-v1.5-7b")
```

#### Other Supported Models

- **MiniGPT-4**: [GitHub](https://github.com/Vision-CAIR/MiniGPT-4)
- **mPLUG-Owl**: [GitHub](https://github.com/X-PLUG/mPLUG-Owl)
- **LLaMA-Adapter**: [GitHub](https://github.com/OpenGVLab/LLaMA-Adapter)
- **InternVL-3.5**: [HuggingFace](https://huggingface.co/OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview)
- **LLaVA-NeXT**: [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT)

**Configuration Steps:**

1. Edit [model/__init__.py](model/__init__.py) to set model paths
2. Edit individual model wrappers in [model/](model/) directory with appropriate paths
3. Test model instantiation before running full experiments

### 3. Extract Logits

Run VLMs on datasets to extract first N tokens' logits.

Scripts are organized by task in [scripts/](scripts/) directory.

#### Example: Run LLaVA-7B on MM-SafetyBench

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/Safety/run_LLaVA_7B.sh
```

#### Example: Run on MAD-Bench

```bash
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/MAD/run_LLaVA_7B.sh
```

#### Available Script Categories

- `scripts/Safety/` - MM-SafetyBench experiments
- `scripts/MAD/` - MAD-Bench experiments  
- `scripts/MathV/` - MathVista experiments
- `scripts/Lines/` - Lines counting task
- `scripts/Squares/` - Squares counting task
- `scripts/Triangle/` - Triangle counting task
- `scripts/OlympicLikeLogo/` - Olympic logo counting task
- `scripts/MMMU/` - MMMU task (Large VLMs only)
- `scripts/MME/` - MME task (Large VLMs only)

#### Script Parameters

Each script calls [run_model.py](run_model.py) with:
- `--model_name`: VLM identifier
- `--model_path`: Path to model checkpoint
- `--dataset`: Dataset name
- `--split`: Data split (train/val/test)
- `--answers_file`: Output path for logits
- `--num_chunks`, `--chunk_idx`: For multi-GPU parallelization

See [run_model.py](run_model.py) for full parameter documentation.

### 4. Evaluate MTRE

Run MTRE evaluation on extracted logits.

Evaluation code is in [non_linear_notebooks/](non_linear_notebooks/).

#### Type 1 Datasets (MAD-Bench, MM-SafetyBench, MME, MMMU)

```bash
cd non_linear_notebooks

# Evaluate on MAD-Bench
python nonlinear_attention_mad_1.py

# Evaluate on MM-SafetyBench
python nonlinear_attention_safety_1.py

# Evaluate on larger models (MMMU and MME)
python MMMU_eval.py <Model> <Method> <Dataset>
```

#### Type 2 Datasets

```bash
cd non_linear_notebooks

# Type 2 evaluation for MAD-Bench
python nonlinear_attention_mad_2.py

# Type 2 evaluation for MM-SafetyBench
python nonlinear_attention_safety_2.py

# Evaluate on larger models (MMMU_2 and MME_2)
python MMMU_eval.py <Model> <Method> <Dataset>
```

#### Counting and Math Tasks

```bash
cd non_linear_notebooks

# Syntax: python arith_all.py <model_name> <dataset_name> <method>

# Example: Evaluate LLaVA-7B on Squares with MTRE
python arith_all.py LLaVA-7B Squares MTRE

# Example: Evaluate on Lines dataset
python arith_all.py LLaVA-7B Lines MTRE

# Example: Evaluate on MathVista
python arith_all.py LLaVA-7B MathV MTRE
```

#### Analyze Logit Differences

```bash
cd non_linear_notebooks
python surprise_scratch.py
```

## ⚙️ Configuration

### MTRE Architecture

**MTRE Core** ([model_archs/attention_models_experimental.py](non_linear_notebooks/model_archs/attention_models_experimental.py)):
- Multi-token log-likelihood ratio aggregation
- Self-attention across first 10 tokens
- Reliability head with configurable architecture

**MTRE-τ** ([model_archs/calib.py](non_linear_notebooks/model_archs/calib.py)):
- Temperature-calibrated variant
- Enhanced reliability estimation with τ parameter
- Enable with `tau=True` argument

### Hyperparameter Configurations

Configurations used in the paper: [non_linear_notebooks/configs/](non_linear_notebooks/configs/)

```python
# Example: Load configuration
import yaml

with open('non_linear_notebooks/configs/mtre_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Key parameters:
# - num_tokens: Number of tokens to analyze (default: 10)
# - attention_heads: Self-attention heads (default: 4)
# - tau: Enable temperature calibration (True/False)
# - reliability_head: Architecture variant (linear, mlp, attention)
```

### Dataset Configuration

Edit [dataset/__init__.py](dataset/__init__.py):

```python
# Example dataset paths
DATASET_PATHS = {
    'MADBench': '/path/to/data/MADBench',
    'MMSafety': '/path/to/data/mm_safety',
    'MathV': '/path/to/MathVista/testmini',
    'MMMU': '/path/to/MMMU',
    'MME': '/path/to/MME',
    'POPE': '/path/to/data/pope',
    'Lines': '/path/to/data/Lines',
    'Squares': '/path/to/data/Squares',
    'Triangle': '/path/to/data/Triangle',
    'OlympicLikeLogo': '/path/to/data/OlympicLikeLogo',
}
```
## 🤖 Supported Models

- **LLaVA Family**: 
  - LLaVA-v1.5-7B, LLaVA-v1.5-13B
  - LLaVA-NeXT (LLaVA-v1.6-34B, LLaVA-v1.6-vicuna-7B)
- **InternVL**: InternVL-3.5 20B (GPT-OSS)
- **MiniGPT-4**: MiniGPT-4-7B, MiniGPT-4-13B
- **mPLUG-Owl**: mPLUG-Owl-7B
- **LLaMA-Adapter**: LLaMA-Adapter-v2-7B

## 📚 Supported Datasets

### Hallucination Detection
- **MAD-Bench**: Deceptive questions across 6 categories (CountOfObject, NonexistentObject, ObjectAttribute, SceneUnderstanding, SpatialRelationship, Normal)

### Safety & Jailbreak Defense
- **MM-SafetyBench**: Multimodal safety benchmark with adversarial and safe image-query pairs

### Visual Reasoning
- **MathVista**: Math reasoning with visual context (testmini split)

### Compositional Geometry (Counting)
- **Lines**: Count parallel/intersecting lines
- **Squares**: Count squares in grid patterns
- **Triangle**: Count triangles in geometric configurations
- **OlympicLikeLogo**: Count overlapping circles in Olympic-style arrangements

### Domain Specific Tasks
- **MMMU**: Large-scale, college-level tasks requiring domain knowledge and deliberate reasoning.
- **MME**:  Perception and cognition abilities

## 📝 Citation

If you use MTRE in your research, please cite our paper:

```bibtex
@misc{zollicoffer2025mtremultitokenreliabilityestimation,
      title={MTRE: Multi-Token Reliability Estimation for Hallucination Detection in VLMs}, 
      author={Geigh Zollicoffer and Minh Vu and Manish Bhattarai},
      year={2025},
      eprint={2505.11741},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.11741}, 
}
```

## 🔬 Methodology

### Abstract

Vision-language models (VLMs) now rival human performance on many multimodal tasks, yet they still hallucinate objects or generate unsafe text. Current hallucination detectors (e.g., single-token linear probing and P(True)) typically analyze only the logit of the first generated token—or just its highest-scoring component—overlooking richer signals embedded within earlier token distributions.

MTRE addresses this limitation through:

1. **Multi-Token Analysis**: Analyzes complete sequences of early logits (first 10 tokens) instead of just the first token
2. **KL Divergence Insights**: Demonstrates that hallucinations accumulate over multiple tokens through Kullback-Leibler divergence analysis
3. **Attention-Based Aggregation**: Uses self-attention to aggregate multi-token log-likelihood ratios
4. **Efficient & Tractable**: Remains computationally efficient despite large vocabulary sizes and long logit sequences

### Key Insights

**Why Multi-Token?**
- Hallucinations often emerge after several tokens as subtle inconsistencies accumulate
- Later tokens provide diagnostic information missed by first-token analysis
- KL divergence between hallucinated and non-hallucinated logits increases over token positions

**MTRE Architecture**:
```
Token Logits (T=10) → Log-Likelihood Ratios → Self-Attention Aggregation → Reliability Score
                                                         ↓
                                              Optional τ Calibration (MTRE-τ)
```

**Performance**: 
- **+9.4% Accuracy** gain over standard detection methods
- **+14.8% AUROC** improvement
- State-of-the-art on MAD-Bench, MM-SafetyBench, MathVista, and compositional geometry tasks

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory during logit extraction
```bash
# Solution: Reduce number of chunks or use fewer GPUs
CUDA_VISIBLE_DEVICES=0 bash ./scripts/Safety/run_LLaVA_7B.sh

# Or edit script to increase num_chunks
```

**Issue**: Model checkpoint not found
```bash
# Solution: Check model path in model/__init__.py
# Ensure HuggingFace cache has downloaded the model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('liuhaotian/llava-v1.5-7b')"
```

**Issue**: Dataset path errors
```bash
# Solution: Edit dataset/__init__.py with correct paths
# Verify data files exist:
ls data/MADBench/
ls data/pope/
```

**Issue**: Import errors for VLM frameworks
```bash
# Solution: Ensure LLaVA or other VLM repos are in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/LLaVA-NeXT
```

**Issue**: Evaluation scripts fail to find logits
```bash
# Solution: Check --answers_file path from step 3 matches evaluation script expectations
# Verify logit files exist before running evaluation
```

### Performance Tips

1. **Multi-GPU Parallelization**: Use `num_chunks` and `chunk_idx` to split workload
2. **Batch Processing**: Process multiple samples simultaneously (edit batch_size in scripts)
3. **Mixed Precision**: Use FP16 for faster inference (already enabled in model wrappers)
4. **Caching**: Logits are saved to disk - reuse for multiple evaluation runs

## 🙏 Acknowledgments

- Code base built on [The-First-To-Know](https://github.com/Qinyu-Allen-Zhao/LVLM-LP/) repository
- Built with [Hugging Face Transformers](https://github.com/huggingface/transformers)
- VLM implementations based on:
  - [LLaVA](https://github.com/haotian-liu/LLaVA)
  - [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
  - [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
  - [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl)
  - [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter)
  - [InternVL3.5(GPT-OSS)](https://github.com/OpenGVLab/InternVL)
- Datasets:
  - [MAD-Bench](https://arxiv.org/abs/2402.13220)
  - [MM-SafetyBench](https://github.com/isXinLiu/MM-SafetyBench)
  - [MathVista](https://mathvista.github.io/)
  - [POPE](https://github.com/AoiDragon/POPE)
  - [COCO](https://cocodataset.org/)

## 📄 Copyright Notice

© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

**LANL O#5020**

## 📜 License

This program is Open-Source under the **BSD-3-Clause License**.
 
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

**THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.**