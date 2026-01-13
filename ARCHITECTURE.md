# MTRE Architecture

This document describes the high-level architecture and data flow of the MTRE (Multi-Token Reliability Estimation) system.

## Overview

MTRE is a two-phase pipeline for hallucination detection in Vision-Language Models (VLMs):

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 1: Logit Extraction                    │
│  Image + Query → VLM → Logits (first 10 tokens) → JSON/NPY      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE 2: MTRE Evaluation                     │
│  Logits → Multi-Head Attention → Reliability Score → Metrics    │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

### Core Modules

| Directory | Purpose |
|-----------|---------|
| `model/` | VLM wrapper classes for different architectures |
| `dataset/` | Dataset loaders and data generation scripts |
| `utils/` | Shared utilities (metrics, prompts, parsing) |
| `non_linear_notebooks/` | MTRE model architectures and evaluation |
| `scripts/` | Shell scripts for batch processing |

### Model Wrappers (`model/`)

Each VLM has a wrapper class that implements:
- `forward_with_probs(image, prompt)` → returns response, output_ids, logits, probs
- `get_p_true(image, prompt)` → returns P(True) baseline score

```
model/
├── __init__.py      # Model factory: build_model(args)
├── base.py          # LargeMultimodalModel base class
├── LLaVA.py         # LLaVA 1.5 wrapper
├── LLaVA_NeXT.py    # LLaVA-NeXT wrapper
├── Intern_VL_3_5.py # InternVL 3.5 wrapper
├── MiniGPT4.py      # MiniGPT-4 wrapper
├── mPLUG_Owl.py     # mPLUG-Owl wrapper
└── LLaMA_Adapter.py # LLaMA-Adapter wrapper
```

### Dataset Loaders (`dataset/`)

Each dataset class implements:
- Data loading from disk
- Label extraction
- Prompt formatting via `Prompter`

```
dataset/
├── __init__.py          # Dataset factory: build_dataset(name, split, prompter, args)
├── MADBench.py          # Deceptive question detection
├── MMSafety.py          # Safety/jailbreak detection
├── MathV.py             # MathVista uncertainty
├── MME.py               # MME benchmark
├── MMMU.py              # MMMU multimodal benchmark
├── POPE.py              # Object hallucination
├── Lines.py             # Counting task: lines
├── Squares.py           # Counting task: squares
├── Triangle.py          # Counting task: triangles
├── OlympicLikeLogo.py   # Counting task: olympic logos
└── gen_scripts/         # Synthetic data generation
```

### MTRE Model Architectures (`non_linear_notebooks/model_archs/`)

```
model_archs/
├── attention_models_experimental.py  # MTRE: Multi-head attention aggregator
├── calib.py                          # MTRE-tau: Calibrated variant
├── calib_tau.py                      # Advanced tau-based calibration
├── baseline_models.py                # LP, P(True) baselines
└── helpers.py                        # Training utilities
```

## Data Flow

### Phase 1: Logit Extraction (`run_model.py`)

```python
# Input
args.model_name = "LLaVA-7B"
args.dataset = "MAD"
args.answers_file = "./output/LLaVA/MAD_val.jsonl"

# Pipeline
1. build_model(args)           # Load VLM
2. build_dataset(...)          # Load dataset
3. for each sample:
   - model.forward_with_probs(image, prompt)
   - Extract logits[:10]       # First 10 tokens
   - Save to JSONL/NPY
```

**Output format (JSONL):**
```json
{
  "image": "image_001.jpg",
  "question": "Is there a cat in the image?",
  "response": "Yes, there is a cat.",
  "label": 1,
  "logits_ref": {"npy": "logits_npy/ab/cd/sample_001.float16.npy", "shape": [10, 32000]}
}
```

### Phase 2: MTRE Evaluation

```python
# Load logits from Phase 1
logits = load_logits(jsonl_file)  # Shape: [N, 10, vocab_size]

# MTRE Architecture
class MTREAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        self.token_embed = nn.Linear(vocab_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, logits):
        # logits: [batch, 10, vocab_size]
        x = self.token_embed(logits)      # [batch, 10, embed_dim]
        x = self.attention(x, x, x)       # Multi-head self-attention
        x = x.mean(dim=1)                 # Pool over tokens
        return self.classifier(x)          # [batch, 1] reliability score
```

## Evaluation Tasks

| Task | Dataset | Metric | Description |
|------|---------|--------|-------------|
| Jailbreak Defense | MM-SafetyBench | Accuracy, AUROC | Detect unsafe/harmful outputs |
| Deceptive Questions | MAD-Bench | Accuracy, AUROC | Identify misleading queries |
| Math Uncertainty | MathVista | Accuracy | Estimate answer reliability |
| Object Hallucination | POPE | Accuracy | Detect non-existent objects |
| Counting | Lines/Squares/Triangles | Accuracy | Count geometric shapes |

## Key Files

| File | Purpose |
|------|---------|
| `run_model.py` | Main logit extraction script |
| `non_linear_notebooks/nonlinear_attention_mad_*.py` | MAD-Bench evaluation |
| `non_linear_notebooks/nonlinear_attention_safety_*.py` | Safety evaluation |
| `non_linear_notebooks/arith_all.py` | Cross-validation for counting/math |
| `non_linear_notebooks/model_archs/attention_models_experimental.py` | MTRE architecture |
| `non_linear_notebooks/model_archs/calib.py` | MTRE-tau architecture |

## Configuration

Hyperparameters are stored in `non_linear_notebooks/configs/`:
- `embed_dim`: Token embedding dimension
- `num_heads`: Attention heads
- `num_layers`: Transformer layers
- `dropout`: Regularization
- `learning_rate`: Optimizer LR

## Adding New Models

1. Create `model/NewModel.py` inheriting from `LargeMultimodalModel`
2. Implement `forward_with_probs()` and optionally `get_p_true()`
3. Register in `model/__init__.py`

## Adding New Datasets

1. Create `dataset/NewDataset.py`
2. Implement data loading and label extraction
3. Register in `dataset/__init__.py`
