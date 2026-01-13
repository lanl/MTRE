# 🤝 Contributing to MTRE

We welcome contributions to MTRE! This guide will help you get started.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Adding New Features](#adding-new-features)
- [Style Guide](#style-guide)

## Code of Conduct

This project is released under the BSD-3-Clause License. By participating, you agree to maintain a respectful and collaborative environment.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MTRE.git
   cd MTRE
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/MTRE.git
   ```

## Development Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for testing VLM integration)
- Git

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install package in editable mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Verify Setup

```bash
# Run basic tests
python -m pytest tests/

# Check code style
flake8 .
black --check .
```

## Project Structure

Understanding the codebase:

```
MTRE/
├── model/                      # VLM wrappers
│   ├── base.py                # Base model interface
│   ├── LLaVA.py               # LLaVA implementation
│   └── ...                    # Other VLM implementations
├── dataset/                    # Dataset loaders
│   ├── __init__.py            # Dataset registry
│   ├── MADBench.py            # MAD-Bench loader
│   └── gen_scripts/           # Data generation
├── utils/                      # Utility functions
│   ├── func.py                # General utilities
│   ├── metric.py              # Evaluation metrics
│   ├── prompt.py              # Prompt templates
│   └── parse.py               # Output parsing
├── non_linear_notebooks/       # MTRE implementations
│   ├── model_archs/           # Core MTRE models
│   │   ├── calib.py          # MTRE-τ
│   │   └── attention_models_experimental.py  # MTRE
│   └── configs/               # Hyperparameters
├── scripts/                    # Experiment scripts
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

### Key Files

- **[run_model.py](run_model.py)**: Main logit extraction script
- **[model/base.py](model/base.py)**: Base class for all VLM wrappers
- **[dataset/__init__.py](dataset/__init__.py)**: Dataset registration
- **[utils/metric.py](utils/metric.py)**: Evaluation metrics (AUROC, Accuracy, etc.)

## Making Changes

### Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** (see guidelines below)

3. **Test your changes**:
   ```bash
   python -m pytest tests/
   ```

4. **Commit with descriptive messages**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

### Commit Message Guidelines

Use clear, descriptive commit messages:

```
✅ Good:
- "Add support for Gemini VLM model"
- "Fix logit extraction for multi-GPU setup"
- "Update MTRE-τ temperature calibration"

❌ Bad:
- "fix bug"
- "update code"
- "changes"
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=. tests/
```

### Writing Tests

Add tests for new features in `tests/`:

```python
# tests/test_new_feature.py
import pytest
from model import build_model

def test_new_model_loads():
    """Test that new model loads correctly."""
    model = build_model("NewModel", model_path="/path/to/checkpoint")
    assert model is not None

def test_logit_extraction():
    """Test logit extraction produces correct shape."""
    model = build_model("LLaVA-7B")
    logits = model.extract_logits(image, question, num_tokens=10)
    assert logits.shape == (10, vocab_size)
```

## Submitting Changes

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guide (run `black .` and `flake8 .`)
- [ ] All tests pass (`pytest tests/`)
- [ ] New features have tests
- [ ] Documentation is updated (README, docstrings, etc.)
- [ ] Commit messages are clear
- [ ] PR description explains changes

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe testing performed

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
```

## Adding New Features

### Adding a New VLM Model

1. **Create model wrapper** in `model/YourModel.py`:

```python
from model.base import BaseModel
import torch

class YourModel(BaseModel):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        # Initialize your model
        self.model = ...
        self.processor = ...
    
    def generate_with_logits(self, image, question, num_tokens=10):
        """Extract logits for first num_tokens."""
        # Implementation
        return logits
    
    def generate_answer(self, image, question):
        """Generate full answer."""
        # Implementation
        return answer
```

2. **Register model** in `model/__init__.py`:

```python
def build_model(model_name, **kwargs):
    if model_name == "YourModel":
        from model.YourModel import YourModel
        return YourModel(**kwargs)
    # ... existing models
```

3. **Add test** in `tests/test_models.py`:

```python
def test_your_model():
    model = build_model("YourModel", model_path="/path")
    assert model is not None
```

4. **Create run script** in `scripts/YourTask/run_YourModel.sh`

5. **Update documentation** in README.md

### Adding a New Dataset

1. **Create dataset loader** in `dataset/YourDataset.py`:

```python
from torch.utils.data import Dataset
from PIL import Image

class YourDataset(Dataset):
    def __init__(self, data_path, split="train"):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.samples = self._load_samples()
    
    def _load_samples(self):
        # Load annotations
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'image': Image.open(sample['image_path']),
            'question': sample['question'],
            'label': sample['label'],
            'id': sample['id']
        }
```

2. **Register dataset** in `dataset/__init__.py`:

```python
def build_dataset(dataset_name, **kwargs):
    if dataset_name == "YourDataset":
        from dataset.YourDataset import YourDataset
        return YourDataset(**kwargs)
    # ... existing datasets
```

3. **Add data setup** in `DATA_SETUP.md`

4. **Create eval script** in `non_linear_notebooks/eval_yourdataset.py`

### Adding a New MTRE Variant

1. **Implement architecture** in `non_linear_notebooks/model_archs/your_variant.py`:

```python
import torch
import torch.nn as nn

class MTREVariant(nn.Module):
    def __init__(self, num_tokens=10, hidden_dim=256):
        super().__init__()
        # Define architecture
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, logits):
        # logits shape: (batch, num_tokens, vocab_size)
        # Implementation
        return reliability_scores
```

2. **Add configuration** in `non_linear_notebooks/configs/your_variant_config.yaml`

3. **Create evaluation script** in `non_linear_notebooks/eval_your_variant.py`

4. **Document** in README.md

## Style Guide

### Python Code Style

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Indentation**: 4 spaces
- **Quotes**: Prefer double quotes `"` over single quotes `'`
- **Imports**: Group standard library, third-party, local
- **Docstrings**: Use Google-style docstrings

### Code Formatting

Use **Black** for automatic formatting:

```bash
# Format all files
black .

# Check without modifying
black --check .
```

### Linting

Use **flake8** for linting:

```bash
flake8 . --max-line-length=100 --ignore=E203,W503
```

### Example: Well-Formatted Code

```python
"""Module for MTRE evaluation metrics."""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List, Tuple


def compute_auroc(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute AUROC for binary classification.
    
    Args:
        predictions: Array of predicted probabilities, shape (N,)
        labels: Array of ground truth labels (0 or 1), shape (N,)
    
    Returns:
        AUROC score between 0 and 1
    
    Raises:
        ValueError: If predictions and labels have different lengths
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, labels={len(labels)}"
        )
    
    return roc_auc_score(labels, predictions)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute multiple classification metrics.
    
    Args:
        predictions: Predicted probabilities
        labels: Ground truth labels
        threshold: Decision threshold for binary classification
    
    Returns:
        Dictionary with AUROC, Accuracy, F1, etc.
    """
    binary_preds = (predictions >= threshold).astype(int)
    
    return {
        "auroc": compute_auroc(predictions, labels),
        "accuracy": accuracy_score(labels, binary_preds),
        "threshold": threshold
    }
```

### Docstring Format

Use **Google-style docstrings**:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief one-line description.
    
    Longer description if needed, explaining purpose and behavior.
    Can span multiple lines.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When this exception is raised
    
    Examples:
        >>> function_name(value1, value2)
        expected_output
    """
    pass
```

## Documentation

### Updating Documentation

When making changes, update relevant documentation:

- **README.md**: High-level overview, installation, usage
- **DATA_SETUP.md**: Dataset preparation instructions
- **CONTRIBUTING.md**: This file (development guidelines)
- **Docstrings**: Function/class documentation
- **Comments**: Inline explanations for complex logic

### Building Docs (Future)

```bash
# When Sphinx docs are added
cd docs/
make html
```

## Questions?

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers (see README.md)

## License

By contributing, you agree that your contributions will be licensed under the BSD-3-Clause License.
