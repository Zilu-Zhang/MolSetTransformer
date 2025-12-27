# MolSetTransformer

A flexible deep learning framework for **molecular property prediction** with support for variable-length, multi-molecule inputs.  Built on the **Set Transformer** architecture, MolSetTransformer enables powerful modeling of molecular set interactions through attention mechanisms.

## ⚗️ Overview

MolSetTransformer is designed for scenarios where predictions depend on **sets of molecules** rather than individual compounds. The framework handles: 

- **Variable-length molecular inputs** (e.g., drug combinations, molecular mixtures)
- **Multiple prediction tasks**:  Regression, Binary/Multi-class Classification, and Multi-label scenarios
- **Multi-task learning** with structure-aware heads
- **Interpretable predictions** via attention score export

## ✨ Key Features

### ⚙️ Flexible Molecular Featurization
- **Morgan Fingerprints** - Circular fingerprints for molecular structure encoding
- **RDKit Descriptors** - Physicochemical property descriptors (requires `descriptastorus`)
- **CheMeleon Embeddings** - Graph neural network embeddings (requires `chemprop`)
- **Pre-computed Features** - Support for HDF5-formatted custom embeddings

### 🤖 Set Transformer Architecture
- **Self-Attention Simulator** - Capture inter-molecular interactions
- **Cross-Attention Integrator** - Aggregate variable-sized sets into fixed representations
- **Dynamic Prediction Heads** - Task-specific heads for multi-task learning

### 📊 Training Strategies
- **Standard Training** - Conventional supervised learning
- **Custom Weighted Training** - Task-specific loss weighting
- **Curriculum Learning** - Progressive training strategies

### 🎯 Advanced Capabilities
- **MC Dropout** - Uncertainty quantification via Monte Carlo dropout
- **Attention Export** - Visualize molecular interaction patterns
- **Batch Evaluation** - Efficient cross-validation workflows

## 🚀 Quick Start
### Installation

1. First, clone the repository so you have the project files and example configs:

```bash
git clone https://github.com/Zilu-Zhang/MolSetTransformer.git
cd MolSetTransformer
```

2. You can set up the environment in two ways: pip-only (quick) or conda (recommended).

#### Option A — pip-only (quick)

This uses pip to install required packages. Useful when you already have a Python environment.

```bash
# Install core dependencies
pip install torch numpy pandas rdkit h5py tqdm

# Optional: additional featurizers (may require build tools)
pip install descriptastorus  # RDKit descriptors
pip install chemprop         # CheMeleon embeddings
```

#### Option B — conda (recommended)

Conda gives a reproducible environment and installs RDKit easily.

1) CPU-only environment:

```bash
# from the repo root (where environment.yml is located)
conda env create -f environment.yml
conda activate molsettransformer
```

2) GPU environment (NVIDIA GPU; example using CUDA 11.8):

```bash
# from the repo root (where environment.cuda.yml is located)
conda env create -f environment.cuda.yml
conda activate molsettransformer-gpu
```

Notes:
- The YAML files install optional featurizers (`descriptastorus`, `chemprop`) via pip. Remove them from the `pip:` list if you don't need these.
- If you want different Python or CUDA versions, edit the YAML files to pin versions you prefer.
- On some platforms, installing `chemprop` may require extra build tools. If you encounter issues, create the conda environment first (without `chemprop`) and install `chemprop` later with pip.

### Basic Usage

1. **Prepare your configuration** using the interactive wizard (`wizard.html`) or create a JSON config manually. 

2. **Run the pipeline**:
```bash
python pipeline_runner.py --config path/to/config. json
```

### Configuration Wizard

Open `wizard.html` in your browser for an interactive configuration builder that guides you through: 
- Task type selection (regression, classification, multi-label, multi-task)
- Data file specification and column mapping
- Featurization pipeline construction
- Model architecture configuration
- Training hyperparameter tuning

## 📁 Project Structure

```
MolSetTransformer/
├── pipeline_runner.py          # Main entry point
├── wizard.html                  # Interactive configuration wizard
├── pipeline_utils/
│   ├── data_manager.py         # Data loading and preprocessing
│   ├── feature_factory.py      # Molecular featurization
│   ├── model_builder.py        # Model instantiation
│   ├── mol_set_transformer.py  # Core Set Transformer model
│   ├── attention_modules.py    # Attention mechanism implementations
│   ├── train_engine.py         # Training loop and strategies
│   ├── inference_engine.py     # Prediction and evaluation
│   └── chemeleon_mp.pt         # Preloaded checkpoint file
└── results/                     # Output directory (auto-generated)
```

## 📋 Data Format

### Input CSV Structure

Your training and test data should be CSV files with: 

| Column | Description |
|--------|-------------|
| `Molecular_Composition` | Semicolon-separated molecule identifiers or SMILES strings |
| `Label` | Target value(s) for prediction |

**Example:**
```csv
Molecular_Composition,Label
CCO;CC(=O)O,0.85
c1ccccc1;CCN(CC)CC,1.23
```

### Dictionary File

For ID-to-SMILES mapping: 
```csv
id,smiles
Mol_001,CCO
Mol_002,c1ccccc1
```

## 📤 Outputs

The pipeline generates organized outputs in `results/<project_name>/`:

```
results/MyProject/
├── config_snapshot.json    # Archived configuration
├── pipeline. log            # Detailed execution log
├── features/               # Cached molecular features
├── models/                 # Saved model checkpoints
└── predictions/
    ├── predictions.csv     # Model predictions
    └── attention/          # Attention score exports (if enabled)
```

## 🔧 Supported Task Types

| Task Type | Sub-Type | Output | Use Case |
|-----------|----------|--------|----------|
| `regression` | `standard` | Continuous value | Property prediction |
| `classification` | `binary` | 0 or 1 | Yes/No predictions |
| `classification` | `multiclass` | Class index | Mutually exclusive categories |
| `multilabel` | `classification` | Multi-hot vector | Multiple concurrent labels |
| `multitask` | `regression` | Continuous value | Multiple Output heads |
| `multitask` | `classification` | 0 or 1 | Multiple Output heads  |

## 📚 Dependencies

### Required
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- RDKit
- h5py
- tqdm
