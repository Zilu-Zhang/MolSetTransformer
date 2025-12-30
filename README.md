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
pip install chemprop==2.2.1
pip install tqdm==4.67.1
pip install h5py==3.15.1
```

#### Option B — conda (recommended)

Conda gives a reproducible environment.

```bash
# from the repo root (where env_cuda.yml is located)
conda env create -f environment.yml
conda activate molsettrans
```

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
Molecule_ID,SMILES
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
