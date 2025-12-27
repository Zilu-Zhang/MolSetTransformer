# MolSetTransformer

A flexible deep learning framework for **molecular property prediction** with support for variable-length, multi-molecule inputs.  Built on the **Set Transformer** architecture, MolSetTransformer enables powerful modeling of molecular set interactions through attention mechanisms.

## ⚗️ Overview

MolSetTransformer is designed for scenarios where predictions depend on **sets of molecules** rather than individual compounds. The framework handles: 

- **Variable-length molecular inputs** (e.g., drug combinations, molecular mixtures)
- **Multiple prediction tasks**:  Regression, Binary/Multi-class Classification, and Multi-label scenarios
- **Multi-task learning** with structure-aware heads
- **Interpretable predictions** via attention score export

## ✨ Key Features

### 🧬 Flexible Molecular Featurization
- **Morgan Fingerprints** - Circular fingerprints for molecular structure encoding
- **RDKit Descriptors** - Physicochemical property descriptors (requires `descriptastorus`)
- **CheMeleon Embeddings** - Graph neural network embeddings (requires `chemprop`)
- **Pre-computed Features** - Support for HDF5-formatted custom embeddings

### 🔬 Set Transformer Architecture
- **Self-Attention Blocks (SAB)** - Capture inter-molecular interactions
- **Pooling Multihead Attention (PMA)** - Aggregate variable-sized sets into fixed representations
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

```bash
# Clone the repository
git clone https://github.com/Zilu-Zhang/MolSetTransformer. git
cd MolSetTransformer

# Install dependencies
pip install torch numpy pandas rdkit h5py tqdm

# Optional: For additional featurization methods
pip install descriptastorus  # RDKit descriptors
pip install chemprop         # CheMeleon embeddings
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
│   └── inference_engine.py     # Prediction and evaluation
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

### Dictionary File (Optional)

For ID-to-SMILES mapping: 
```csv
id,smiles
Mol_001,CCO
Mol_002,c1ccccc1
```

## ⚙️ Configuration

### Example JSON Configuration

```json
{
  "project_name": "MyMolecularProject",
  "mode": "application",
  "task":  {
    "type":  "regression",
    "sub_type": "standard"
  },
  "data":  {
    "train_path": "data/train.csv",
    "test_path": "data/test.csv",
    "mol_col": "Molecular_Composition",
    "label_col": "Label",
    "featurization": {
      "feature_pipeline": ["morgan"]
    }
  },
  "model_architecture": {
    "model_dim": 256,
    "nhead": 8,
    "num_attention_blocks": 4,
    "num_integrators": 6,
    "dropouts": {
      "input": 0.1,
      "transformer": 0.1,
      "head": 0.2
    },
    "random_seed": 42
  },
  "training": {
    "strategy": "standard",
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

## 🏗️ Architecture Details

### Model Pipeline

```
Input Molecules → Featurization → [B, M, D]
                                      ↓
                            Input Projection
                                      ↓
                         Self-Attention Blocks (SAB)
                             (Inter-molecular learning)
                                      ↓
                         Pooling Attention (PMA)
                             (Fixed-size aggregation)
                                      ↓
                           Prediction Head(s)
                                      ↓
                                  Output
```

### Attention Mechanisms

- **MoleculeSimulator (SAB)**: Self-attention for modeling interactions between molecules in a set
- **CrossIntegration (PMA)**: Cross-attention pooling with learnable seed vectors for set aggregation

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
| `multilabel` | - | Multi-hot vector | Multiple concurrent labels |
| `multitask` | - | Multiple heads | Structure-based task separation |

## 📚 Dependencies

### Required
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- RDKit
- h5py
- tqdm

### Optional
- `descriptastorus` - For RDKit descriptor generation
- `chemprop` - For CheMeleon graph embeddings

## 📖 Citation

If you use MolSetTransformer in your research, please cite: 

```bibtex
@software{molsettransformer,
  author = {Zhang, Zilu},
  title = {MolSetTransformer: Set Transformer for Molecular Property Prediction},
  year = {2025},
  url = {https://github.com/Zilu-Zhang/MolSetTransformer}
}
```

## 📝 License

This project is open source.  Please check the repository for license details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests. 

---

**Built with ❤️ for computational chemistry and drug discovery**
