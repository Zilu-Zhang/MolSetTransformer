#!/usr/bin/env python3
"""
FeatureFactory
==============

A high-level provider for generating molecular features for machine learning pipelines.
This module manages:
1.  Feature Generation: Morgan fingerprints, RDKit descriptors, and CheMeleon embeddings.
2.  Dependency Management: Graceful handling of optional packages (Chemprop, Descriptastorus).
3.  Caching: HDF5-based caching to prevent redundant computations across runs.

Dependencies:
    - rdkit (Required)
    - descriptastorus (Optional: for RDKit descriptors)
    - chemprop (Optional: for CheMeleon embeddings)
    - h5py, numpy, tqdm
"""

import hashlib
import logging
import shutil
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.request import urlretrieve

import h5py
import numpy as np
from tqdm import tqdm

# --- Dependency Configuration ---

# 1. RDKit (Mandatory)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, MolFromSmiles
    try:
        from rdkit.Chem import rdFingerprintGenerator
        RDKIT_NEW_API = True
    except ImportError:
        RDKIT_NEW_API = False
except ImportError:
    raise ImportError(
        "RDKit is required but not installed. "
        "Install via conda: conda install -c conda-forge rdkit"
    )

# 2. Descriptastorus (Optional)
try:
    from descriptastorus.descriptors import rdNormalizedDescriptors
    DESCRIPTASTORUS_AVAILABLE = True
except ImportError:
    DESCRIPTASTORUS_AVAILABLE = False

# 3. Chemprop (Optional)
try:
    import torch
    from chemprop import featurizers, nn
    from chemprop.data import BatchMolGraph
    from chemprop.models import MPNN
    from chemprop.nn import RegressionFFN
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False


# --- Constants ---

DEFAULT_MORGAN_RADIUS = 4
DEFAULT_MORGAN_NBITS = 2048

# Maps User Interface display names to internal feature keys
UI_NAME_MAP = {
    "Morgan Fingerprints": "morgan",
    "RDKit Physicochemical Properties (Normalized)": "rdkit",
    "CheMeleon Embeddings": "chemeleon"
}


def _smiles_to_key(smiles: str) -> str:
    """Generates a deterministic MD5 hash of a SMILES string for H5 group keys."""
    return hashlib.md5(smiles.encode('utf-8')).hexdigest()


class CheMeleonFingerprint:
    """
    Wrapper for generating CheMeleon embeddings using a pre-trained Chemprop model.
    
    This class handles:
    1. Automatic downloading of model checkpoints from Zenodo.
    2. Model initialization on the appropriate device (CPU/CUDA).
    3. Batch inference for molecular graphs.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the CheMeleon featurizer.

        Args:
            device: 'cpu' or 'cuda'. If None, auto-detects availability.

        Raises:
            ImportError: If 'chemprop' is not installed.
            RuntimeError: If checkpoint download fails.
        """
        if not CHEMPROP_AVAILABLE:
             raise ImportError("CheMeleon features require 'chemprop'. Please install it.")

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Checkpoint Management ---
        ckpt_dir = Path.home() / ".chemprop"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        
        if not mp_path.exists():
            logging.info("Downloading CheMeleon checkpoint from Zenodo...")
            try:
                urlretrieve(
                    r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                    mp_path,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download CheMeleon checkpoint: {e}")
                
        # --- Model Loading ---
        # `weights_only=True` is used for security when loading untrusted checkpoints
        chemeleon_mp = torch.load(mp_path, map_location=self.device, weights_only=True)
        mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
        mp.load_state_dict(chemeleon_mp['state_dict'])
        
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim), 
        )
        self.model.eval()
        self.model.to(device=self.device)

    def __call__(self, molecules: List[Any]) -> np.ndarray:
        """
        Generates embeddings for a list of molecules.

        Args:
            molecules: List of SMILES strings or RDKit Mol objects.

        Returns:
            np.ndarray: Matrix of embeddings (shape: [N, hidden_dim]).
        """
        # Ensure inputs are RDKit Mols
        mols = [MolFromSmiles(m) if isinstance(m, str) else m for m in molecules]
        
        # Create graph batch
        # Note: Caller must filter None types before passing here; BatchMolGraph crashes on None.
        bmg = BatchMolGraph([self.featurizer(m) for m in mols])
        bmg.to(device=self.model.device)
        
        with torch.no_grad():
            return self.model.fingerprint(bmg).cpu().numpy()


class FeatureFactory:
    """
    Orchestrates the generation and caching of molecular features.

    Features are stored in an HDF5 cache to improve performance on subsequent runs.
    The cache is automatically invalidated if the configuration (radius, nbits, etc.) changes.
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Args:
            config: Configuration dictionary containing:
                    - 'methods' or 'feature_types': List of features to generate.
                    - 'morgan_radius', 'morgan_nbits': Parameters for Morgan FP.
                    - 'chemeleon_device': Device for deep learning inference.
            output_dir: Path where the 'features_cache.h5' will be stored.
        """
        self.config = config or {}
        self.precomputed_path = Path(self.config['path']) if self.config.get('path') else None

        # Parse and normalize feature types
        raw_types = self.config.get('methods') or self.config.get('feature_types', ["morgan"])
        self.feature_types = []
        for t in raw_types: 
            if t in UI_NAME_MAP: 
                self.feature_types.append(UI_NAME_MAP[t])
            elif t in UI_NAME_MAP.values():
                self.feature_types.append(t)
            else:
                logging.warning(f"Unknown feature type '{t}' requested. Skipping.")

        if not self.feature_types:
            logging.warning("No valid feature types specified. Defaulting to 'morgan'.")
            self.feature_types = ["morgan"]

        # Ensure environment supports requested features
        self._validate_dependencies()

        # Configuration
        self.batch_size = int(self.config.get('batch_size', 64))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.output_dir / "features_cache.h5"

        self.morgan_radius = int(self.config.get('morgan_radius', DEFAULT_MORGAN_RADIUS))
        self.morgan_nbits = int(self.config.get('morgan_nbits', DEFAULT_MORGAN_NBITS))

        self.chemeleon_batch = int(self.config.get('chemeleon_batch', 64))
        self.chemeleon_device = self.config.get('chemeleon_device', None)

        # Initialize Generators
        self._init_rdkit()
        self._init_chemeleon()

        # Compute Dimensions & Validate Cache
        self._feature_dims = self._detect_dimensions()
        self._expected_dim = sum(self._feature_dims.values())
        self._validate_cache_with_dims()
        self._log_feature_dimensions()

    def _validate_dependencies(self):
        """Checks if optional packages are installed for requested features."""
        missing_deps = []

        if "rdkit" in self.feature_types and not DESCRIPTASTORUS_AVAILABLE:
            missing_deps.append(
                "RDKit Physicochemical Properties requires 'descriptastorus'. "
                "Install with: pip install descriptastorus"
            )

        if "chemeleon" in self.feature_types and not CHEMPROP_AVAILABLE:
            missing_deps.append(
                "CheMeleon Embeddings require 'chemprop'. "
                "Install with: pip install chemprop"
            )

        if missing_deps:
            raise ImportError("\n".join(missing_deps))

    def _init_rdkit(self):
        """Initializes the Descriptastorus engine if requested."""
        self.rdkit_gen = None
        if "rdkit" in self.feature_types:
            try: 
                self.rdkit_gen = rdNormalizedDescriptors.RDKit2DNormalized()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Descriptastorus: {e}")

    def _init_chemeleon(self):
        """Initializes the CheMeleon engine if requested."""
        self.chemeleon = None
        if "chemeleon" in self.feature_types:
            self.chemeleon = CheMeleonFingerprint(device=self.chemeleon_device)

    def _detect_dimensions(self) -> Dict[str, int]:
        """Runs a dummy molecule (Methane) to determine feature vector sizes."""
        dims = {}
        test_smiles = "C"

        for ft in self.feature_types:
            if ft == "morgan":
                dims["morgan"] = self.morgan_nbits

            elif ft == "rdkit":
                try: 
                    # Exclude validity flag (index 0) from descriptastorus output
                    _, values = self.rdkit_gen.process(test_smiles)
                    dims["rdkit"] = len(values) - 1 
                except Exception as e: 
                    raise RuntimeError(f"Failed to detect RDKit dimension: {e}")

            elif ft == "chemeleon": 
                try:
                    feature_batch = self.chemeleon([test_smiles])
                    dims["chemeleon"] = feature_batch.shape[1]
                except Exception as e:
                    raise RuntimeError(f"Failed to detect CheMeleon dimension: {e}")

        return dims

    def _log_feature_dimensions(self):
        """Prints a summary of the active feature set to the log."""
        logging.info("=" * 55)
        logging.info("FEATURE DIMENSION SUMMARY")
        logging.info("=" * 55)

        for ft in self.feature_types:
            dim = self._feature_dims.get(ft, 0)
            logging.info(f"  {ft.upper():<20} {dim:>6} dims")

        logging.info("-" * 55)
        logging.info(f"  TOTAL FEATURE VECTOR:   {self._expected_dim:>6} dims")
        logging.info("=" * 55)

    def get_expected_dim(self) -> int:
        """Returns the total size of the concatenated feature vector."""
        return self._expected_dim

    def get_feature_dims(self) -> Dict[str, int]:
        """Returns a dictionary of individual feature dimensions."""
        return self._feature_dims.copy()

    def _get_config_hash(self) -> str:
        """Creates a unique hash based on the current feature configuration."""
        sig = sorted(self.feature_types)
        sig.append(f"rad_{self.morgan_radius}")
        sig.append(f"nbits_{self.morgan_nbits}")
        for ft, dim in sorted(self._feature_dims.items()):
            sig.append(f"{ft}_{dim}")
        return hashlib.md5("_".join(sig).encode()).hexdigest()

    def _validate_cache_with_dims(self):
        """
        Compares the current config hash with the hash stored in the cache file.
        If they mismatch, the old cache is moved to a backup file to prevent data corruption.
        """
        if not self.cache_path.exists():
            return

        current_hash = self._get_config_hash()
        should_reset = False
        try:
            with h5py.File(self.cache_path, 'r') as hf: 
                if hf.attrs.get('config_hash', '') != current_hash:
                    should_reset = True
        except Exception: 
            should_reset = True

        if should_reset:
            backup_name = self.cache_path.with_suffix(f".bak_{np.random.randint(1000)}.h5")
            shutil.move(self.cache_path, backup_name)
            logging.info(f"Existing cache incompatible with current config. Moved to {backup_name}.")

    def _morgan_fp(self, smiles: str) -> np.ndarray:
        """Generates a Morgan fingerprint bit vector as a float array."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((self.morgan_nbits,), dtype=np.float32)

        if RDKIT_NEW_API:
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.morgan_radius, fpSize=self.morgan_nbits
            )
            bitvect = gen.GetFingerprint(mol)
        else: 
            bitvect = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.morgan_radius, nBits=self.morgan_nbits
            )

        # Efficient conversion to numpy
        arr = np.zeros((self.morgan_nbits,), dtype=np.float32)
        try:
            DataStructs.ConvertToNumpyArray(bitvect, arr)
        except Exception: 
            # Fallback for older RDKit versions
            bs = bitvect.ToBitString()
            for i, ch in enumerate(bs):
                arr[i] = 1.0 if ch == "1" else 0.0
        return arr

    def _rdkit_desc_process(self, smiles: str) -> np.ndarray:
        """Generates RDKit descriptors using Descriptastorus."""
        try:
            _, values = self.rdkit_gen.process(smiles)
            # Skip index 0 (boolean validity flag)
            arr = np.asarray(values[1:], dtype=np.float32)
            # Clean NaNs and Infs
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception: 
            return np.zeros((self._feature_dims["rdkit"],), dtype=np.float32)

    def batch_generate(self, smiles_list: List[str], batch_size: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]: 
        """
        Generates features for a batch of SMILES strings from scratch.
        
        Note: This method computes features even if they exist in cache. 
        Use `get_features` for cached retrieval.

        Args:
            smiles_list: List of SMILES strings.
            batch_size: (Unused in current logic but kept for interface compatibility).

        Returns:
            Dict[smiles, Dict[feature_type, np.ndarray]]
        """
        # Deduplicate inputs
        smiles_list = list(dict.fromkeys([s for s in smiles_list if s]))
        out = {}

        # 1. Pre-compute CheMeleon embeddings (Batch Inference)
        chemeleon_cache = {}
        if "chemeleon" in self.feature_types and self.chemeleon: 
            logging.info("Computing CheMeleon fingerprints in batches...")
            
            for i in tqdm(range(0, len(smiles_list), self.chemeleon_batch), desc="CheMeleon Inference"):
                subset = smiles_list[i:i + self.chemeleon_batch]
                
                # Filter valid molecules
                valid_batch_indices = []
                valid_batch_mols = []
                for idx, smi in enumerate(subset):
                    if Chem.MolFromSmiles(smi):
                        valid_batch_indices.append(idx)
                        valid_batch_mols.append(smi)

                if valid_batch_mols:
                    try:
                        fps = self.chemeleon(valid_batch_mols)
                        # Map results back to SMILES strings
                        for k, original_idx in enumerate(valid_batch_indices):
                            smi_key = subset[original_idx]
                            chemeleon_cache[smi_key] = fps[k].astype(np.float32)
                    except Exception as e: 
                        logging.error(f"CheMeleon batch failed: {e}")

        # 2. Generate remaining features (CPU-bound)
        for s in tqdm(smiles_list, desc="Generating Feature Components"):
            parts = {}
            for ft in self.feature_types:
                if ft == "morgan":
                    parts['morgan'] = self._morgan_fp(s)
                elif ft == "rdkit": 
                    parts['rdkit'] = self._rdkit_desc_process(s)
                elif ft == "chemeleon": 
                    # Retrieve pre-calculated or default to zero
                    if s in chemeleon_cache:
                        parts['chemeleon'] = chemeleon_cache[s]
                    else:
                        parts['chemeleon'] = np.zeros(
                            (self._feature_dims["chemeleon"],), dtype=np.float32
                        )
            out[s] = parts
        return out

    def get_features(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Main entry point. Retrieves features for a list of SMILES strings.
        
        Strategy:
        1. Check HDF5 cache for existing features.
        2. Identify missing molecules.
        3. Compute missing features via `batch_generate`.
        4. Update cache.
        5. Return concatenated feature vectors for all inputs.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Dict[smiles, np.ndarray]: Map of SMILES to concatenated feature vectors.
        """
        smiles_list = [s for s in smiles_list if s]
        if not smiles_list: 
            return {}

        final_features = {}

        # Initialize cache file if it doesn't exist
        if not self.cache_path.exists():
            with h5py.File(self.cache_path, 'w') as hf:
                hf.attrs['created_by'] = "FeatureFactory"
                hf.attrs['config_hash'] = self._get_config_hash()
                for ft, dim in self._feature_dims.items():
                    hf.attrs[f'dim_{ft}'] = dim

        # Identify missing features
        missing = []
        with h5py.File(self.cache_path, 'r') as hf:
            for s in smiles_list: 
                key = _smiles_to_key(s)
                if key not in hf: 
                    missing.append(s)

        # Compute and write missing features
        if missing:
            logging.info(f"Generating features for {len(missing)} new molecules.")
            generated = self.batch_generate(missing, batch_size=self.batch_size)

            with h5py.File(self.cache_path, 'a') as hf:
                for smi, components in generated.items():
                    key = _smiles_to_key(smi)
                    if key in hf: continue # Race condition check
                    
                    grp = hf.create_group(key)
                    for feat_name, feat_arr in components.items():
                        grp.create_dataset(feat_name, data=feat_arr)

        # Assemble final result
        with h5py.File(self.cache_path, 'r') as hf:
            for s in smiles_list: 
                key = _smiles_to_key(s)
                if key in hf:
                    grp = hf[key]
                    vecs = []

                    # Concatenate enabled features in order
                    for ft in self.feature_types:
                        if ft in grp:
                            vecs.append(grp[ft][()])
                        else:
                            # Handle case where feature type was added to config later
                            dim = self._feature_dims.get(ft, 0)
                            vecs.append(np.zeros((dim,), dtype=np.float32))

                    if vecs:
                        final_features[s] = np.concatenate(vecs).astype(np.float32)
                    else: 
                        final_features[s] = np.zeros((self._expected_dim,), dtype=np.float32)
                else:
                    # Fallback if generation failed entirely for a molecule
                    final_features[s] = np.zeros((self._expected_dim,), dtype=np.float32)

        return final_features