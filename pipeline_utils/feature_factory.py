#!/usr/bin/env python3
"""
FeatureFactory
==============

A high-level provider for generating molecular features for machine learning pipelines.
This module manages:
1.  Feature Generation: Morgan fingerprints, RDKit descriptors, and CheMeleon embeddings.
2.  Dependency Management: Enforces RDKit requirement; handles optional packages gracefully.
3.  Caching: HDF5-based caching to prevent redundant computations across runs.
"""

import hashlib
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from urllib.request import urlretrieve

import h5py
import numpy as np

# --- Dependency Configuration ---

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, MolFromSmiles
    try:
        from rdkit.Chem import rdFingerprintGenerator
        RDKIT_NEW_API = True
    except ImportError:
        RDKIT_NEW_API = False
except ImportError:
    raise ImportError("RDKit is required but not installed.")

try:
    from descriptastorus.descriptors import rdNormalizedDescriptors
    DESCRIPTASTORUS_AVAILABLE = True
except ImportError:
    DESCRIPTASTORUS_AVAILABLE = False

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
    Checks for the model checkpoint in the script directory; downloads if missing.
    """

    def __init__(self, device: Optional[str] = None):
        if not CHEMPROP_AVAILABLE:
             raise ImportError("CheMeleon features require 'chemprop'.")

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # --- Checkpoint Management ---
        # Look for checkpoint in the same directory as this script
        script_dir = Path(__file__).resolve().parent
        mp_path = script_dir / "chemeleon_mp.pt"
        
        if not mp_path.exists():
            logging.info(f"CheMeleon checkpoint not found at {mp_path}. Downloading from Zenodo...")
            try:
                urlretrieve(
                    r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                    mp_path,
                )
                logging.info("Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download CheMeleon checkpoint: {e}")
                
        # --- Model Loading ---
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
        mols = [MolFromSmiles(m) if isinstance(m, str) else m for m in molecules]
        # Note: Invalid SMILES strings will result in None values here, which may cause the featurizer to crash.
        bmg = BatchMolGraph([self.featurizer(m) for m in mols])
        bmg.to(device=self.model.device)
        
        with torch.no_grad():
            return self.model.fingerprint(bmg).cpu().numpy()


class FeatureFactory:
    """
    Orchestrates the generation and caching of molecular features.
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config or {}
        
        # Parse feature types to preserve user-defined order
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

        self._validate_dependencies()

        self.batch_size = int(self.config.get('batch_size', 64))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.output_dir / "features_cache.h5"

        self.morgan_radius = int(self.config.get('morgan_radius', DEFAULT_MORGAN_RADIUS))
        self.morgan_nbits = int(self.config.get('morgan_nbits', DEFAULT_MORGAN_NBITS))

        self.chemeleon_batch = int(self.config.get('chemeleon_batch', 64))
        self.chemeleon_device = self.config.get('chemeleon_device', None)

        self._init_rdkit()
        self._init_chemeleon()

        # Dimension checks and logging will follow self.feature_types order
        self._feature_dims = self._detect_dimensions()
        self._expected_dim = sum(self._feature_dims.values())
        self._validate_cache_with_dims()
        self._log_feature_dimensions()

    def _validate_dependencies(self):
        missing_deps = []
        if "rdkit" in self.feature_types and not DESCRIPTASTORUS_AVAILABLE:
            missing_deps.append("RDKit Physicochemical Properties requires 'descriptastorus'.")
        if "chemeleon" in self.feature_types and not CHEMPROP_AVAILABLE:
            missing_deps.append("CheMeleon Embeddings require 'chemprop'.")
        if missing_deps:
            raise ImportError("\n".join(missing_deps))

    def _init_rdkit(self):
        self.rdkit_gen = None
        if "rdkit" in self.feature_types:
            try: 
                self.rdkit_gen = rdNormalizedDescriptors.RDKit2DNormalized()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Descriptastorus: {e}")

    def _init_chemeleon(self):
        self.chemeleon = None
        if "chemeleon" in self.feature_types:
            self.chemeleon = CheMeleonFingerprint(device=self.chemeleon_device)

    def _detect_dimensions(self) -> Dict[str, int]:
        dims = {}
        test_smiles = "C"

        # Calculates dimensions in the order defined by self.feature_types
        for ft in self.feature_types:
            if ft == "morgan":
                dims["morgan"] = self.morgan_nbits
            elif ft == "rdkit":
                try: 
                    results = self.rdkit_gen.process(test_smiles)
                    dims["rdkit"] = len(results) - 1 
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
        logging.info("=" * 55)
        logging.info("FEATURE DIMENSION SUMMARY")
        logging.info("=" * 55)
        # Displays dimensions in the order defined by self.feature_types
        for ft in self.feature_types:
            dim = self._feature_dims.get(ft, 0)
            logging.info(f"  {ft.upper():<20} {dim:>6} dims")
        logging.info("-" * 55)
        logging.info(f"  TOTAL FEATURE VECTOR:   {self._expected_dim:>6} dims")
        logging.info("=" * 55)

    def get_expected_dim(self) -> int:
        return self._expected_dim

    def get_feature_dims(self) -> Dict[str, int]:
        return self._feature_dims.copy()

    def _get_config_hash(self) -> str:
        sig = sorted(self.feature_types)
        sig.append(f"rad_{self.morgan_radius}")
        sig.append(f"nbits_{self.morgan_nbits}")
        for ft, dim in sorted(self._feature_dims.items()):
            sig.append(f"{ft}_{dim}")
        return hashlib.md5("_".join(sig).encode()).hexdigest()

    def _validate_cache_with_dims(self):
        if not self.cache_path.exists(): return
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
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return np.zeros((self.morgan_nbits,), dtype=np.float32)

        if RDKIT_NEW_API:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.morgan_radius, fpSize=self.morgan_nbits)
            bitvect = gen.GetFingerprint(mol)
        else: 
            bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.morgan_radius, nBits=self.morgan_nbits)

        arr = np.zeros((self.morgan_nbits,), dtype=np.float32)
        try:
            DataStructs.ConvertToNumpyArray(bitvect, arr)
        except Exception: 
            bs = bitvect.ToBitString()
            for i, ch in enumerate(bs): arr[i] = 1.0 if ch == "1" else 0.0
        return arr

    def _rdkit_desc_process(self, smiles: str) -> np.ndarray:
        try:
            results = self.rdkit_gen.process(smiles)
            arr = np.asarray(results[1:], dtype=np.float32)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception: 
            return np.zeros((self._feature_dims["rdkit"],), dtype=np.float32)

    def batch_generate(self, smiles_list: List[str], batch_size: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]: 
        """
        Generates features for a batch of SMILES strings from scratch.
        Calculates features strictly in the order defined by feature_types.
        """
        smiles_list = list(dict.fromkeys([s for s in smiles_list if s]))
        
        # Initialize storage for all features
        all_features = {s: {} for s in smiles_list}

        # Iterate strictly in user-defined order
        for ft in self.feature_types:
            
            # 1. CheMeleon (Batch Optimized)
            if ft == "chemeleon" and self.chemeleon:
                logging.info("  > Calculating CheMeleon Embeddings...")
                chemeleon_results = {}
                for i in range(0, len(smiles_list), self.chemeleon_batch):
                    subset = smiles_list[i:i + self.chemeleon_batch]
                    valid_batch_indices = []
                    valid_batch_mols = []
                    for idx, smi in enumerate(subset):
                        if Chem.MolFromSmiles(smi):
                            valid_batch_indices.append(idx)
                            valid_batch_mols.append(smi)

                    if valid_batch_mols:
                        try:
                            fps = self.chemeleon(valid_batch_mols)
                            for k, original_idx in enumerate(valid_batch_indices):
                                smi_key = subset[original_idx]
                                chemeleon_results[smi_key] = fps[k].astype(np.float32)
                        except Exception as e: 
                            logging.error(f"CheMeleon batch failed: {e}")
                
                # Assign results
                for s in smiles_list:
                    val = chemeleon_results.get(s, np.zeros((self._feature_dims["chemeleon"],), dtype=np.float32))
                    all_features[s]['chemeleon'] = val

            # 2. Morgan Fingerprints
            elif ft == "morgan":
                logging.info("  > Calculating Morgan Fingerprints...")
                for s in smiles_list:
                    all_features[s]['morgan'] = self._morgan_fp(s)

            # 3. RDKit Descriptors
            elif ft == "rdkit":
                logging.info("  > Calculating RDKit Descriptors...")
                for s in smiles_list:
                    all_features[s]['rdkit'] = self._rdkit_desc_process(s)

        return all_features

    def get_features(self, smiles_list: List[str]) -> Dict[str, np.ndarray]:
        """
        Retrieves features for a list of SMILES, computing only what is missing from cache.
        """
        smiles_list = [s for s in smiles_list if s]
        if not smiles_list: return {}

        final_features = {}

        if not self.cache_path.exists():
            with h5py.File(self.cache_path, 'w') as hf:
                hf.attrs['created_by'] = "FeatureFactory"
                hf.attrs['config_hash'] = self._get_config_hash()
                for ft, dim in self._feature_dims.items():
                    hf.attrs[f'dim_{ft}'] = dim

        missing = []
        with h5py.File(self.cache_path, 'r') as hf:
            for s in smiles_list: 
                key = _smiles_to_key(s)
                if key not in hf: missing.append(s)

        if missing:
            logging.info(f"Generating features for {len(missing)} new molecules.")
            generated = self.batch_generate(missing, batch_size=self.batch_size)

            with h5py.File(self.cache_path, 'a') as hf:
                for smi, components in generated.items():
                    key = _smiles_to_key(smi)
                    if key in hf: continue 
                    grp = hf.create_group(key)
                    for feat_name, feat_arr in components.items():
                        grp.create_dataset(feat_name, data=feat_arr)

        with h5py.File(self.cache_path, 'r') as hf:
            for s in smiles_list: 
                key = _smiles_to_key(s)
                if key in hf:
                    grp = hf[key]
                    vecs = []
                    # Concatenate strictly in user-defined order
                    for ft in self.feature_types:
                        if ft in grp: vecs.append(grp[ft][()])
                        else:
                            dim = self._feature_dims.get(ft, 0)
                            vecs.append(np.zeros((dim,), dtype=np.float32))
                    if vecs: final_features[s] = np.concatenate(vecs).astype(np.float32)
                    else: final_features[s] = np.zeros((self._expected_dim,), dtype=np.float32)
                else:
                    final_features[s] = np.zeros((self._expected_dim,), dtype=np.float32)

        return final_features