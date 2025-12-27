#!/usr/bin/env python3
"""
FeatureFactory:  High-level provider for molecular features. 
Supports Morgan fingerprints, RDKit descriptors, and CheMeleon embeddings.
"""

import logging
import h5py
import numpy as np
import hashlib
import shutil
import threading
from pathlib import Path
from urllib.request import urlretrieve
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm

# --- Dependency Checks ---

# RDKit is always required
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
        "Install with:  conda install -c conda-forge rdkit"
    )

# Check for descriptastorus (required for rdkit features)
try:
    from descriptastorus.descriptors import rdNormalizedDescriptors
    DESCRIPTASTORUS_AVAILABLE = True
except ImportError:
    DESCRIPTASTORUS_AVAILABLE = False

# Check for chemprop (required for chemeleon features)
try:
    import torch
    from chemprop import featurizers, nn
    from chemprop.data import BatchMolGraph
    from chemprop.models import MPNN
    from chemprop.nn import RegressionFFN
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False


DEFAULT_MORGAN_RADIUS = 4
DEFAULT_MORGAN_NBITS = 2048

UI_NAME_MAP = {
    "Morgan Fingerprints": "morgan",
    "RDKit Physicochemical Properties (Normalized)": "rdkit",
    "CheMeleon Embeddings": "chemeleon"
}


def _smiles_to_key(smiles: str) -> str:
    """Generates a fixed-length MD5 hash for the H5 group key."""
    return hashlib.md5(smiles.encode('utf-8')).hexdigest()


class CheMeleonFingerprint:
    """
    Simplified CheMeleon embedding generator.
    Adapted from chemeleon_fingerprint.py
    """
    def __init__(self, device: Optional[str] = None):
        if not CHEMPROP_AVAILABLE:
             raise ImportError("CheMeleon features require 'chemprop'. Please install it.")

        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        
        # Determine device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Checkpoint handling
        ckpt_dir = Path.home() / ".chemprop"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        
        if not mp_path.exists():
            logging.info("Downloading CheMeleon checkpoint...")
            try:
                urlretrieve(
                    r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                    mp_path,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download CheMeleon checkpoint: {e}")
                
        # Load Model
        # using weights_only=True for security as in the provided script
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
        Generates embeddings for a list of SMILES or RDKit Mols.
        """
        # Convert strings to Mols if necessary (Handle mixed input)
        mols = [MolFromSmiles(m) if isinstance(m, str) else m for m in molecules]
        
        # Note: BatchMolGraph will fail if any mol is None. 
        # The caller (FeatureFactory) handles filtering valid molecules.
        bmg = BatchMolGraph([self.featurizer(m) for m in mols])
        bmg.to(device=self.model.device)
        
        with torch.no_grad():
            # Using cpu().numpy() for compatibility across torch versions
            return self.model.fingerprint(bmg).cpu().numpy()


class FeatureFactory:
    """
    Factory class for generating molecular features.
    Supports Morgan fingerprints, RDKit descriptors, and CheMeleon embeddings. 
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config or {}
        self.source = self.config.get('source', 'pipeline_builder')
        self.precomputed_path = Path(self.config['path']) if self.config.get('path') else None

        # Parse feature types from config
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

        # Validate dependencies (Strict check as requested)
        self._validate_dependencies()

        self.batch_size = int(self.config.get('batch_size', 64))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.output_dir / "features_cache.h5"

        # Morgan parameters (user-configurable)
        self.morgan_radius = int(self.config.get('morgan_radius', DEFAULT_MORGAN_RADIUS))
        self.morgan_nbits = int(self.config.get('morgan_nbits', DEFAULT_MORGAN_NBITS))

        # CheMeleon parameters
        self.chemeleon_batch = int(self.config.get('chemeleon_batch', 64))
        self.chemeleon_device = self.config.get('chemeleon_device', None)

        # Initialize components
        self._init_rdkit()
        self._init_chemeleon()

        # Detect dimensions 
        self._feature_dims = self._detect_dimensions()
        self._expected_dim = sum(self._feature_dims.values())

        # Validate cache
        self._validate_cache_with_dims()
        self._log_feature_dimensions()

    def _validate_dependencies(self):
        """Validate that all required dependencies are installed."""
        missing_deps = []

        if "rdkit" in self.feature_types and not DESCRIPTASTORUS_AVAILABLE:
            missing_deps.append(
                "RDKit Physicochemical Properties requires 'descriptastorus' package.\n"
                "  Install with: pip install descriptastorus"
            )

        if "chemeleon" in self.feature_types and not CHEMPROP_AVAILABLE:
            missing_deps.append(
                "CheMeleon Embeddings require 'chemprop' package.\n"
                "  Install with: pip install chemprop"
            )

        if missing_deps:
            raise ImportError("\n".join(missing_deps))

    def _init_rdkit(self):
        """Initialize RDKit descriptor generator."""
        self.rdkit_gen = None
        if "rdkit" in self.feature_types:
            try: 
                self.rdkit_gen = rdNormalizedDescriptors.RDKit2DNormalized()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Descriptastorus: {e}")

    def _init_chemeleon(self):
        """Initialize CheMeleon if requested."""
        self.chemeleon = None
        if "chemeleon" in self.feature_types:
            self.chemeleon = CheMeleonFingerprint(device=self.chemeleon_device)

    def _detect_dimensions(self) -> Dict[str, int]:
        """Detect dimensions by computing one sample."""
        dims = {}
        test_smiles = "C"  # Methane

        for ft in self.feature_types:
            if ft == "morgan":
                dims["morgan"] = self.morgan_nbits

            elif ft == "rdkit":
                try: 
                    _, values = self.rdkit_gen.process(test_smiles)
                    dims["rdkit"] = len(values) - 1 # Exclude validity flag
                except Exception as e: 
                    raise RuntimeError(f"Failed to detect RDKit dimension: {e}")

            elif ft == "chemeleon": 
                try:
                    # New API returns array directly
                    # Pass as list to match __call__ expectation
                    feature_batch = self.chemeleon([test_smiles])
                    dims["chemeleon"] = feature_batch.shape[1]
                except Exception as e:
                    raise RuntimeError(f"Failed to detect CheMeleon dimension: {e}")

        return dims

    def _log_feature_dimensions(self):
        """Log detected feature dimensions."""
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
        return self._expected_dim

    def get_feature_dims(self) -> Dict[str, int]:
        return self._feature_dims.copy()

    def _get_config_hash(self) -> str:
        """Generate hash of configuration for cache validation."""
        sig = sorted(self.feature_types)
        sig.append(f"rad_{self.morgan_radius}")
        sig.append(f"nbits_{self.morgan_nbits}")
        for ft, dim in sorted(self._feature_dims.items()):
            sig.append(f"{ft}_{dim}")
        return hashlib.md5("_".join(sig).encode()).hexdigest()

    def _validate_cache_with_dims(self):
        """Reset cache if configuration has changed."""
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
            logging.info(f"Old cache moved to {backup_name} (config changed)")

    def _morgan_fp(self, smiles: str) -> np.ndarray:
        """Generate Morgan fingerprint."""
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

        arr = np.zeros((self.morgan_nbits,), dtype=np.float32)
        try:
            DataStructs.ConvertToNumpyArray(bitvect, arr)
        except Exception: 
            bs = bitvect.ToBitString()
            for i, ch in enumerate(bs):
                arr[i] = 1.0 if ch == "1" else 0.0
        return arr

    def _rdkit_desc_process(self, smiles: str) -> np.ndarray:
        """Generate RDKit descriptors."""
        try:
            _, values = self.rdkit_gen.process(smiles)
            arr = np.asarray(values[1:], dtype=np.float32)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception: 
            return np.zeros((self._feature_dims["rdkit"],), dtype=np.float32)

    def batch_generate(self, smiles_list: List[str], batch_size: Optional[int] = None) -> Dict[str, Dict[str, np.ndarray]]: 
        """Generate features for a batch of SMILES strings."""
        smiles_list = list(dict.fromkeys([s for s in smiles_list if s]))
        out = {}

        # Pre-compute CheMeleon embeddings
        chemeleon_cache = {}
        if "chemeleon" in self.feature_types and self.chemeleon: 
            logging.info("Computing CheMeleon fingerprints in batches...")
            for i in tqdm(range(0, len(smiles_list), self.chemeleon_batch), desc="CheMeleon Inference"):
                subset = smiles_list[i:i + self.chemeleon_batch]
                
                # Filter valid molecules for CheMeleon batch processing
                valid_batch_indices = []
                valid_batch_mols = []
                for idx, smi in enumerate(subset):
                    # We must validate because CheMeleonFeaturizer expects valid inputs
                    if Chem.MolFromSmiles(smi):
                        valid_batch_indices.append(idx)
                        valid_batch_mols.append(smi)

                if valid_batch_mols:
                    try:
                        # New API Call: returns np.ndarray directly
                        fps = self.chemeleon(valid_batch_mols)
                        
                        # Map back to SMILES using indices
                        for k, original_idx in enumerate(valid_batch_indices):
                            smi_key = subset[original_idx]
                            chemeleon_cache[smi_key] = fps[k].astype(np.float32)
                            
                    except Exception as e: 
                        logging.error(f"CheMeleon batch failed: {e}")

        # Assemble features
        for s in tqdm(smiles_list, desc="Generating Feature Components"):
            parts = {}
            for ft in self.feature_types:
                if ft == "morgan":
                    parts['morgan'] = self._morgan_fp(s)
                elif ft == "rdkit": 
                    parts['rdkit'] = self._rdkit_desc_process(s)
                elif ft == "chemeleon": 
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
        Get features for a list of SMILES strings. 
        Uses cache when available, generates missing features. 
        """
        smiles_list = [s for s in smiles_list if s]
        if not smiles_list: 
            return {}

        final_features = {}

        # Initialize cache file if needed
        if not self.cache_path.exists():
            with h5py.File(self.cache_path, 'w') as hf:
                hf.attrs['created_by'] = "FeatureFactory"
                hf.attrs['config_hash'] = self._get_config_hash()
                for ft, dim in self._feature_dims.items():
                    hf.attrs[f'dim_{ft}'] = dim

        # Find missing SMILES
        missing = []
        with h5py.File(self.cache_path, 'r') as hf:
            for s in smiles_list: 
                key = _smiles_to_key(s)
                if key not in hf: 
                    missing.append(s)

        # Generate missing features
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

        # Load features from cache
        with h5py.File(self.cache_path, 'r') as hf:
            for s in smiles_list: 
                key = _smiles_to_key(s)
                if key in hf:
                    grp = hf[key]
                    vecs = []

                    for ft in self.feature_types:
                        if ft in grp:
                            vecs.append(grp[ft][()])
                        else:
                            dim = self._feature_dims.get(ft, 0)
                            vecs.append(np.zeros((dim,), dtype=np.float32))

                    if vecs:
                        final_features[s] = np.concatenate(vecs).astype(np.float32)
                    else: 
                        final_features[s] = np.zeros((self._expected_dim,), dtype=np.float32)
                else:
                    final_features[s] = np.zeros((self._expected_dim,), dtype=np.float32)

        return final_features