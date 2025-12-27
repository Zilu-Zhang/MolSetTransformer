#!/usr/bin/env python3
"""
FeatureFactory:  High-level provider for molecular features. 
Thread-safe singleton for CheMeleon, with automatic dimension detection. 
Strict dependency enforcement - no fallbacks. 
"""

import logging
import h5py
import numpy as np
import hashlib
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from tqdm import tqdm

# RDKit is always required
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
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
    from descriptastorus. descriptors import rdNormalizedDescriptors
    DESCRIPTASTORUS_AVAILABLE = True
except ImportError:
    DESCRIPTASTORUS_AVAILABLE = False

# Check for chemprop (required for chemeleon features)
try:
    import torch
    from chemprop import featurizers, nn
    from chemprop.models import MPNN
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
    """Thread-safe singleton for CheMeleon fingerprint generation."""
    _instance = None
    _lock = threading. Lock()

    def __new__(cls, device:  Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CheMeleonFingerprint, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, device: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if self._initialized:
                return

            if not CHEMPROP_AVAILABLE:
                raise ImportError(
                    "CheMeleon features require 'chemprop' package which is not installed.\n"
                    "Install with: pip install chemprop\n"
                    "For GPU support:  pip install chemprop[cuda]"
                )

            self. device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            # Suppress chemprop logging
            logger = logging.getLogger('chemprop')
            logger.setLevel(logging.WARNING)

            self. featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
            agg = nn.MeanAggregation()

            ckpt_dir = Path. home() / ".chemprop"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            mp_path = ckpt_dir / "chemeleon_mp.pt"

            if not mp_path.exists():
                try:
                    logging.info("Downloading CheMeleon checkpoint...")
                    from urllib.request import urlretrieve
                    urlretrieve(
                        r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                        mp_path
                    )
                    logging.info("CheMeleon checkpoint downloaded successfully.")
                except Exception as e:
                    raise RuntimeError(f"Failed to download CheMeleon checkpoint: {e}")

            chemeleon_mp = torch.load(mp_path, map_location=self. device)
            mp = nn.BondMessagePassing(**chemeleon_mp['hyper_parameters'])
            mp.load_state_dict(chemeleon_mp['state_dict'])

            from chemprop.nn import RegressionFFN
            self. model = MPNN(
                message_passing=mp,
                agg=agg,
                predictor=RegressionFFN(input_dim=mp.output_dim)
            ).to(self.device).eval()

            self._initialized = True
            logging.info(f"CheMeleon initialized on device: {self.device}")

    def compute_single(self, smiles:  str) -> Optional[np.ndarray]:
        """Compute embedding for a single SMILES.  Returns None if invalid."""
        mol = Chem. MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
        if mol is None:
            return None

        from chemprop.data import BatchMolGraph
        bmg = BatchMolGraph([self.featurizer(mol)])
        bmg.to(self.model.device)

        with torch.no_grad():
            fp = self.model.fingerprint(bmg).cpu().numpy()

        return fp[0]. astype(np.float32)

    def __call__(self, molecules: List[str]) -> Tuple[np.ndarray, List[str]]: 
        """Compute embeddings for a batch of SMILES strings."""
        valid_mols = []
        valid_smiles = []

        for m in molecules:
            try:
                mol = Chem.MolFromSmiles(m) if isinstance(m, str) else m
                if mol: 
                    valid_mols.append(mol)
                    valid_smiles. append(Chem.MolToSmiles(mol))
            except Exception: 
                continue

        if not valid_mols: 
            return np.array([]), []

        from chemprop.data import BatchMolGraph
        bmg = BatchMolGraph([self.featurizer(m) for m in valid_mols])
        bmg.to(self. model.device)

        with torch.no_grad():
            fps = self.model. fingerprint(bmg).cpu().numpy()

        return fps, valid_smiles


class FeatureFactory:
    """
    Factory class for generating molecular features.
    Supports Morgan fingerprints, RDKit descriptors, and CheMeleon embeddings. 

    Dimension handling:
    - Morgan:  User-configurable (radius, nbits)
    - RDKit: Auto-detected by computing one feature and using len()
    - CheMeleon: Auto-detected by computing one feature and using len()

    Strict dependency enforcement:
    - 'rdkit' features require descriptastorus package
    - 'chemeleon' features require chemprop package
    """

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        self.config = config or {}
        self. source = self.config.get('source', 'pipeline_builder')
        self.precomputed_path = Path(self.config['path']) if self.config.get('path') else None

        # Parse feature types from config
        raw_types = self.config. get('methods') or self.config.get('feature_types', ["morgan"])
        self.feature_types = []
        for t in raw_types: 
            if t in UI_NAME_MAP: 
                self.feature_types.append(UI_NAME_MAP[t])
            elif t in UI_NAME_MAP. values():
                self.feature_types. append(t)
            else:
                logging.warning(f"Unknown feature type '{t}' requested.  Skipping.")

        if not self.feature_types:
            logging.warning("No valid feature types specified.  Defaulting to 'morgan'.")
            self.feature_types = ["morgan"]

        # Validate dependencies BEFORE proceeding
        self._validate_dependencies()

        self.batch_size = int(self.config. get('batch_size', 64))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.output_dir / "features_cache.h5"

        # Morgan parameters (user-configurable)
        self.morgan_radius = int(self.config. get('morgan_radius', DEFAULT_MORGAN_RADIUS))
        self.morgan_nbits = int(self.config.get('morgan_nbits', DEFAULT_MORGAN_NBITS))

        # CheMeleon parameters
        self.chemeleon_batch = int(self.config.get('chemeleon_batch', 64))
        self.chemeleon_device = self.config. get('chemeleon_device', None)

        # Initialize components
        self._init_rdkit()
        self._init_chemeleon()

        # Detect dimensions by computing one sample for each feature type
        self._feature_dims = self._detect_dimensions()

        # Calculate total expected dimension
        self._expected_dim = sum(self._feature_dims.values())

        # Now validate cache with known dimensions
        self._validate_cache_with_dims()

        # Log detected dimensions
        self._log_feature_dimensions()

    def _validate_dependencies(self):
        """Validate that all required dependencies are installed."""
        missing_deps = []

        if "rdkit" in self.feature_types and not DESCRIPTASTORUS_AVAILABLE:
            missing_deps.append(
                "RDKit Physicochemical Properties requires 'descriptastorus' package.\n"
                "  Install with: pip install descriptastorus\n"
                "  Or remove 'rdkit' from feature types in your configuration."
            )

        if "chemeleon" in self. feature_types and not CHEMPROP_AVAILABLE:
            missing_deps.append(
                "CheMeleon Embeddings require 'chemprop' package.\n"
                "  Install with: pip install chemprop\n"
                "  For GPU support: pip install chemprop[cuda]\n"
                "  Or remove 'chemeleon' from feature types in your configuration."
            )

        if missing_deps:
            error_msg = (
                "Missing required dependencies for requested feature types:\n\n" +
                "\n\n".join(missing_deps)
            )
            raise ImportError(error_msg)

    def _init_rdkit(self):
        """Initialize RDKit descriptor generator."""
        self. rdkit_gen = None

        if "rdkit" in self.feature_types:
            try: 
                self.rdkit_gen = rdNormalizedDescriptors.RDKit2DNormalized()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Descriptastorus RDKit2DNormalized: {e}\n"
                    "Please ensure descriptastorus is properly installed."
                )

    def _init_chemeleon(self):
        """Initialize CheMeleon if requested."""
        self.chemeleon = None

        if "chemeleon" in self.feature_types:
            self.chemeleon = CheMeleonFingerprint(device=self.chemeleon_device)

    def _detect_dimensions(self) -> Dict[str, int]:
        """
        Detect dimensions for all feature types by computing one sample.
        Simple approach: compute the feature, use len() to get dimension.
        """
        dims = {}
        test_smiles = "C"  # Methane - simplest valid molecule

        for ft in self.feature_types:
            if ft == "morgan":
                # Morgan dimension is user-specified
                dims["morgan"] = self. morgan_nbits

            elif ft == "rdkit":
                # Compute one RDKit feature and get its length
                try: 
                    _, values = self.rdkit_gen. process(test_smiles)
                    # First value is validity flag, rest are descriptors
                    feature = np.asarray(values[1:], dtype=np.float32)
                    dims["rdkit"] = len(feature)
                except Exception as e: 
                    raise RuntimeError(f"Failed to detect RDKit dimension: {e}")

            elif ft == "chemeleon": 
                # Compute one CheMeleon feature and get its length
                try:
                    feature = self.chemeleon.compute_single(test_smiles)
                    if feature is None:
                        raise RuntimeError("CheMeleon returned None for test molecule")
                    dims["chemeleon"] = len(feature)
                except Exception as e:
                    raise RuntimeError(f"Failed to detect CheMeleon dimension: {e}")

        return dims

    def _log_feature_dimensions(self):
        """Log detected feature dimensions in a formatted way."""
        logging.info("=" * 55)
        logging.info("FEATURE DIMENSION SUMMARY")
        logging.info("=" * 55)

        for ft in self.feature_types:
            dim = self._feature_dims.get(ft, 0)
            if ft == "morgan": 
                logging.info(
                    f"  Morgan Fingerprints:    {dim: >6} bits  "
                    f"(radius={self.morgan_radius}, user-configurable)"
                )
            elif ft == "rdkit": 
                logging.info(
                    f"  RDKit Descriptors:     {dim:>6} dims  "
                    f"(auto-detected)"
                )
            elif ft == "chemeleon":
                logging.info(
                    f"  CheMeleon Embeddings:  {dim:>6} dims  "
                    f"(auto-detected)"
                )

        logging.info("-" * 55)
        logging.info(f"  TOTAL FEATURE VECTOR:   {self._expected_dim:>6} dims")
        logging.info("=" * 55)

    def get_expected_dim(self) -> int:
        """Returns the total expected feature dimension."""
        return self._expected_dim

    def get_feature_dims(self) -> Dict[str, int]:
        """Returns a dictionary of individual feature dimensions."""
        return self._feature_dims. copy()

    def _get_config_hash(self) -> str:
        """Generate hash of current configuration for cache validation."""
        sig = sorted(self.feature_types)
        sig. append(f"rad_{self.morgan_radius}")
        sig.append(f"nbits_{self.morgan_nbits}")
        # Include detected dimensions in hash for consistency
        for ft, dim in sorted(self._feature_dims.items()):
            sig.append(f"{ft}_{dim}")
        return hashlib.md5("_".join(sig).encode()).hexdigest()

    def _validate_cache_with_dims(self):
        """Validate cache after dimensions are detected."""
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
            backup_name = self.cache_path.with_suffix(f".bak_{np.random.randint(1000)}. h5")
            shutil.move(self.cache_path, backup_name)
            logging. info(f"Old cache moved to {backup_name} (config changed)")

    def _morgan_fp(self, smiles:  str) -> np.ndarray:
        """Generate Morgan fingerprint for a SMILES string."""
        mol = Chem. MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"Invalid SMILES for Morgan FP: {smiles}")
            return np.zeros((self. morgan_nbits,), dtype=np.float32)

        if RDKIT_NEW_API:
            gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.morgan_radius,
                fpSize=self.morgan_nbits
            )
            bitvect = gen. GetFingerprint(mol)
        else: 
            bitvect = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.morgan_radius,
                nBits=self. morgan_nbits
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
        """Generate RDKit descriptors for a SMILES string."""
        try:
            _, values = self.rdkit_gen.process(smiles)
            arr = np.asarray(values[1:], dtype=np. float32)  # Skip validity flag
            # Replace NaN/Inf with 0
            arr = np. nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr
        except Exception as e: 
            logging.warning(f"RDKit descriptor generation failed for {smiles}: {e}")
            return np.zeros((self._feature_dims["rdkit"],), dtype=np.float32)

    def _chemeleon_single(self, smiles:  str) -> np.ndarray:
        """Generate CheMeleon embedding for a single SMILES string."""
        result = self.chemeleon.compute_single(smiles)
        if result is None:
            logging.warning(f"CheMeleon failed for:  {smiles}")
            return np.zeros((self._feature_dims["chemeleon"],), dtype=np.float32)
        return result

    def batch_generate(self, smiles_list: List[str], batch_size: Optional[int] = None) -> Dict[str, Dict[str, np. ndarray]]: 
        """Generate features for a batch of SMILES strings."""
        smiles_list = list(dict.fromkeys([s for s in smiles_list if s]))
        out = {}

        # Pre-compute CheMeleon embeddings in batches
        chemeleon_cache = {}
        if "chemeleon" in self.feature_types and self.chemeleon: 
            logging.info("Computing CheMeleon fingerprints in batches...")
            for i in tqdm(range(0, len(smiles_list), self.chemeleon_batch), desc="CheMeleon Inference"):
                subset = smiles_list[i:i + self.chemeleon_batch]
                try:
                    fps, valid_smiles = self.chemeleon(subset)
                    if fps. size > 0:
                        for j, smi in enumerate(valid_smiles):
                            chemeleon_cache[smi] = np.asarray(fps[j], dtype=np. float32)
                except Exception as e: 
                    logging. error(f"CheMeleon batch failed: {e}")
                    raise RuntimeError(f"CheMeleon feature generation failed:  {e}")

        # Generate features for each molecule
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
                        # Try canonical form
                        mol = Chem. MolFromSmiles(s)
                        canonical = Chem. MolToSmiles(mol) if mol else None
                        if canonical and canonical in chemeleon_cache:
                            parts['chemeleon'] = chemeleon_cache[canonical]
                        else: 
                            logging.warning(f"CheMeleon embedding not found for:  {s}")
                            parts['chemeleon'] = np. zeros(
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
            with h5py. File(self.cache_path, 'w') as hf:
                hf.attrs['created_by'] = "FeatureFactory"
                hf. attrs['config_hash'] = self._get_config_hash()
                # Store detected dimensions in cache metadata
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
            logging. info(f"Generating features for {len(missing)} new molecules.")
            generated = self.batch_generate(missing, batch_size=self.batch_size)

            with h5py.File(self.cache_path, 'a') as hf:
                for smi, components in generated.items():
                    key = _smiles_to_key(smi)
                    if key in hf:
                        continue
                    grp = hf. create_group(key)

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
                            logging.error(f"Feature type '{ft}' missing in cache for {s}")
                            dim = self._feature_dims. get(ft, 0)
                            vecs.append(np.zeros((dim,), dtype=np.float32))

                    if vecs:
                        try:
                            final_features[s] = np.concatenate(vecs).astype(np. float32)
                        except ValueError as e:
                            logging.error(f"Feature concatenation failed for {s}: {e}")
                            final_features[s] = np.zeros((self._expected_dim,), dtype=np.float32)
                    else: 
                        final_features[s] = np.zeros((self._expected_dim,), dtype=np.float32)
                else:
                    logging.error(f"SMILES not found in cache after generation:  {s}")
                    final_features[s] = np. zeros((self._expected_dim,), dtype=np.float32)

        return final_features