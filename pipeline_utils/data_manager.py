#!/usr/bin/env python3
"""
DataManager Module

Handles data ingestion (CSV), multi-label merging, schema validation, 
auto-detection of classification tasks, and the creation of PyTorch DataLoaders.
"""

import csv
import hashlib
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .feature_factory import FeatureFactory


def smiles_to_key(smiles: str) -> str:
    """Generates a consistent hash key for a given SMILES string."""
    return hashlib.md5(smiles.encode('utf-8')).hexdigest()


class LengthGroupedBatchSampler(Sampler):
    """
    Sampler that yields batches where all samples have the same number of molecules.
    
    This is essential for structure-based multi-task learning (or Set Transformers) 
    to ensure correct head routing and avoid unnecessary padding artifacts.
    """
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by the length of their 'smiles_list'
        self.groups = {}
        for idx in range(len(data_source)):
            # Direct DataFrame access avoids overhead of loading features
            row = data_source.data.iloc[idx]
            l = len(row.get('smiles_list', []))
            
            if l not in self.groups:
                self.groups[l] = []
            self.groups[l].append(idx)

    def __iter__(self):
        batches = []
        for l, indices in self.groups.items():
            indices_local = list(indices)
            if self.shuffle:
                np.random.shuffle(indices_local)
            
            # Chunk indices into batches
            for i in range(0, len(indices_local), self.batch_size):
                batches.append(indices_local[i:i + self.batch_size])
        
        if self.shuffle:
            np.random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self):
        return sum((len(g) + self.batch_size - 1) // self.batch_size for g in self.groups.values())


class DataManager: 
    """
    Orchestrates data loading, validation, feature caching, and DataLoader creation.
    """
    def __init__(self, data_config: dict, feature_cache_dir: Path, task_config: dict):
        self.config = data_config or {}
        self.task_config = task_config
        self.cache_dir = Path(feature_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        feature_cfg = self.config.get('featurization', {})
        self.feature_factory = FeatureFactory(feature_cfg, self.cache_dir)
        
        # State detected during preparation
        self.label_map = None 
        self.num_classes = 1

    def _load_csv_safe(self, path: str) -> pd.DataFrame:
        """Helper to load CSVs with error handling."""
        try:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"CSV file is empty: {path}")
            return df
        except Exception as e:
            logging.error(f"Failed to load CSV at {path}: {e}")
            raise

    def _merge_duplicate_inputs(self, df: pd.DataFrame, mol_col: str, label_col: str, sep: str = ';') -> pd.DataFrame:
        """
        Groups data by molecular composition and merges labels.
        Crucial for multi-label tasks where data might initially be in 'long' format.
        """
        logging.info("Scanning for duplicate inputs to merge (Multi-Label Strategy)...")
        
        def merge_labels(series):
            unique_labels = set()
            for item in series:
                # Support both pre-separated strings and raw values
                parts = str(item).split(sep)
                for p in parts:
                    p_clean = p.strip()
                    if p_clean:
                        unique_labels.add(p_clean)
            return sep.join(sorted(list(unique_labels)))

        original_count = len(df)
        df_merged = df.groupby(mol_col)[label_col].apply(merge_labels).reset_index()
        
        new_count = len(df_merged)
        if new_count < original_count:
            logging.info(f"  > Merged {original_count} rows into {new_count} unique input samples.")
        else:
            logging.info("  > No duplicates found. Data kept as is.")
            
        return df_merged

    def validate_data_coverage(self, train_path: str, test_path: str):
        """
        Validates schema integrity and dictionary coverage.
        Fails early if critical columns are missing or if dictionary mapping coverage is too low.
        """
        logging.info("--- Starting Pre-flight Data Validation ---")
        
        mol_col = self.config.get('mol_col', 'Molecular_Composition')
        label_col = self.config.get('label_col', 'Label')
        
        logging.info(f"Validating against configured columns -> Molecule: '{mol_col}', Label: '{label_col}'")

        # 1. Load Data for Inspection
        try:
            df_train = self._load_csv_safe(train_path)
            df_test = self._load_csv_safe(test_path)
        except Exception as e:
            raise RuntimeError(f"Data Validation Failed: Could not load files. {e}")

        # 2. Schema Integrity Check
        for split_name, df in [("Train", df_train), ("Test", df_test)]:
            missing_cols = [c for c in [mol_col, label_col] if c not in df.columns]
            if missing_cols:
                raise ValueError(f"[{split_name}] Critical columns missing: {missing_cols}. Found in CSV: {list(df.columns)}")
            
            if df[mol_col].dropna().empty:
                raise ValueError(f"[{split_name}] Molecule column '{mol_col}' is entirely empty or null.")
            if df[label_col].dropna().empty:
                raise ValueError(f"[{split_name}] Label column '{label_col}' is entirely empty or null.")

        # 3. Dictionary Coverage Check (If enabled)
        dict_path = self.config.get('dictionary_path')
        if dict_path:
            p_dict = Path(dict_path)
            if not p_dict.exists():
                raise FileNotFoundError(f"Dictionary configured but not found at: {dict_path}")
                
            logging.info(f"Validating ID coverage against dictionary: {dict_path}")
            dict_df = self._load_csv_safe(dict_path)
            
            # Dynamic Dictionary Columns
            d_cols = self.config.get('dictionary_cols', {})
            id_col = d_cols.get('id', 'id')
            
            if id_col not in dict_df.columns:
                 raise ValueError(f"Dictionary file missing configured ID column: '{id_col}'. Found: {list(dict_df.columns)}")

            valid_ids = set(dict_df[id_col].astype(str).str.strip())
            train_input_ids = set()
            
            # Parse training inputs to check coverage
            raw_inputs = df_train[mol_col].astype(str)
            for val in raw_inputs:
                tokens = val.replace('"', '').replace("'", "").split(';')
                for t in tokens:
                    if t.strip():
                        train_input_ids.add(t.strip())
            
            total_unique_inputs = len(train_input_ids)
            mapped_inputs = len(train_input_ids.intersection(valid_ids))
            
            if total_unique_inputs == 0:
                raise ValueError(f"No valid input tokens found in training column '{mol_col}' after parsing.")

            coverage_pct = (mapped_inputs / total_unique_inputs) * 100
            logging.info(f"Dictionary Coverage (Train): {coverage_pct:.2f}% ({mapped_inputs}/{total_unique_inputs} unique IDs mapped).")
            
            # Threshold Checks
            if coverage_pct < 1.0:
                raise ValueError(f"CRITICAL: Less than 1% of training IDs found in dictionary ({coverage_pct:.2f}%). Check ID format or dictionary file.")
            elif coverage_pct < 90.0:
                logging.warning(f"Warning: Dictionary coverage is low ({coverage_pct:.2f}%). Many molecules will be ignored or zeroed out.")
        
        logging.info("--- Data Validation Passed ---")

    def prepare_data(self, train_path: str, test_path: str, batch_size: int):
        """
        Main entry point for loading data.
        Performs merging, parsing, label detection, and DataLoader construction.
        """
        logging.info("Loading and Analyzing Data...")
        df_train = self._load_csv_safe(train_path)
        df_test = self._load_csv_safe(test_path)
        
        mol_col = self.config.get('mol_col', 'Molecular_Composition')
        label_col = self.config.get('label_col', 'Label')
        
        t_type = self.task_config.get('type')
        sub_type = self.task_config.get('sub_type')

        # 1. Optional Merge for Multi-Label
        if t_type == 'multilabel':
            df_train = self._merge_duplicate_inputs(df_train, mol_col, label_col)
            df_test = self._merge_duplicate_inputs(df_test, mol_col, label_col)

        # 2. Auto-Detect Classes / Labels
        if t_type == 'multilabel':
            # Collect all unique tags for binary classification
            all_tags = set()
            sep = ';' 
            for val in df_train[label_col]:
                parts = str(val).split(sep)
                for p in parts:
                    if p.strip():
                        all_tags.add(p.strip())
            
            sorted_tags = sorted(list(all_tags))
            self.label_map = {tag: i for i, tag in enumerate(sorted_tags)}
            self.num_classes = len(sorted_tags)
            logging.info(f"Auto-detected {self.num_classes} unique tags for Multi-Label (Binary Mode).")
            logging.info(f"  > First 5 tags: {sorted_tags[:5]}")

        elif t_type == 'classification' and sub_type == 'multiclass':
            # Collect mutually exclusive classes
            unique_vals = sorted(df_train[label_col].astype(str).unique())
            self.label_map = {val: i for i, val in enumerate(unique_vals)}
            self.num_classes = len(unique_vals)
            logging.info(f"Auto-detected {self.num_classes} classes for Multi-Class.")
            logging.info(f"  > Classes: {unique_vals}")
            
        else:
            # Standard Regression or Binary Classification
            self.num_classes = 1

        # 3. Parse Molecules (and apply Dictionary lookup if configured)
        df_train['smiles_list'] = self._parse_mol_column(df_train, mol_col)
        df_test['smiles_list'] = self._parse_mol_column(df_test, mol_col)

        # 4. Discover Task Map (For Multi-Task Curriculum)
        task_map = {}
        if t_type == 'multitask':
            lengths = df_train['smiles_list'].apply(len).unique()
            task_map = {f"{l}_mol": int(l) for l in sorted(lengths)}
            logging.info(f"Multi-Task Structure Detected: {task_map}")
        
        # 5. Build Datasets
        train_ds = MoleculeDataset(df_train, self.feature_factory, label_col, self.task_config, self.label_map, self.num_classes)
        test_ds = MoleculeDataset(df_test, self.feature_factory, label_col, self.task_config, self.label_map, self.num_classes)

        # 6. Build Loaders
        # Use LengthGroupedBatchSampler for Multi-task to ensure correct routing
        if t_type == 'multitask':
            logging.info("Enabling LengthGroupedBatchSampler for Structure-Based Multi-Tasking.")
            loaders = {
                'train': DataLoader(train_ds, batch_sampler=LengthGroupedBatchSampler(train_ds, batch_size, shuffle=True), collate_fn=self.collate_fn),
                'test': DataLoader(test_ds, batch_sampler=LengthGroupedBatchSampler(test_ds, batch_size, shuffle=False), collate_fn=self.collate_fn)
            }
        else:
            loaders = {
                'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn),
                'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
            }

        return loaders, {'features': self.feature_factory.get_expected_dim(), 'output_dim': self.num_classes}, task_map

    def _parse_mol_column(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        Parses molecule string (semicolon separated).
        Handles Dictionary lookup (ID -> SMILES) if a dictionary is loaded.
        """
        dict_path = self.config.get('dictionary_path')
        mapping = {}

        # Load Dictionary if exists
        if dict_path and Path(dict_path).exists():
            try:
                dict_df = self._load_csv_safe(dict_path)
                d_cols = self.config.get('dictionary_cols', {})
                id_col = d_cols.get('id')
                smi_col = d_cols.get('smiles')
                
                if id_col and smi_col:
                    mapping = dict(zip(dict_df[id_col].astype(str), dict_df[smi_col]))
                    logging.info(f"Loaded Dictionary: {len(mapping)} entries mapped.")
                else:
                    logging.warning("Dictionary columns (id/smiles) not specified in config. Skipping lookup.")
            except Exception as e:
                logging.warning(f"Failed to load dictionary: {e}")

        def parse_row(x):
            if pd.isna(x): return []
            s = str(x).strip()
            
            # Robust CSV parsing for quoted strings like "A;B"
            tokens = []
            if '"' in s or "'" in s:
                try:
                    reader = csv.reader([s], delimiter=';', skipinitialspace=True)
                    tokens = [i.strip() for i in next(reader) if i.strip()]
                except Exception: 
                    tokens = [i.strip() for i in s.split(";") if i.strip()]
            else:
                tokens = [i.strip() for i in s.split(";") if i.strip()]

            # Apply mapping if available, otherwise return token as-is
            if mapping:
                return [mapping.get(t, t) for t in tokens]
            return tokens

        return df[col].apply(parse_row)

    @staticmethod
    def collate_fn(batch):
        """
        Robust collator that handles feature padding and label stacking (scalar vs vector).
        """
        batch_size = len(batch)
        
        # 1. Feature Padding
        # Note: If LengthGroupedBatchSampler is used, max_m will equal min_m (no padding effectively).
        lengths = [len(item['features']) for item in batch]
        max_m = max(lengths) if lengths else 1
        feat_dim = batch[0]['features'][0].shape[0] if batch and batch[0]['features'] else 1
        
        features_tensor = torch.zeros((batch_size, max_m, feat_dim), dtype=torch.float32)
        mask_tensor = torch.zeros((batch_size, max_m), dtype=torch.bool)
        
        for i, item in enumerate(batch):
            flist = item['features']
            if flist:
                arr = np.stack(flist)
                m = arr.shape[0]
                features_tensor[i, :m, :] = torch.from_numpy(arr)
                mask_tensor[i, :m] = True
        
        # 2. Label Stacking
        first_lbl = batch[0]['label_processed']
        
        if isinstance(first_lbl, torch.Tensor):
            if first_lbl.ndim > 0:
                # Stack vectors (Multi-Label / Multi-Class)
                labels_tensor = torch.stack([b['label_processed'] for b in batch])
            else:
                # Stack scalars (Regression / Binary)
                labels_tensor = torch.stack([b['label_processed'] for b in batch])
        else:
            # Fallback for raw floats
            labels_tensor = torch.tensor([b['label_processed'] for b in batch], dtype=torch.float32)

        return {
            'features': features_tensor,
            'mask': mask_tensor,
            'labels': labels_tensor,
            'sample_ids': [b['id'] for b in batch]
        }


class MoleculeDataset(Dataset):
    """
    PyTorch Dataset wrapper. 
    Handles lazy feature generation and label encoding based on task type.
    """
    def __init__(self, df, factory, label_col, task_config, label_map, num_classes):
        self.data = df
        self.factory = factory
        self.label_col = label_col
        self.task_config = task_config
        self.label_map = label_map
        self.num_classes = num_classes

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row.get('smiles_list', [])
        
        # Lazy feature generation
        feats_dict = self.factory.get_features(smiles)
        feats = [feats_dict[s] for s in smiles if s in feats_dict]
        
        # Fallback for empty/failed features
        if not feats: 
            feats = [np.zeros((self.factory.get_expected_dim(),), dtype=np.float32)]

        # --- Label Processing ---
        raw_val = row.get(self.label_col)
        processed_label = 0.0
        
        t_type = self.task_config.get('type')
        sub_type = self.task_config.get('sub_type')

        if t_type == 'multilabel':
            # Create Multi-Hot Vector [0, 1, 0, 0, 1...] for binary encoding
            vec = torch.zeros(self.num_classes, dtype=torch.float32)
            parts = str(raw_val).split(';')
            for p in parts:
                p = p.strip()
                if p in self.label_map:
                    vec[self.label_map[p]] = 1.0
            processed_label = vec
            
        elif t_type == 'classification' and sub_type == 'multiclass':
            # Map string to Long Integer Index
            idx_val = self.label_map.get(str(raw_val), 0)
            processed_label = torch.tensor(idx_val, dtype=torch.long)
            
        else:
            # Standard Regression / Binary Classification
            try:
                processed_label = torch.tensor(float(raw_val), dtype=torch.float32)
            except:
                processed_label = torch.tensor(0.0, dtype=torch.float32)

        return {
            'features': feats, 
            'label_processed': processed_label, 
            'id': int(idx)
        }