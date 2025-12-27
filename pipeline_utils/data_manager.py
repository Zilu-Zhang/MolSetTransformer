#!/usr/bin/env python3
"""
DataManager Module

Handles data ingestion (CSV), multi-label merging, schema validation, 
auto-detection of classification tasks, and the creation of PyTorch DataLoaders.
"""

import csv
import logging
from pathlib import Path
from typing import Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

try:
    from rdkit import Chem
except ImportError:
    Chem = None

from .feature_factory import FeatureFactory


class LengthGroupedBatchSampler(Sampler):
    """
    Sampler that yields batches where all samples have the same number of molecules.
    Essential for structure-based multi-task learning.
    """
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.groups = {}
        for idx in range(len(data_source)):
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
        
        self.label_map = None 
        self.num_classes = 1

    def _load_csv_safe(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
            if df.empty: raise ValueError(f"CSV file is empty: {path}")
            return df
        except Exception as e:
            logging.error(f"Failed to load CSV at {path}: {e}")
            raise

    def _merge_duplicate_inputs(self, df: pd.DataFrame, mol_col: str, label_col: str, sep: str = ';') -> pd.DataFrame:
        logging.info("Scanning for duplicate inputs to merge (Multi-Label Strategy)...")
        def merge_labels(series):
            unique_labels = set()
            for item in series:
                parts = str(item).split(sep)
                for p in parts:
                    if p.strip(): unique_labels.add(p.strip())
            return sep.join(sorted(list(unique_labels)))

        original_count = len(df)
        df_merged = df.groupby(mol_col)[label_col].apply(merge_labels).reset_index()
        if len(df_merged) < original_count:
            logging.info(f"  > Merged {original_count} rows into {len(df_merged)} unique input samples.")
        return df_merged

    def _analyze_smiles_stats(self, name: str, smiles_lists: pd.Series):
        """
        Calculates and logs detailed statistics about SMILES validity.
        """
        all_smiles = []
        for s_list in smiles_lists:
            all_smiles.extend(s_list)
        
        unique_smiles = set(all_smiles)
        total_unique = len(unique_smiles)
        
        # 1. Validity Check
        valid_count = 0
        if Chem:
            for s in unique_smiles:
                if Chem.MolFromSmiles(s):
                    valid_count += 1
        else:
            valid_count = total_unique

        valid_pct = (valid_count / total_unique * 100) if total_unique > 0 else 0
        
        stats_msg = (
            f"\n--- {name} Set Statistics ---\n"
            f"  > Total Unique SMILES: {total_unique}\n"
            f"  > Valid SMILES (RDKit): {valid_count} ({valid_pct:.2f}%)\n"
        )
        if not Chem:
            stats_msg += "  > [WARNING] RDKit not installed; skipping validity check.\n"
        
        logging.info(stats_msg)

    def prepare_data(self, train_path: str, test_path: str, batch_size: int):
        """
        Main entry point for loading data.
        Performs merging, parsing, label detection, and DataLoader construction.
        """
        logging.info("Loading Data...")
        df_train = self._load_csv_safe(train_path)
        df_test = self._load_csv_safe(test_path)
        
        mol_col = self.config.get('mol_col', 'Molecular_Composition')
        label_col = self.config.get('label_col', 'Label')
        
        t_type = self.task_config.get('type')
        
        # 1. Multi-Label Merge
        if t_type == 'multilabel':
            df_train = self._merge_duplicate_inputs(df_train, mol_col, label_col)
            df_test = self._merge_duplicate_inputs(df_test, mol_col, label_col)

        # 2. Parse Molecules & Load Dictionary
        dict_path = self.config.get('dictionary_path')
        dict_mapping = {}
        if dict_path and Path(dict_path).exists():
            try:
                d_df = self._load_csv_safe(dict_path)
                d_cols = self.config.get('dictionary_cols', {})
                if d_cols.get('id') and d_cols.get('smiles'):
                    dict_mapping = dict(zip(d_df[d_cols['id']].astype(str), d_df[d_cols['smiles']]))
                    logging.info(f"Loaded Dictionary: {len(dict_mapping)} entries.")
            except Exception as e:
                logging.warning(f"Dictionary load failed: {e}")

        logging.info("Parsing Molecular Composition...")
        df_train['smiles_list'] = self._parse_mol_column(df_train, mol_col, dict_mapping)
        df_test['smiles_list'] = self._parse_mol_column(df_test, mol_col, dict_mapping)

        # 3. Report Statistics
        self._analyze_smiles_stats("Train", df_train['smiles_list'])
        self._analyze_smiles_stats("Test", df_test['smiles_list'])

        # 4. Eager Feature Generation
        logging.info("--- Starting Bulk Feature Generation ---")
        all_unique_smiles = set()
        for sl in df_train['smiles_list']: all_unique_smiles.update(sl)
        for sl in df_test['smiles_list']: all_unique_smiles.update(sl)
        
        feature_cache = self.feature_factory.get_features(list(all_unique_smiles))
        logging.info("--- Feature Generation Complete ---")

        # 5. Label Detection
        if t_type == 'multilabel':
            all_tags = set()
            for val in df_train[label_col]:
                for p in str(val).split(';'):
                    if p.strip(): all_tags.add(p.strip())
            sorted_tags = sorted(list(all_tags))
            self.label_map = {tag: i for i, tag in enumerate(sorted_tags)}
            self.num_classes = len(sorted_tags)
        elif t_type == 'classification' and self.task_config.get('sub_type') == 'multiclass':
            unique_vals = sorted(df_train[label_col].astype(str).unique())
            self.label_map = {val: i for i, val in enumerate(unique_vals)}
            self.num_classes = len(unique_vals)
        else:
            self.num_classes = 1

        # 6. Task Map & Dataset
        task_map = {}
        if t_type == 'multitask':
            lengths = df_train['smiles_list'].apply(len).unique()
            task_map = {f"{l}_mol": int(l) for l in sorted(lengths)}

        train_ds = MoleculeDataset(df_train, feature_cache, self.feature_factory.get_expected_dim(), label_col, self.task_config, self.label_map, self.num_classes)
        test_ds = MoleculeDataset(df_test, feature_cache, self.feature_factory.get_expected_dim(), label_col, self.task_config, self.label_map, self.num_classes)

        # 7. Build Loaders
        if t_type == 'multitask':
            loaders = {
                'train': DataLoader(train_ds, batch_sampler=LengthGroupedBatchSampler(train_ds, batch_size, True), collate_fn=self.collate_fn),
                'test': DataLoader(test_ds, batch_sampler=LengthGroupedBatchSampler(test_ds, batch_size, False), collate_fn=self.collate_fn)
            }
        else:
            loaders = {
                'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn),
                'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
            }

        return loaders, {'features': self.feature_factory.get_expected_dim(), 'output_dim': self.num_classes}, task_map

    def _parse_mol_column(self, df: pd.DataFrame, col: str, mapping: dict) -> pd.Series:
        def parse_row(x):
            if pd.isna(x): return []
            s = str(x).strip()
            tokens = [t.strip() for t in s.split(";") if t.strip()]
            return [mapping.get(t, t) for t in tokens]
        return df[col].apply(parse_row)

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)
        lengths = [len(item['features']) for item in batch]
        max_m = max(lengths) if lengths else 1
        feat_dim = batch[0]['features'][0].shape[0] if batch and batch[0]['features'] else 1
        
        features_tensor = torch.zeros((batch_size, max_m, feat_dim), dtype=torch.float32)
        mask_tensor = torch.zeros((batch_size, max_m), dtype=torch.bool)
        
        for i, item in enumerate(batch):
            flist = item['features']
            if flist:
                arr = np.stack(flist)
                features_tensor[i, :arr.shape[0], :] = torch.from_numpy(arr)
                mask_tensor[i, :arr.shape[0]] = True
        
        first_lbl = batch[0]['label_processed']
        if isinstance(first_lbl, torch.Tensor) and first_lbl.ndim > 0:
            labels_tensor = torch.stack([b['label_processed'] for b in batch])
        else:
            labels_tensor = torch.tensor([b['label_processed'] for b in batch], dtype=torch.float32)

        return {
            'features': features_tensor,
            'mask': mask_tensor,
            'labels': labels_tensor,
            'sample_ids': [b['id'] for b in batch]
        }


class MoleculeDataset(Dataset):
    def __init__(self, df, feature_cache, expected_dim, label_col, task_config, label_map, num_classes):
        self.data = df
        self.feature_cache = feature_cache
        self.expected_dim = expected_dim
        self.label_col = label_col
        self.task_config = task_config
        self.label_map = label_map
        self.num_classes = num_classes

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row.get('smiles_list', [])
        
        # Fast Dictionary Lookup
        feats = [self.feature_cache[s] for s in smiles if s in self.feature_cache]
        if not feats: feats = [np.zeros((self.expected_dim,), dtype=np.float32)]

        raw_val = row.get(self.label_col)
        processed_label = 0.0
        t_type = self.task_config.get('type')
        
        if t_type == 'multilabel':
            vec = torch.zeros(self.num_classes, dtype=torch.float32)
            for p in str(raw_val).split(';'):
                if p.strip() in self.label_map: vec[self.label_map[p.strip()]] = 1.0
            processed_label = vec
        elif t_type == 'classification' and self.task_config.get('sub_type') == 'multiclass':
            processed_label = torch.tensor(self.label_map.get(str(raw_val), 0), dtype=torch.long)
        else:
            try: processed_label = torch.tensor(float(raw_val), dtype=torch.float32)
            except: processed_label = torch.tensor(0.0, dtype=torch.float32)

        return {'features': feats, 'label_processed': processed_label, 'id': int(idx)}