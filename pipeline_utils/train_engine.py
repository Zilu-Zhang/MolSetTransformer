#!/usr/bin/env python3
"""
TrainEngine Module
==================

Manages the model training lifecycle, supporting:
1. Standard Training (Single-task & Uniform Multi-task)
2. Curriculum Learning (Staged training with primary/complementary tasks)
3. Dynamic Loss Weighting for Imbalanced Data

Handles device management, optimizer configuration, and check-pointing.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

class TrainEngine:
    """
    Orchestrates the training loop, loss calculation, and optimization.
    """
    def __init__(self, train_config, model_save_dir, task_id, task_config):
        self.config = train_config
        self.save_dir = model_save_dir
        self.task_id = task_id
        self.task_config = task_config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self, model, loaders, task_map):
        """
        Main entry point for training. Selects the strategy based on configuration.
        """
        model.to(self.device)
        strategy = self.config.get('strategy', 'standard')
        
        # Calculate loss function (injecting train loader for dynamic weight calculation)
        criterion = self._get_loss_function(loaders.get('train'))
        
        logging.info(f"Training Strategy: {strategy}")
        logging.info(f"Loss Function: {criterion.__class__.__name__}")

        if strategy == "standard" or strategy == "custom_weights":
            self._train_standard(model, loaders['train'], criterion, task_map)
        elif strategy == "curriculum_learning":
            self._train_curriculum(model, loaders['train'], criterion, task_map)
        
        final_path = self.save_dir / f"{self.task_id}_model_final.pth"
        torch.save(model.state_dict(), final_path)
        logging.info(f"Training Complete. Model saved to {final_path}")
        return final_path

    def _calculate_pos_weights(self, loader):
        """
        Scans the data loader to calculate class imbalance weights.
        
        Formula: weight_class_i = number_of_negatives_i / number_of_positives_i
        
        This ensures that the loss function penalizes missing a rare positive class 
        significantly more than missing a common negative class.
        """
        logging.info("  > Scanning dataset to calculate pos_weight for imbalance correction...")
        total_pos = None
        total_count = 0
        
        # Iterate purely to count, no gradient needed.
        # Calculations are performed on CPU to preserve GPU memory.
        with torch.no_grad():
            for batch in loader:
                # 'labels' comes from the collate_fn (Batch, Num_Classes)
                labels = batch['labels'] 
                
                if total_pos is None:
                    total_pos = torch.zeros(labels.shape[1])
                
                total_pos += labels.float().cpu().sum(dim=0)
                total_count += labels.size(0)
            
        if total_count == 0:
            logging.warning("  > Dataset empty. Skipping weight calculation.")
            return None

        # Calculate Negatives
        num_neg = total_count - total_pos
        
        # Calculate Weights: neg/pos (add epsilon to avoid divide by zero)
        # Example: 99 negs, 1 pos -> weight = 99.
        pos_weights = num_neg / (total_pos + 1e-6)
        
        logging.info(f"  > Class weights calculated. Min: {pos_weights.min():.2f}, Max: {pos_weights.max():.2f}")
        
        return pos_weights.to(self.device)

    def _get_loss_function(self, loader=None):
        """
        Determines the appropriate loss function based on task type.
        
        Args:
            loader: Optional DataLoader. Used to calculate dynamic weights for 
                    imbalanced multi-label datasets.
        """
        t_type = self.task_config.get('type')
        sub_type = self.task_config.get('sub_type')
        
        if t_type == 'regression':
            return nn.MSELoss()
            
        elif t_type == 'classification':
            if sub_type == 'multiclass':
                return nn.CrossEntropyLoss()
            else:
                return nn.BCEWithLogitsLoss()
                
        elif t_type == 'multilabel':
            # Calculate weights if a loader is provided
            weights = None
            if loader is not None:
                weights = self._calculate_pos_weights(loader)
            return nn.BCEWithLogitsLoss(pos_weight=weights)
            
        elif t_type == 'multitask':
            if sub_type == 'regression':
                return nn.MSELoss()
            else:
                return nn.BCEWithLogitsLoss()
        
        return nn.MSELoss() 

    def _get_task_key_for_batch(self, features, task_map, labels=None):
        """
        Routes the batch to a specific model head.
        For Structure-Based Multi-Task learning, selects head based on input length.
        """
        if not task_map: return 'default'
        
        first_key = next(iter(task_map))
        
        # Check for Structure-Based Multi-Task keys (denoted by "_mol")
        if "_mol" in first_key:
            # With LengthGroupedBatchSampler, dim 1 is the uniform length of the batch
            max_m = features.shape[1]
            key = f"{max_m}_mol"
            return key if key in task_map else 'default'
        
        return 'default'

    def _process_batch(self, model, batch, criterion, optimizer, task_map, weight=1.0):
        """
        Processes a single batch: Forward pass -> Loss -> Backward pass -> Step.
        """
        features = batch['features'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Determine active head
        task_key = self._get_task_key_for_batch(features, task_map, labels)
        
        optimizer.zero_grad()
        preds = model(features, task_key)
        
        # --- Shape Handling ---
        # Use squeeze(-1) to align dimensions while preserving the batch dimension (even if batch_size=1).
        if preds.shape[-1] == 1 and labels.ndim == 1:
            loss = criterion(preds.squeeze(-1), labels)
        else:
            loss = criterion(preds, labels)
            
        loss = loss * weight
        loss.backward()
        optimizer.step()
        return loss.item()

    def _train_standard(self, model, loader, criterion, task_map):
        """Standard training loop."""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        epochs = self.config.get('epochs', 10)
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                loss = self._process_batch(model, batch, criterion, optimizer, task_map)
                total_loss += loss
            
            avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
            logging.info(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    def _split_loader(self, original_loader, task_val, batch_size, task_map):
        """
        Creates a subset DataLoader for a specific task (e.g., specific molecule length).
        """
        dataset = original_loader.dataset
        indices = []
        target_val_str = str(task_val)
        
        for i in range(len(dataset)):
            # Assumes underlying data structure supports 'smiles_list'
            item = dataset.data.iloc[i] 
            actual_len = len(item.get('smiles_list', []))
            if str(actual_len) == target_val_str:
                indices.append(i)
                
        if not indices: return None
        subset = Subset(dataset, indices)
        
        return DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=original_loader.collate_fn
        )

    def _train_curriculum(self, model, loader, criterion, task_map):
        """
        Executes a 3-stage Curriculum Learning strategy.
        """
        config = self.config['curriculum_config']
        stages = config['stages']
        mapping = config['task_mapping']
        
        def get_bs(stage_cfg, key=None):
            bs = stage_cfg['batch_size']
            if isinstance(bs, dict) and key: return int(bs[key])
            if isinstance(bs, int): return bs
            return 32

        # Split data streams
        loader_pri = self._split_loader(loader, mapping['primary'], get_bs(stages['transfer'], 'primary'), task_map)
        loader_comp = self._split_loader(loader, mapping['complementary'], get_bs(stages['transfer'], 'complementary'), task_map)

        if not loader_pri or not loader_comp:
            raise ValueError("Curriculum learning requires data for both primary and complementary tasks.")

        # Stage 1: Warm Up
        logging.info(">>> Stage 1: Warm Up")
        opt = torch.optim.Adam(model.parameters(), lr=stages['warm_up']['lr'])
        self._run_single_task_stage(model, loader_comp, criterion, opt, stages['warm_up']['epochs'], task_map)

        # Stage 2: Transfer
        logging.info(">>> Stage 2: Transfer")
        opt = torch.optim.Adam(model.parameters(), lr=stages['transfer']['lr'])
        bal = stages['transfer'].get('balance', {'primary': 0.5, 'complementary': 0.5})
        self._run_dual_stream_stage(model, loader_pri, loader_comp, criterion, opt, stages['transfer']['epochs'], bal['primary'], bal['complementary'], task_map)

        # Stage 3: Fine Tune
        logging.info(">>> Stage 3: Fine Tune")
        opt = torch.optim.Adam(model.parameters(), lr=stages['fine_tune']['lr'])
        self._run_single_task_stage(model, loader_pri, criterion, opt, stages['fine_tune']['epochs'], task_map)

    def _run_single_task_stage(self, model, loader, criterion, optimizer, epochs, task_map):
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                loss = self._process_batch(model, batch, criterion, optimizer, task_map)
                total_loss += loss
            logging.info(f"Stage Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    def _run_dual_stream_stage(self, model, loader_pri, loader_comp, criterion, optimizer, epochs, w_pri, w_comp, task_map):
        """
        Alternates training between primary and complementary data streams.
        """
        model.train()
        for epoch in range(epochs):
            iter_pri = iter(loader_pri)
            iter_comp = iter(loader_comp)
            steps = max(len(loader_pri), len(loader_comp))
            total_loss = 0.0
            
            for _ in range(steps):
                optimizer.zero_grad()
                
                # Fetch batches, cycling if exhausted
                try: batch_pri = next(iter_pri)
                except StopIteration: iter_pri = iter(loader_pri); batch_pri = next(iter_pri)
                
                loss_p = self._process_batch(model, batch_pri, criterion, optimizer, task_map, weight=w_pri) 

                try: batch_comp = next(iter_comp)
                except StopIteration: iter_comp = iter(loader_comp); batch_comp = next(iter_comp)
                
                loss_c = self._process_batch(model, batch_comp, criterion, optimizer, task_map, weight=w_comp)
                
                total_loss += (loss_p + loss_c)
            
            logging.info(f"Transfer Epoch {epoch+1}/{epochs} - Combined Loss: {total_loss/steps:.4f}")
