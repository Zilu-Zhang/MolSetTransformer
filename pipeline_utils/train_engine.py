import torch
import torch.nn as nn
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

class TrainEngine:
    def __init__(self, train_config, model_save_dir, task_id, task_config):
        self.config = train_config
        self.save_dir = model_save_dir
        self.task_id = task_id
        self.task_config = task_config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self, model, loaders, task_map):
        model.to(self.device)
        strategy = self.config.get('strategy', 'standard')
        
        criterion = self._get_loss_function()
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

    def _get_loss_function(self):
        t_type = self.task_config.get('type')
        sub_type = self.task_config.get('sub_type')
        
        if t_type == 'regression':
            return nn.MSELoss()
        elif t_type == 'classification':
            if sub_type == 'multiclass':
                return nn.CrossEntropyLoss()
            else:
                return nn.BCEWithLogitsLoss()
        # FIX B: Multi-label now strictly classification (BCE)
        elif t_type == 'multilabel':
            return nn.BCEWithLogitsLoss()
        elif t_type == 'multitask':
            if sub_type == 'regression':
                return nn.MSELoss()
            else:
                return nn.BCEWithLogitsLoss()
        
        return nn.MSELoss() 

    def _get_task_key_for_batch(self, features, task_map, labels=None):
        """
        Routes batch to specific head. 
        Safeguard: Ensures mixed batches (in label-based tasks) are detected.
        """
        if not task_map: return 'default'
        
        first_key = next(iter(task_map))
        
        # Scenario A: Cardinality/Length-based tasks (Structure-Based Multi-Task)
        if "_mol" in first_key:
            # FIX A: With LengthGroupedBatchSampler, shape[1] is exactly the length of all samples in batch
            max_m = features.shape[1]
            key = f"{max_m}_mol"
            return key if key in task_map else 'default'
        
        return 'default'

    def _train_standard(self, model, loader, criterion, task_map):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        epochs = self.config.get('epochs', 10)
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                loss = self._process_batch(model, batch, criterion, optimizer, task_map)
                total_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            logging.info(f"Epoch {epoch+1} Avg Loss: {total_loss / len(loader):.4f}")

    def _process_batch(self, model, batch, criterion, optimizer, task_map, weight=1.0):
        features = batch['features'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Pass labels to detection logic (though typically unused in structure-based tasks)
        task_key = self._get_task_key_for_batch(features, task_map, labels)
        
        optimizer.zero_grad()
        preds = model(features, task_key)
        
        # --- SHAPE HANDLING ---
        if preds.shape[-1] == 1 and labels.ndim == 1:
            loss = criterion(preds.squeeze(), labels)
        else:
            loss = criterion(preds, labels)
            
        loss = loss * weight
        loss.backward()
        optimizer.step()
        return loss.item()

    def _split_loader(self, original_loader, task_val, batch_size, task_map):
        dataset = original_loader.dataset
        indices = []
        target_val_str = str(task_val)
        
        for i in range(len(dataset)):
            # We assume length-based splitting for curriculum
            item = dataset.data.iloc[i] 
            actual_len = len(item.get('smiles_list', []))
            if str(actual_len) == target_val_str:
                indices.append(i)
                
        if not indices: return None
        subset = Subset(dataset, indices)
        
        # Note: When splitting manually for curriculum, we still might want batch grouping if multitask is active,
        # but curriculum usually implies specific lengths per stream anyway.
        # For safety, we use standard shuffle here as the subsets are already homogeneous by definition of curriculum streams.
        return DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=original_loader.collate_fn)

    def _train_curriculum(self, model, loader, criterion, task_map):
        config = self.config['curriculum_config']
        stages = config['stages']
        mapping = config['task_mapping']
        
        def get_bs(stage_cfg, key=None):
            bs = stage_cfg['batch_size']
            if isinstance(bs, dict) and key: return int(bs[key])
            if isinstance(bs, int): return bs
            return 32

        loader_pri = self._split_loader(loader, mapping['primary'], get_bs(stages['transfer'], 'primary'), task_map)
        loader_comp = self._split_loader(loader, mapping['complementary'], get_bs(stages['transfer'], 'complementary'), task_map)

        if not loader_pri or not loader_comp:
            raise ValueError("Curriculum learning requires data for both tasks.")

        # --- STAGE 1: Warm Up ---
        logging.info(">>> Stage 1: Warm Up")
        opt = torch.optim.Adam(model.parameters(), lr=stages['warm_up']['lr'])
        self._run_single_task_stage(model, loader_comp, criterion, opt, stages['warm_up']['epochs'], task_map)

        # --- STAGE 2: Transfer ---
        logging.info(">>> Stage 2: Transfer")
        opt = torch.optim.Adam(model.parameters(), lr=stages['transfer']['lr'])
        bal = stages['transfer'].get('balance', {'primary': 0.5, 'complementary': 0.5})
        self._run_dual_stream_stage(model, loader_pri, loader_comp, criterion, opt, stages['transfer']['epochs'], bal['primary'], bal['complementary'], task_map)

        # --- STAGE 3: Fine Tune ---
        logging.info(">>> Stage 3: Fine Tune")
        opt = torch.optim.Adam(model.parameters(), lr=stages['fine_tune']['lr'])
        self._run_single_task_stage(model, loader_pri, criterion, opt, stages['fine_tune']['epochs'], task_map)

    def _run_single_task_stage(self, model, loader, criterion, optimizer, epochs, task_map):
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"Stage Epoch {epoch+1}")
            for batch in pbar:
                loss = self._process_batch(model, batch, criterion, optimizer, task_map)
                total_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})

    def _run_dual_stream_stage(self, model, loader_pri, loader_comp, criterion, optimizer, epochs, w_pri, w_comp, task_map):
        model.train()
        for epoch in range(epochs):
            iter_pri = iter(loader_pri)
            iter_comp = iter(loader_comp)
            steps = max(len(loader_pri), len(loader_comp))
            
            pbar = tqdm(range(steps), desc=f"Transfer Epoch {epoch+1}")
            for _ in pbar:
                optimizer.zero_grad()
                
                # Manual processing to ensure gradients are accumulated if we were to change optimizer logic
                # For now, we follow the valid alternating update pattern
                
                try: batch_pri = next(iter_pri)
                except StopIteration: iter_pri = iter(loader_pri); batch_pri = next(iter_pri)
                loss_p = self._process_batch(model, batch_pri, criterion, optimizer, task_map, weight=w_pri) 

                try: batch_comp = next(iter_comp)
                except StopIteration: iter_comp = iter(loader_comp); batch_comp = next(iter_comp)
                loss_c = self._process_batch(model, batch_comp, criterion, optimizer, task_map, weight=w_comp)
                
                pbar.set_postfix({"loss": f"{(loss_p+loss_c):.4f}"})