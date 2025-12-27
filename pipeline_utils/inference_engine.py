#!/usr/bin/env python3
"""
InferenceEngine Module
======================

Handles model inference, uncertainty estimation via MC Dropout, and 
attention score extraction.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class InferenceEngine:
    """
    Manages inference workflows, including:
    - Loading checkpoints
    - MC Dropout for uncertainty
    - Exporting predictions and attention maps
    - Merging results with original metadata
    """
    def __init__(self, model_config: dict, predictions_dir: Path, task_config: dict, label_map: dict = None):
        self.config = model_config or {}
        self.predictions_dir = Path(predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.attn_dir = self.predictions_dir / "attention"
        self.attn_dir.mkdir(parents=True, exist_ok=True)
        
        self.mc_iterations = int(self.config.get("mc_dropout_iter", 1))
        self.export_attention = self.config.get("attention_score_export", False)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.task_config = task_config
        self.idx_to_label = {v: k for k, v in label_map.items()} if label_map else None

    def _load_checkpoint(self, model: torch.nn.Module, checkpoint_path: str):
        if not checkpoint_path: return model
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        state = cp['state_dict'] if (isinstance(cp, dict) and 'state_dict' in cp) else cp
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Handle DataParallel state dicts if necessary
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state)
        return model

    def _enable_dropout(self, model: torch.nn.Module):
        """Forces Dropout layers to train mode for MC Uncertainty estimation."""
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout): m.train()

    def run(self, model: torch.nn.Module, best_model_path: str, test_loader, task_id: str):
        logging.info(f"Starting inference for task: {task_id}")
        
        if best_model_path and Path(best_model_path).exists():
            model = self._load_checkpoint(model, best_model_path)
        
        model.to(self.device)
        model.eval()
        
        rows = []
        attn_collected = []
        MAX_ATTN_SAMPLES = 50 

        available_heads = list(model.heads.keys()) if hasattr(model, 'heads') else []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                feats = batch['features'].to(self.device)
                sample_ids = batch['sample_ids']
                
                # Determine head based on input length
                max_m = feats.shape[1]
                struct_key = f"{max_m}_mol" 
                
                target_heads = []
                t_type = self.task_config.get('type')

                if t_type == 'multitask':
                    if struct_key in available_heads: 
                        target_heads.append(struct_key)
                    else: 
                        raise ValueError(f"Input len {max_m} not supported by heads {available_heads}")
                else:
                    target_heads.append(available_heads[0] if available_heads else 'default')

                for head_key in target_heads:
                    # MC Dropout Logic
                    if self.mc_iterations > 1: self._enable_dropout(model)
                    else: model.eval()

                    logits_stack = []
                    for mc_i in range(self.mc_iterations):
                        # Only save attention for the first MC pass to save space
                        should_save_attn = (self.export_attention and mc_i == 0 and len(attn_collected) < MAX_ATTN_SAMPLES)
                        
                        out = model(feats, head_key, return_attention=should_save_attn)
                        
                        if isinstance(out, tuple):
                            pred, attn = out
                            if should_save_attn and attn:
                                for k in range(len(sample_ids)):
                                    if len(attn_collected) >= MAX_ATTN_SAMPLES: break
                                    # Handle tuple vs list attention outputs
                                    sample_attn = [layer_att[k].cpu().numpy() for layer_att in attn] if isinstance(attn, list) else attn[k].cpu().numpy()
                                    attn_collected.append({ "sample_id": sample_ids[k], "attention": sample_attn })
                        else: 
                            pred = out
                        
                        logits_stack.append(pred.cpu().numpy())

                    # Aggregate Predictions
                    logits_mean = np.mean(np.stack(logits_stack, axis=0), axis=0)
                    logits_std = np.std(np.stack(logits_stack, axis=0), axis=0)

                    # Post-process logits into readable format
                    for i in range(logits_mean.shape[0]):
                        row = {"sample_id": sample_ids[i] if sample_ids else None}
                        
                        if t_type == 'multilabel':
                            probs = 1.0 / (1.0 + np.exp(-logits_mean[i]))
                            active = np.where(probs > 0.5)[0]
                            labels = [str(self.idx_to_label[x]) if self.idx_to_label else str(x) for x in active]
                            row["Predicted_Labels"] = "; ".join(labels)
                            row["All_Probabilities"] = "; ".join([f"{p:.4f}" for p in probs])
                            if self.mc_iterations > 1: 
                                row["Uncertainty_Avg"] = np.mean(logits_std[i])

                        elif t_type == 'classification' and self.task_config.get('sub_type') == 'multiclass':
                            probs = np.exp(logits_mean[i] - np.max(logits_mean[i]))
                            probs /= probs.sum()
                            idx = np.argmax(probs)
                            row["Predicted_Class"] = self.idx_to_label[idx] if self.idx_to_label else idx
                            row["Max_Probability"] = probs[idx]

                        else:
                            val = float(logits_mean[i].item())
                            if t_type == 'classification' or (self.task_config.get('sub_type') == 'binary'):
                                prob = 1.0 / (1.0 + np.exp(-val))
                                row["Prediction_Probability"] = prob
                                row["Predicted_Class"] = 1 if prob > 0.5 else 0
                            else:
                                row["Prediction"] = val
                            
                            if self.mc_iterations > 1: 
                                row["Uncertainty"] = float(logits_std[i].item())

                        rows.append(row)

        pred_df = pd.DataFrame(rows)
        
        # --- Merge and Clean ---
        if not pred_df.empty and 'sample_id' in pred_df.columns:
            pred_df.set_index('sample_id', inplace=True)
            if hasattr(test_loader.dataset, 'data'):
                original_df = test_loader.dataset.data.copy()
                final_df = original_df.join(pred_df, how='left')
                
                # Clean internal pipeline columns while keeping metadata
                cols_to_drop = ['features', 'label_processed', 'smiles_list'] 
                final_df.drop(columns=[c for c in cols_to_drop if c in final_df.columns], inplace=True)
            else:
                final_df = pred_df
        else:
            final_df = pred_df

        out_csv = self.predictions_dir / f"{task_id}_predictions.csv"
        final_df.to_csv(out_csv)
        logging.info(f"Saved merged predictions to {out_csv}")

        # --- Export Attention ---
        if attn_collected:
            attn_path = self.attn_dir / f"{task_id}_attention.npz"
            save_dict = {f"id_{item['sample_id']}": item['attention'] for item in attn_collected}
            np.savez_compressed(attn_path, **save_dict)