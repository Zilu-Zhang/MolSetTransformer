#!/usr/bin/env python3
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

class InferenceEngine:
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
        
        # Inverse map for decoding (Index -> "Label Name")
        self.idx_to_label = {v: k for k, v in label_map.items()} if label_map else None

    def _load_checkpoint(self, model: torch.nn.Module, checkpoint_path: str):
        if not checkpoint_path: return model
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location=self.device)
        state = cp['state_dict'] if (isinstance(cp, dict) and 'state_dict' in cp) else cp
        try: model.load_state_dict(state)
        except RuntimeError:
            new_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(new_state)
        return model

    def _enable_dropout(self, model: torch.nn.Module):
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

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
                
                # Dynamic Head Routing
                # FIX A: LengthGroupedBatchSampler ensures this is accurate without padding artifacts
                max_m = feats.shape[1]
                struct_key = f"{max_m}_mol" 
                
                target_heads = []
                t_type = self.task_config.get('type')

                # --- STRICT LOGIC IMPLEMENTATION ---
                if t_type == 'multitask':
                    # Case A: Structure-Based Multi-Task
                    # STRICT: The input length MUST match a specific trained head.
                    if struct_key in available_heads:
                        target_heads.append(struct_key)
                    else:
                        # Stop immediately if the model doesn't know this physics
                        raise ValueError(
                            f"[Strict Inference] Input sample contains {max_m} molecules, "
                            f"but the model was only trained for these tasks: {available_heads}. "
                            "In Multi-Task mode, input length must match a trained task head."
                        )
                else:
                    # Case B: Standard / Multi-Label / Single Task
                    # FLEXIBLE: Length doesn't dictate the task; use the shared head.
                    if 'default' in available_heads:
                        target_heads.append('default')
                    elif available_heads:
                        # Fallback to the single existing head (common in single-task models)
                        target_heads.append(available_heads[0])
                    else:
                         raise RuntimeError("No model heads found. Model structure is invalid.")

                for head_key in target_heads:
                    if self.mc_iterations > 1: self._enable_dropout(model)
                    else: model.eval()

                    logits_stack = []
                    
                    for mc_i in range(self.mc_iterations):
                        should_save_attn = (self.export_attention and mc_i == 0 and len(attn_collected) < MAX_ATTN_SAMPLES)
                        out = model(feats, head_key, return_attention=should_save_attn)
                        
                        if isinstance(out, tuple):
                            pred, attn = out
                            if should_save_attn and attn:
                                current_batch_size = len(sample_ids)
                                for k in range(current_batch_size):
                                    if len(attn_collected) >= MAX_ATTN_SAMPLES: break
                                    sample_attn = [layer_att[k].cpu().numpy() for layer_att in attn] if isinstance(attn, list) else attn[k].cpu().numpy()
                                    attn_collected.append({ "sample_id": sample_ids[k], "attention": sample_attn })
                        else: pred = out
                        logits_stack.append(pred.cpu().numpy())

                    # --- AGGREGATION ---
                    # Shape: (MC_Iter, Batch_Size, Output_Dim)
                    logits_stack = np.stack(logits_stack, axis=0)
                    
                    logits_mean = np.mean(logits_stack, axis=0)
                    logits_std = np.std(logits_stack, axis=0)
                    
                    t_type = self.task_config.get('type')
                    sub_type = self.task_config.get('sub_type')

                    for i in range(logits_mean.shape[0]):
                        row = {"sample_id": sample_ids[i] if sample_ids else None}
                        
                        # --- OUTPUT LOGIC ---
                        
                        # 1. MULTI-LABEL (e.g. DDI)
                        # FIX B: Unified Logic (Binary Mode Only)
                        if t_type == 'multilabel':
                            probs = 1.0 / (1.0 + np.exp(-logits_mean[i]))
                            active_indices = np.where(probs > 0.5)[0]
                            decoded_labels = []
                            for idx in active_indices:
                                if self.idx_to_label and idx in self.idx_to_label:
                                    decoded_labels.append(str(self.idx_to_label[idx]))
                                else:
                                    decoded_labels.append(str(idx))
                            
                            row["Predicted_Labels"] = "; ".join(decoded_labels)
                            row["All_Probabilities"] = "; ".join([f"{p:.4f}" for p in probs])
                            if self.mc_iterations > 1:
                                row["Uncertainty_Avg"] = np.mean(logits_std[i])

                        # 2. MULTI-CLASS CLASSIFICATION
                        elif t_type == 'classification' and sub_type == 'multiclass':
                            # Softmax
                            raw_logits = logits_mean[i]
                            exp_logits = np.exp(raw_logits - np.max(raw_logits))
                            probs = exp_logits / exp_logits.sum()
                            
                            pred_idx = np.argmax(probs)
                            row["Predicted_Class_Index"] = pred_idx
                            if self.idx_to_label and pred_idx in self.idx_to_label:
                                row["Predicted_Class"] = self.idx_to_label[pred_idx]
                            else:
                                row["Predicted_Class"] = pred_idx
                                
                            row["Max_Probability"] = probs[pred_idx]
                            row["All_Probabilities"] = "; ".join([f"{p:.4f}" for p in probs])

                        # 3. STANDARD REGRESSION / BINARY CLASS
                        else:
                            val = float(logits_mean[i].item())
                            
                            if t_type == 'classification' or (sub_type == 'binary'):
                                prob = 1.0 / (1.0 + np.exp(-val))
                                row["Prediction_Probability"] = prob
                                row["Predicted_Class"] = 1 if prob > 0.5 else 0
                            else:
                                row["Prediction"] = val
                            
                            if self.mc_iterations > 1:
                                row["Uncertainty"] = float(logits_std[i].item())

                        rows.append(row)

        pred_df = pd.DataFrame(rows)
        
        # --- MERGE WITH ORIGINAL DATA ---
        if not pred_df.empty and 'sample_id' in pred_df.columns:
            pred_df.set_index('sample_id', inplace=True)
            
            if hasattr(test_loader.dataset, 'data'):
                original_df = test_loader.dataset.data.copy()
                final_df = original_df.join(pred_df, how='left')
                
                cols_to_drop = ['features', 'label_processed']
                for c in cols_to_drop:
                    if c in final_df.columns: final_df.drop(columns=[c], inplace=True)
            else:
                final_df = pred_df
        else:
            final_df = pred_df

        out_csv = self.predictions_dir / f"{task_id}_predictions.csv"
        final_df.to_csv(out_csv)
        logging.info(f"Saved merged predictions to {out_csv}")

        if attn_collected:
            attn_path = self.attn_dir / f"{task_id}_attention_top{MAX_ATTN_SAMPLES}.npz"
            save_dict = {}
            for i, item in enumerate(attn_collected):
                save_dict[f"id_{item['sample_id']}"] = item['attention']
            np.savez_compressed(attn_path, **save_dict)