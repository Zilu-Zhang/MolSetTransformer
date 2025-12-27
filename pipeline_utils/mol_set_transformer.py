import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .attention_modules import MoleculeSimulator, CrossIntegration

class MolSetTransformer(nn.Module):
    """
    Set Transformer Architecture for Molecular Property Prediction.
    
    Pipeline:
    1. Input Projection (Embedding)
    2. Encoder: Stack of Self-Attention Blocks (SAB) processing element interactions.
    3. Pooling: Cross-Attention (PMA) compressing variable sets into fixed latent seeds.
    4. Heads: Dynamic MLP heads for single-task, multi-task, or multi-label output.
    """
    def __init__(self, 
                 input_dim: int, 
                 model_dim: int, 
                 nhead: int, 
                 num_sab_layers: int, 
                 num_seeds: int, 
                 dropout_params: Dict[str, float], 
                 head_hidden_layers: List[int], 
                 task_structure: Optional[Dict], 
                 output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim

        # --- Shared Backbone ---
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.input_dropout = nn.Dropout(p=dropout_params.get('input', 0.0))
        
        # Encoder: Contextualize elements via self-attention
        self.encoder_layers = nn.ModuleList(
            [MoleculeSimulator(dim_in=model_dim, dim_out=model_dim, num_heads=nhead, 
                               dropout=dropout_params.get('transformer', 0.0), ln=True) 
             for _ in range(num_sab_layers)]
        )
        
        # Pooling: Aggregate elements into k seed vectors
        self.pooling = CrossIntegration(dim=model_dim, num_heads=nhead, num_seeds=num_seeds, 
                                        dropout=dropout_params.get('transformer', 0.0), ln=True)

        # --- Dynamic Prediction Heads ---
        self.heads = nn.ModuleDict()
        
        # Case A: Structure-Based Multi-Task 
        # (e.g., separate heads for 2-molecule inputs vs 3-molecule inputs)
        if task_structure:
            for task_key, _ in task_structure.items():
                head_input_dim = model_dim * num_seeds 
                self.heads[task_key] = self._create_head(
                    head_input_dim, head_hidden_layers, dropout_params.get('head', 0.0), output_dim
                )
        # Case B: Standard / Multi-Label / Single Task
        # (A single shared head for all inputs)
        else:
            head_input_dim = model_dim * num_seeds
            self.heads['default'] = self._create_head(
                head_input_dim, head_hidden_layers, dropout_params.get('head', 0.0), output_dim
            )

    def _create_head(self, input_dim: int, hidden_layers: List[int], dr: float, out_dim: int) -> nn.Sequential:
        """Constructs an MLP prediction head with LayerNorm and ReLU."""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dr))
            current_dim = hidden_dim
        
        # Final projection layer
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, task_key: str = 'default', return_attention: bool = False):
        """
        Args:
            x: Input tensor (Batch, Set_Size, Feat_Dim)
            task_key: Key to select the specific prediction head (used in structural multi-tasking).
            return_attention: If True, returns list of attention maps from encoder layers.
        """
        # Ensure valid head selection
        if task_key not in self.heads:
            if 'default' in self.heads:
                task_key = 'default'
            else:
                raise ValueError(f"Task head '{task_key}' not found. Available: {list(self.heads.keys())}")

        # 1. Project Input
        x = self.input_projection(x)
        x = self.input_dropout(x)
        
        # 2. Apply Encoder Layers (SABs)
        attention_scores = []
        for layer in self.encoder_layers:
            if return_attention:
                x, att = layer(x, return_attention=True)
                attention_scores.append(att)
            else:
                x = layer(x)

        # 3. Apply Pooling (PMA)
        pooled = self.pooling(x)
        
        # Flatten pooled seeds: (Batch, Num_Seeds, Dim) -> (Batch, Num_Seeds * Dim)
        pooled_flat = pooled.view(pooled.size(0), -1)
        
        # 4. Prediction Head
        prediction = self.heads[task_key](pooled_flat)
            
        if return_attention:
            return prediction, attention_scores
        return prediction