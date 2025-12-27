import torch
import torch.nn as nn
from .attention_modules import MoleculeSimulator, CrossIntegration

class MolSetTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_sab_layers, num_seeds, 
                 dropout_params, head_hidden_layers, task_structure, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim

        # --- Shared Backbone ---
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.input_dropout = nn.Dropout(p=dropout_params.get('input', 0.0))
        
        self.encoder_layers = nn.ModuleList(
            [MoleculeSimulator(dim_in=model_dim, dim_out=model_dim, num_heads=nhead, 
                               dropout=dropout_params.get('transformer', 0.0), ln=True) 
             for _ in range(num_sab_layers)]
        )
        
        self.pooling = CrossIntegration(dim=model_dim, num_heads=nhead, num_seeds=num_seeds, 
                                        dropout=dropout_params.get('transformer', 0.0), ln=True)

        # --- Dynamic Heads ---
        self.heads = nn.ModuleDict()
        
        # Case A: Structure-Based Multi-Task (Separate heads per length)
        if task_structure:
            for task_key, _ in task_structure.items():
                head_input_dim = model_dim * num_seeds 
                self.heads[task_key] = self._create_head(
                    head_input_dim, head_hidden_layers, dropout_params.get('head', 0.0), output_dim
                )
        # Case B: Standard / Multi-Label (Single shared head)
        else:
            head_input_dim = model_dim * num_seeds
            self.heads['default'] = self._create_head(
                head_input_dim, head_hidden_layers, dropout_params.get('head', 0.0), output_dim
            )

    def _create_head(self, input_dim, hidden_layers, dr, out_dim):
        """Builds MLP head ending with dynamic output size."""
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dr))
            current_dim = hidden_dim
        
        # Final layer maps to output_dim (1 or N classes)
        layers.append(nn.Linear(current_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, x, task_key='default', return_attention=False):
        # Validation to ensure 'default' is used if no task_key provided
        if task_key not in self.heads:
            if 'default' in self.heads:
                task_key = 'default'
            else:
                raise ValueError(f"Task head '{task_key}' not found. Available: {list(self.heads.keys())}")

        x = self.input_projection(x)
        x = self.input_dropout(x)
        
        attention_scores = []
        for layer in self.encoder_layers:
            if return_attention:
                x, att = layer(x, return_attention=True)
                attention_scores.append(att)
            else:
                x = layer(x)

        pooled = self.pooling(x)
        pooled_flat = pooled.view(pooled.size(0), -1)
        
        prediction = self.heads[task_key](pooled_flat)
            
        if return_attention:
            return prediction, attention_scores
        return prediction