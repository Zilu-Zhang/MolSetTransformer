import logging
import torch
import torch.nn as nn
from typing import Dict
from .mol_set_transformer import MolSetTransformer

class ModelBuilder:
    """
    Factory class for instantiating MolSetTransformer models from a configuration dictionary.
    """
    def __init__(self, arch_config: Dict):
        self.config = arch_config

    def build(self, input_dim: int, task_structure: Dict, output_dim: int = 1) -> MolSetTransformer:
        """
        Builds the Set Transformer based on configuration.
        
        Args:
            input_dim: Integer feature dimension of a single element (e.g., molecule).
            task_structure: Dict mapping task names to cardinality (e.g., {'2_mol': 2}).
                            If empty/None, implies a single shared head ('default').
            output_dim: The size of the final prediction layer. 
                        - 1 for Regression or Binary Classification.
                        - N for Multi-Label or Multi-Class.
        
        Returns:
            An instance of MolSetTransformer.
        """
        logging.info(f"Building Model | Input Dim: {input_dim} | Output Dim: {output_dim}")
        
        # Retrieve config values with defaults
        model_dim = int(self.config.get('model_dim', 256))
        nhead = int(self.config.get('nhead', 8))
        
        # Architecture Validation
        if model_dim % nhead != 0:
            raise ValueError(
                f"Configuration Error: Model Dimension ({model_dim}) must be divisible by Heads ({nhead})."
            )

        # Instantiate Model
        model = MolSetTransformer(
            input_dim=int(input_dim),
            model_dim=model_dim,
            nhead=nhead,
            num_sab_layers=int(self.config.get('num_attention_blocks', 4)),
            num_seeds=int(self.config.get('num_integrators', 6)),
            dropout_params=self.config.get('dropouts', {}),
            head_hidden_layers=self.config.get('head_hidden_layers', []),
            task_structure=task_structure,
            output_dim=output_dim 
        )
        
        return model