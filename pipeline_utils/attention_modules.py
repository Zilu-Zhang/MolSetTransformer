import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionBlock(nn.Module):
    """
    Standard Multihead Attention Block (MAB).
    
    This block computes Scaled Dot-Product Attention between a Query (Q) set 
    and Key/Value (K/V) sets. It serves as the fundamental building block 
    for the Set Transformer architecture.
    """
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False, dropout: float = 0.0):
        super(AttentionBlock, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        
        # Calculate dimension per head.
        # Note: dim_V must be divisible by num_heads.
        self.dim_head = self.dim_V // self.num_heads
        
        self.attention_dropout = nn.Dropout(p=dropout)
        self.residual_dropout = nn.Dropout(p=dropout)

        # Linear Projections
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        
        # Layer Normalization
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None
        
        # Output Projection
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, return_attention: bool = False):
        """
        Args:
            Q: Query tensor of shape (Batch, Seq_Len_Q, Dim_Q)
            K: Key/Value tensor of shape (Batch, Seq_Len_K, Dim_K)
            return_attention: If True, returns the attention weights.
        """
        # 1. Linear Projections
        Q_proj = self.fc_q(Q)
        K_proj, V_proj = self.fc_k(K), self.fc_v(K)

        # 2. Split into heads and concatenate along batch dimension for parallel processing
        # Result Shape: (Batch * Num_Heads, Seq_Len, Dim_Head)
        Q_ = torch.cat(Q_proj.split(self.dim_head, 2), 0)
        K_ = torch.cat(K_proj.split(self.dim_head, 2), 0)
        V_ = torch.cat(V_proj.split(self.dim_head, 2), 0)

        # 3. Scaled Dot-Product Attention
        # Scaling by sqrt(dim_head) prevents gradients from vanishing in softmax
        scale_factor = math.sqrt(self.dim_head)
        score = Q_.bmm(K_.transpose(1, 2)) / scale_factor
        
        A = torch.softmax(score, 2)
        A_do = self.attention_dropout(A)

        # 4. Aggregate Value and Concatenate Heads
        # Reshape back to (Batch, Seq_Len, Dim_V)
        O = torch.cat((Q_ + A_do.bmm(V_)).split(Q.size(0), 0), 2)
        
        # 5. Residual Connection + Layer Norm (Block 1)
        O = O if self.ln0 is None else self.ln0(O)
        
        # 6. Feed-forward Network + Residual (Block 2)
        O = O + self.residual_dropout(F.relu(self.fc_o(O)))
        O = O if self.ln1 is None else self.ln1(O)

        if return_attention:
            # Reshape A to (Batch, Heads, Q_len, K_len) and average across heads
            A_reshaped = A.view(-1, self.num_heads, A.size(1), A.size(2))
            A_mean = A_reshaped.mean(dim=1)
            return O, A_mean
            
        return O

class MoleculeSimulator(nn.Module):
    """
    Set Attention Block (SAB).
    
    Performs self-attention on a set of elements (e.g., atoms or molecules).
    Input X serves as both Query, Key, and Value (X -> X).
    """
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False, dropout: float = 0.0):
        super(MoleculeSimulator, self).__init__()
        self.mab = AttentionBlock(dim_in, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X: torch.Tensor, return_attention: bool = False):
        return self.mab(X, X, return_attention=return_attention)

class CrossIntegration(nn.Module):
    """
    Pooling Multihead Attention (PMA).
    
    Aggregates information from a variable-sized set (X) into a fixed-size set of 
    latent vectors (Seeds) via cross-attention (Seeds -> X).
    """
    def __init__(self, dim: int, num_heads: int, num_seeds: int, ln: bool = False, dropout: float = 0.0):
        super(CrossIntegration, self).__init__()
        # Learnable Seed Vectors
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        
        self.mab = AttentionBlock(dim, dim, dim, num_heads, ln=ln, dropout=dropout)

    def forward(self, X: torch.Tensor):
        # Expand seeds to match batch size: (Batch, Num_Seeds, Dim)
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)