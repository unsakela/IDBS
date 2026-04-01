"""
Bidirectional Cross-Modal Attention Module for IDBS System

This module implements a bidirectional cross-attention mechanism for fusing
information from multiple sensor modalities in the Intelligent Driver Behavior
Sensing system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for cross-modal interaction."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention."""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V), attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        
        return output, attention_weights


class CrossModalAttention(nn.Module):
    """
    Bidirectional Cross-Modal Attention Module
    
    This module implements bidirectional cross-attention between two modalities,
    allowing each modality to attend to the other and produce a fused representation.
    """
    
    def __init__(self, d_model, num_heads=8, dropout=0.1, fusion_strategy='concat'):
        super(CrossModalAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.fusion_strategy = fusion_strategy
        
        # Cross-attention layers
        self.attention_mod1_to_mod2 = MultiHeadAttention(d_model, num_heads, dropout)
        self.attention_mod2_to_mod1 = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward networks
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Fusion layer
        if fusion_strategy == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
        elif fusion_strategy == 'add':
            self.fusion_layer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model)
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def forward(self, modality1, modality2, mask=None):
        """
        Forward pass of bidirectional cross-attention
        
        Args:
            modality1: Tensor of shape (batch_size, seq_len1, d_model)
            modality2: Tensor of shape (batch_size, seq_len2, d_model)
            mask: Optional attention mask
            
        Returns:
            fused_output: Fused representation (batch_size, seq_len, d_model)
            attention_weights: Dictionary of attention weights for interpretability
        """
        # Cross-attention: modality1 attends to modality2
        mod1_attended, attn_weights_1_to_2 = self.attention_mod1_to_mod2(
            modality1, modality2, modality2, mask
        )
        mod1_attended = self.layer_norm1(mod1_attended + modality1)
        mod1_attended = self.ffn1(mod1_attended)
        mod1_attended = self.layer_norm1(mod1_attended + mod1_attended)
        
        # Cross-attention: modality2 attends to modality1
        mod2_attended, attn_weights_2_to_1 = self.attention_mod2_to_mod1(
            modality2, modality1, modality1, mask
        )
        mod2_attended = self.layer_norm2(mod2_attended + modality2)
        mod2_attended = self.ffn2(mod2_attended)
        mod2_attended = self.layer_norm2(mod2_attended + mod2_attended)
        
        # Fusion strategy
        if self.fusion_strategy == 'concat':
            # Concatenate along feature dimension
            fused = torch.cat([mod1_attended, mod2_attended], dim=-1)
            fused_output = self.fusion_layer(fused)
        elif self.fusion_strategy == 'add':
            # Element-wise addition
            fused = mod1_attended + mod2_attended
            fused_output = self.fusion_layer(fused)
        
        attention_weights = {
            'mod1_to_mod2': attn_weights_1_to_2,
            'mod2_to_mod1': attn_weights_2_to_1
        }
        
        return fused_output, attention_weights


if __name__ == "__main__":
    """Test the cross-attention module with sample data."""
    print("Testing Cross-Modal Attention Module...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Model parameters
    batch_size = 4
    seq_len1 = 50
    seq_len2 = 30
    d_model = 128
    num_heads = 8
    
    # Create sample input tensors
    modality1 = torch.randn(batch_size, seq_len1, d_model)
    modality2 = torch.randn(batch_size, seq_len2, d_model)
    
    print(f"Input modality1 shape: {modality1.shape}")
    print(f"Input modality2 shape: {modality2.shape}")
    
    # Initialize the cross-attention module
    cross_attention = CrossModalAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.1,
        fusion_strategy='concat'
    )
    
    print(f"\nModel initialized with d_model={d_model}, num_heads={num_heads}")
    print(f"Total parameters: {sum(p.numel() for p in cross_attention.parameters()):,}")
    
    # Forward pass
    fused_output, attention_weights = cross_attention(modality1, modality2)
    
    print(f"\nFused output shape: {fused_output.shape}")
    print(f"Attention weights keys: {list(attention_weights.keys())}")
    print(f"Attention mod1_to_mod2 shape: {attention_weights['mod1_to_mod2'].shape}")
    print(f"Attention mod2_to_mod1 shape: {attention_weights['mod2_to_mod1'].shape}")
    
    # Test with different fusion strategy
    cross_attention_add = CrossModalAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.1,
        fusion_strategy='add'
    )
    
    fused_output_add, _ = cross_attention_add(modality1, modality2)
    print(f"\nFused output (add strategy) shape: {fused_output_add.shape}")
    
    print("\n✓ Cross-Modal Attention Module test completed successfully!")
