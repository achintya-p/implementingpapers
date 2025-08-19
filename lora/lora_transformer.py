"""
LoRA-Adapted Transformer Implementation

This module demonstrates how to apply LoRA (Low-Rank Adaptation) to a 
Transformer model, specifically targeting the attention mechanism's 
query, key, value, and output projections.

The paper shows that adapting just the query and value projections 
often provides the best performance-efficiency trade-off.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from lora_layer import LoRALinear, mark_only_lora_as_trainable, count_parameters


class LoRAMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with LoRA adaptation applied to Q, K, V, and output projections.
    
    This implementation allows selective application of LoRA to different
    projection matrices. The paper suggests that adapting only Q and V
    projections often provides the best results.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        # LoRA parameters
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        # Which projections to adapt with LoRA
        adapt_q: bool = True,
        adapt_k: bool = False,  # Paper suggests K adaptation is less important
        adapt_v: bool = True,
        adapt_o: bool = False,  # Output projection adaptation
    ):
        """
        Initialize LoRA Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout rate
            adapt_q: Whether to apply LoRA to query projection
            adapt_k: Whether to apply LoRA to key projection  
            adapt_v: Whether to apply LoRA to value projection
            adapt_o: Whether to apply LoRA to output projection
        """
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1 / math.sqrt(self.d_k)
        
        # Query, Key, Value projections with optional LoRA
        if adapt_q:
            self.w_q = LoRALinear(d_model, d_model, r=lora_r, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout, bias=False)
        else:
            self.w_q = nn.Linear(d_model, d_model, bias=False)
            
        if adapt_k:
            self.w_k = LoRALinear(d_model, d_model, r=lora_r, lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout, bias=False)
        else:
            self.w_k = nn.Linear(d_model, d_model, bias=False)
            
        if adapt_v:
            self.w_v = LoRALinear(d_model, d_model, r=lora_r, lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout, bias=False)
        else:
            self.w_v = nn.Linear(d_model, d_model, bias=False)
            
        if adapt_o:
            self.w_o = LoRALinear(d_model, d_model, r=lora_r, lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout, bias=False)
        else:
            self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for multi-head attention with LoRA.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]  
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections with optional LoRA adaptation
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)    # [batch_size, seq_len, d_model]
        V = self.w_v(value)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return output


class LoRATransformerBlock(nn.Module):
    """
    Transformer block with LoRA-adapted attention and optional LoRA on feedforward layers.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        # LoRA parameters
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        # Which components to adapt
        adapt_attention: bool = True,
        adapt_feedforward: bool = False,
        # Attention-specific LoRA settings
        adapt_q: bool = True,
        adapt_k: bool = False,
        adapt_v: bool = True,
        adapt_o: bool = False,
    ):
        """
        Initialize LoRA Transformer block.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feedforward dimension
            dropout: Dropout rate
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout rate
            adapt_attention: Whether to apply LoRA to attention layers
            adapt_feedforward: Whether to apply LoRA to feedforward layers
            adapt_q, adapt_k, adapt_v, adapt_o: Which attention projections to adapt
        """
        super().__init__()
        
        # Multi-head attention with optional LoRA
        if adapt_attention:
            self.attention = LoRAMultiHeadAttention(
                d_model, num_heads, dropout, lora_r, lora_alpha, lora_dropout,
                adapt_q, adapt_k, adapt_v, adapt_o
            )
        else:
            self.attention = LoRAMultiHeadAttention(
                d_model, num_heads, dropout, 0, lora_alpha, lora_dropout,
                False, False, False, False  # No LoRA adaptation
            )
        
        # Feedforward network with optional LoRA
        if adapt_feedforward:
            self.ff1 = LoRALinear(d_model, d_ff, r=lora_r, lora_alpha=lora_alpha, 
                                lora_dropout=lora_dropout)
            self.ff2 = LoRALinear(d_ff, d_model, r=lora_r, lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout)
        else:
            self.ff1 = nn.Linear(d_model, d_ff)
            self.ff2 = nn.Linear(d_ff, d_model)
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual connection
        ff_out = self.ff2(F.relu(self.ff1(x)))
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class LoRATransformer(nn.Module):
    """
    Complete Transformer model with LoRA adaptation capabilities.
    
    This model can be used as a drop-in replacement for standard transformers
    with the added benefit of parameter-efficient fine-tuning via LoRA.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        # LoRA parameters
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        # Adaptation settings
        adapt_attention: bool = True,
        adapt_feedforward: bool = False,
        adapt_embedding: bool = False,
        # Attention-specific settings
        adapt_q: bool = True,
        adapt_k: bool = False,
        adapt_v: bool = True,
        adapt_o: bool = False,
    ):
        """
        Initialize LoRA Transformer.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feedforward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            lora_r: LoRA rank (lower = fewer parameters)
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout for LoRA weights
            adapt_attention: Apply LoRA to attention layers
            adapt_feedforward: Apply LoRA to feedforward layers
            adapt_embedding: Apply LoRA to embedding layer
            adapt_q, adapt_k, adapt_v, adapt_o: Which attention projections to adapt
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token and positional embeddings
        if adapt_embedding:
            # Note: LoRA for embeddings requires special handling since embedding
            # is not a linear layer. This is a simplified version.
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            # For now, we don't adapt embeddings with LoRA (would need custom implementation)
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            LoRATransformerBlock(
                d_model, num_heads, d_ff, dropout,
                lora_r, lora_alpha, lora_dropout,
                adapt_attention, adapt_feedforward,
                adapt_q, adapt_k, adapt_v, adapt_o
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights according to standard practice."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Create position IDs
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        pos_emb = self.pos_embedding(pos_ids)        # [1, seq_len, d_model]
        x = token_emb + pos_emb
        
        # Convert attention mask to the format expected by attention layers
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.expand(batch_size, 1, seq_len, seq_len)
        else:
            extended_mask = None
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, extended_mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def configure_lora_training(self):
        """Configure the model for LoRA training by freezing non-LoRA parameters."""
        mark_only_lora_as_trainable(self)
        
    def get_parameter_stats(self) -> dict:
        """Get statistics about model parameters."""
        total_params, trainable_params = count_parameters(self)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'reduction_factor': total_params / trainable_params if trainable_params > 0 else float('inf')
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing LoRA Transformer...")
    
    # Create a small LoRA transformer for testing
    model = LoRATransformer(
        vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_len=512,
        lora_r=8,
        lora_alpha=16,
        adapt_attention=True,
        adapt_feedforward=False,  # Only adapt attention for efficiency
    )
    
    # Configure for LoRA training
    model.configure_lora_training()
    
    # Print parameter statistics
    stats = model.get_parameter_stats()
    print(f"Total parameters: {stats['total_parameters']:,}")
    print(f"Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"Frozen parameters: {stats['frozen_parameters']:,}")
    print(f"Parameter reduction: {stats['reduction_factor']:.2f}x")
    
    # Test forward pass
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("LoRA Transformer test completed successfully!")
