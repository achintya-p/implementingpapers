"""
LoRA (Low-Rank Adaptation) Implementation

Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
Authors: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, 
         Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
ArXiv: https://arxiv.org/abs/2106.09685

This module implements the core LoRA layer that can be used to adapt 
any linear layer in a neural network with low-rank matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LoRALayer(nn.Module):
    """
    Base LoRA layer that implements low-rank adaptation.
    
    The key idea is to represent the weight update ΔW as a product of two 
    low-rank matrices: ΔW = BA, where B ∈ R^{d×r} and A ∈ R^{r×k}
    with rank r << min(d,k).
    
    During forward pass: h = W₀x + ΔWx = W₀x + BAx
    where W₀ are the frozen pre-trained weights.
    """
    
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        merge_weights: bool = True
    ):
        """
        Initialize LoRA layer parameters.
        
        Args:
            r: Rank of the low-rank adaptation. Lower r = fewer parameters.
            lora_alpha: Scaling factor for LoRA weights. Controls the magnitude 
                       of the adaptation. Final scaling is lora_alpha/r.
            lora_dropout: Dropout rate applied to LoRA weights during training.
            merge_weights: Whether to merge LoRA weights with original weights
                          during inference for efficiency.
        """
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        self.merged = False
        
        # Scaling factor as described in the paper
        self.scaling = self.lora_alpha / self.r
        
        # Dropout layer
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = lambda x: x


class LoRALinear(LoRALayer):
    """
    LoRA adaptation of a linear layer.
    
    Replaces nn.Linear with frozen weights W₀ and trainable low-rank 
    matrices A and B such that the adapted output is:
    y = W₀x + (scaling * B @ A @ x)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        bias: bool = True,
        **kwargs
    ):
        """
        Initialize LoRA Linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features  
            r: Rank of LoRA adaptation
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout rate for LoRA weights
            fan_in_fan_out: Whether to use fan-in fan-out initialization
            merge_weights: Whether to merge weights during inference
            bias: Whether to include bias term
        """
        super().__init__(r, lora_alpha, lora_dropout, merge_weights)
        
        self.in_features = in_features
        self.out_features = out_features
        self.fan_in_fan_out = fan_in_fan_out
        
        # Original linear layer (frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA matrices
        if r > 0:
            # Matrix A: (r, in_features) - initialized with random Gaussian
            self.lora_A = nn.Parameter(torch.randn(r, in_features))
            # Matrix B: (out_features, r) - initialized to zero  
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            
            # Initialize A with random values, B with zeros (as in paper)
            self._reset_parameters()
            
        # Freeze the original linear layer weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
    def _reset_parameters(self):
        """Initialize LoRA matrices according to the paper."""
        if hasattr(self, 'lora_A'):
            # Initialize A with random Gaussian (similar to nn.Linear)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Initialize B with zeros so that initially ΔW = BA = 0
            nn.init.zeros_(self.lora_B)
    
    def train(self, mode: bool = True):
        """Override train mode to handle weight merging."""
        super().train(mode)
        if mode and self.merge_weights and self.merged:
            # If switching to training mode and weights are merged, unmerge them
            if self.r > 0:
                # Subtract the LoRA contribution from merged weights
                self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
        elif not mode and self.merge_weights and not self.merged:
            # If switching to eval mode and weights are not merged, merge them
            if self.r > 0:
                # Add the LoRA contribution to the original weights
                self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA linear layer.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Original linear transformation
        result = self.linear(x)
        
        # Add LoRA adaptation if rank > 0 and not merged
        if self.r > 0 and not self.merged:
            # Apply dropout to input if specified
            lora_input = self.lora_dropout_layer(x)
            # Compute LoRA contribution: scaling * B @ A @ x
            lora_output = (lora_input @ self.lora_A.T @ self.lora_B.T) * self.scaling
            result = result + lora_output
            
        return result
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'r={self.r}, lora_alpha={self.lora_alpha}, lora_dropout={self.lora_dropout}'


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """
    Freeze all parameters except LoRA parameters.
    
    This function sets requires_grad=False for all parameters except
    those containing 'lora_' in their name.
    
    Args:
        model: PyTorch model containing LoRA layers
    """
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def get_lora_parameters(model: nn.Module) -> list:
    """
    Get only the LoRA parameters from a model.
    
    Args:
        model: PyTorch model containing LoRA layers
        
    Returns:
        List of LoRA parameters that require gradients
    """
    return [param for name, param in model.named_parameters() 
            if 'lora_' in name and param.requires_grad]


def count_parameters(model: nn.Module) -> tuple:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Example usage and testing
    print("Testing LoRA Linear Layer...")
    
    # Create a LoRA linear layer
    lora_layer = LoRALinear(
        in_features=512, 
        out_features=256, 
        r=8, 
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(32, 512)  # Batch of 32, 512 features
    output = lora_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total, trainable = count_parameters(lora_layer)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Reduction factor: {total/trainable:.2f}x")
    
    # Test training/eval mode switching
    lora_layer.eval()
    output_eval = lora_layer(x)
    lora_layer.train()
    output_train = lora_layer(x)
    
    print(f"Output difference between train/eval: {torch.norm(output_eval - output_train)}")
