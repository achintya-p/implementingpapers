"""
LoRA Usage Examples

This script demonstrates the key concepts and usage patterns of LoRA
(Low-Rank Adaptation) with practical examples.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from lora_layer import LoRALinear, mark_only_lora_as_trainable, count_parameters
from lora_transformer import LoRATransformer


def example_1_basic_lora_layer():
    """Example 1: Basic LoRA layer usage and parameter efficiency."""
    print("=" * 60)
    print("EXAMPLE 1: Basic LoRA Layer")
    print("=" * 60)
    
    # Create a standard linear layer
    standard_layer = nn.Linear(1024, 512)
    standard_params = sum(p.numel() for p in standard_layer.parameters())
    
    # Create equivalent LoRA layer
    lora_layer = LoRALinear(
        in_features=1024,
        out_features=512,
        r=8,                    # Low rank
        lora_alpha=16,          # Scaling factor
        lora_dropout=0.1
    )
    
    # Count parameters
    total_params, trainable_params = count_parameters(lora_layer)
    
    print(f"Standard Linear Layer: {standard_params:,} parameters")
    print(f"LoRA Layer Total: {total_params:,} parameters")
    print(f"LoRA Layer Trainable: {trainable_params:,} parameters")
    print(f"Parameter Reduction: {total_params / trainable_params:.1f}x")
    
    # Test forward pass
    x = torch.randn(16, 1024)
    
    # Standard layer output
    with torch.no_grad():
        standard_output = standard_layer(x)
    
    # LoRA layer output (initially should be different due to random initialization)
    lora_output = lora_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {lora_output.shape}")
    print(f"Output difference norm: {torch.norm(standard_output - lora_output.detach()):.4f}")
    
    print("\n✅ Basic LoRA layer example completed!\n")


def example_2_weight_merging():
    """Example 2: Demonstrate weight merging for inference efficiency."""
    print("=" * 60)
    print("EXAMPLE 2: Weight Merging")
    print("=" * 60)
    
    # Create LoRA layer with merging enabled
    lora_layer = LoRALinear(
        in_features=256,
        out_features=128,
        r=4,
        lora_alpha=8,
        merge_weights=True  # Enable weight merging
    )
    
    x = torch.randn(8, 256)
    
    # Training mode (weights separate)
    lora_layer.train()
    train_output = lora_layer(x)
    print(f"Training mode - Weights merged: {lora_layer.merged}")
    
    # Evaluation mode (weights merged for efficiency)
    lora_layer.eval()
    eval_output = lora_layer(x)
    print(f"Evaluation mode - Weights merged: {lora_layer.merged}")
    
    # Outputs should be identical
    difference = torch.norm(train_output - eval_output)
    print(f"Output difference: {difference:.6f}")
    print("(Should be near zero - same computation, different implementation)")
    
    print("\n✅ Weight merging example completed!\n")


def example_3_selective_adaptation():
    """Example 3: Selective LoRA adaptation in transformers."""
    print("=" * 60)
    print("EXAMPLE 3: Selective Adaptation")
    print("=" * 60)
    
    configs = [
        {
            'name': 'Full Adaptation',
            'adapt_q': True, 'adapt_k': True, 'adapt_v': True, 'adapt_o': True,
            'adapt_feedforward': True
        },
        {
            'name': 'Attention Only',
            'adapt_q': True, 'adapt_k': False, 'adapt_v': True, 'adapt_o': False,
            'adapt_feedforward': False
        },
        {
            'name': 'Query/Value Only',
            'adapt_q': True, 'adapt_k': False, 'adapt_v': True, 'adapt_o': False,
            'adapt_feedforward': False
        }
    ]
    
    for config in configs:
        model = LoRATransformer(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            lora_r=8,
            adapt_attention=True,
            adapt_feedforward=config['adapt_feedforward'],
            adapt_q=config['adapt_q'],
            adapt_k=config['adapt_k'],
            adapt_v=config['adapt_v'],
            adapt_o=config['adapt_o']
        )
        
        model.configure_lora_training()
        stats = model.get_parameter_stats()
        
        print(f"{config['name']}:")
        print(f"  Trainable: {stats['trainable_parameters']:,} parameters")
        print(f"  Reduction: {stats['reduction_factor']:.1f}x")
        print()
    
    print("✅ Selective adaptation example completed!\n")


def example_4_parameter_scaling():
    """Example 4: Effect of LoRA rank and scaling on parameter count."""
    print("=" * 60)
    print("EXAMPLE 4: Parameter Scaling")
    print("=" * 60)
    
    base_model_size = 512
    
    print("Effect of LoRA Rank (r):")
    print("-" * 30)
    for r in [1, 2, 4, 8, 16, 32]:
        layer = LoRALinear(base_model_size, base_model_size, r=r, lora_alpha=16)
        total, trainable = count_parameters(layer)
        reduction = total / trainable
        print(f"  r={r:2d}: {trainable:,} trainable params, {reduction:.1f}x reduction")
    
    print("\nEffect of LoRA Alpha (scaling):")
    print("-" * 35)
    layer = LoRALinear(base_model_size, base_model_size, r=8, lora_alpha=16)
    x = torch.randn(1, base_model_size)
    
    # Test different alpha values
    for alpha in [1, 8, 16, 32]:
        layer.lora_alpha = alpha
        layer.scaling = alpha / layer.r
        output = layer(x)
        output_norm = torch.norm(output)
        print(f"  alpha={alpha:2d}: output norm = {output_norm:.4f}")
    
    print("\n✅ Parameter scaling example completed!\n")


def example_5_fine_tuning_simulation():
    """Example 5: Simulate fine-tuning process."""
    print("=" * 60)
    print("EXAMPLE 5: Fine-tuning Simulation")
    print("=" * 60)
    
    # Create a small model for quick demonstration
    model = LoRATransformer(
        vocab_size=100,
        d_model=128,
        num_heads=4,
        num_layers=2,
        lora_r=4,
        adapt_attention=True,
        adapt_feedforward=False
    )
    
    # Configure for LoRA training
    model.configure_lora_training()
    stats = model.get_parameter_stats()
    
    print(f"Model configured for LoRA training:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  Parameter reduction: {stats['reduction_factor']:.1f}x")
    
    # Create dummy data
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    targets = torch.randint(0, 100, (batch_size, seq_len))
    
    # Setup optimizer for LoRA parameters only
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = optim.Adam(lora_params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining with {len(lora_params)} LoRA parameter groups...")
    
    # Training loop
    model.train()
    initial_loss = None
    
    for step in range(5):
        optimizer.zero_grad()
        
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        if initial_loss is None:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step+1}: Loss = {loss.item():.4f}")
    
    improvement = initial_loss - loss.item()
    print(f"\nLoss improvement: {improvement:.4f}")
    print("(Positive value indicates learning)")
    
    print("\n✅ Fine-tuning simulation completed!\n")


def main():
    """Run all examples."""
    print("🚀 LoRA Implementation Examples")
    print("This script demonstrates key LoRA concepts with practical examples.\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    example_1_basic_lora_layer()
    example_2_weight_merging()
    example_3_selective_adaptation()
    example_4_parameter_scaling()
    example_5_fine_tuning_simulation()
    
    print("🎉 All examples completed successfully!")
    print("\nNext steps:")
    print("  1. Try running: python train_lora.py")
    print("  2. Experiment with different LoRA configurations")
    print("  3. Apply LoRA to your own models and tasks")


if __name__ == "__main__":
    main()
