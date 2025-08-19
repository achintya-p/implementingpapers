# LoRA: Low-Rank Adaptation Implementation

This repository contains a complete implementation of LoRA (Low-Rank Adaptation) from the paper:

**"LoRA: Low-Rank Adaptation of Large Language Models"**  
*Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*  
ArXiv: https://arxiv.org/abs/2106.09685

## 🎯 Overview

LoRA is a parameter-efficient fine-tuning technique that enables adapting large pre-trained models with significantly fewer trainable parameters. Instead of fine-tuning all model parameters, LoRA freezes the original weights and introduces trainable low-rank matrices into each layer.

### Key Benefits
- **Parameter Efficiency**: Reduces trainable parameters by up to 10,000x
- **Memory Efficiency**: Lower GPU memory requirements during training
- **Storage Efficiency**: Adapter weights are much smaller to store and share
- **No Inference Latency**: Can merge adapter weights for deployment
- **Task Switching**: Multiple adapters can be stored for different tasks

## 🏗️ Architecture

LoRA works by representing the weight update ΔW as a product of two low-rank matrices:

```
ΔW = B × A
```

Where:
- **A ∈ ℝ^(r×d)**: Initialized with random Gaussian values
- **B ∈ ℝ^(k×r)**: Initialized with zeros
- **r**: Rank (r << min(d,k))

The forward pass becomes:
```
h = W₀x + ΔWx = W₀x + BAx = W₀x + (scaling) × B(Ax)
```

## 📁 File Structure

```
lora/
├── lora_layer.py          # Core LoRA layer implementation
├── lora_transformer.py    # LoRA-adapted Transformer model
├── train_lora.py         # Training script with examples
├── README.md             # This documentation
└── examples/             # Usage examples (created when running scripts)
```

## 🚀 Quick Start

### 1. Basic LoRA Layer Usage

```python
from lora_layer import LoRALinear

# Replace any nn.Linear layer with LoRALinear
original_layer = nn.Linear(512, 256)
lora_layer = LoRALinear(
    in_features=512,
    out_features=256,
    r=8,                    # Rank (lower = fewer parameters)
    lora_alpha=16,          # Scaling factor
    lora_dropout=0.1        # Dropout for LoRA weights
)

# Use exactly like nn.Linear
x = torch.randn(32, 512)
output = lora_layer(x)  # Shape: [32, 256]
```

### 2. LoRA Transformer

```python
from lora_transformer import LoRATransformer

# Create a transformer with LoRA adapters
model = LoRATransformer(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    lora_r=8,               # LoRA rank
    lora_alpha=16,          # LoRA scaling
    adapt_attention=True,   # Apply LoRA to attention
    adapt_feedforward=False # Skip feedforward (for efficiency)
)

# Configure for LoRA training (freeze non-LoRA parameters)
model.configure_lora_training()

# Check parameter reduction
stats = model.get_parameter_stats()
print(f"Parameter reduction: {stats['reduction_factor']:.1f}x")
```

### 3. Training with LoRA

```python
from train_lora import LoRATrainer

# Create trainer
trainer = LoRATrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-4
)

# Train (only LoRA parameters will be updated)
history = trainer.train(epochs=10)

# Plot training progress
trainer.plot_training_history()
```

## 🔧 Installation & Dependencies

```bash
# Required packages
pip install torch torchvision
pip install matplotlib numpy
```

## 📊 Running the Examples

### Basic Test
```bash
# Test core LoRA layer
python lora_layer.py
```

### Transformer Test  
```bash
# Test LoRA transformer
python lora_transformer.py
```

### Full Training Example
```bash
# Run training with default parameters
python train_lora.py

# Custom training configuration
python train_lora.py --lora_r 16 --epochs 10 --batch_size 64
```

### Command Line Arguments
```bash
python train_lora.py --help

# Key arguments:
--lora_r 8              # LoRA rank (lower = fewer parameters)
--lora_alpha 16         # LoRA scaling factor  
--epochs 10             # Number of training epochs
--learning_rate 1e-4    # Learning rate for LoRA parameters
--batch_size 32         # Training batch size
```

## 🎨 Key Features

### 1. Flexible LoRA Application
- **Selective Adaptation**: Choose which layers to adapt (Q, K, V, O projections)
- **Rank Control**: Adjust rank `r` to balance efficiency vs. performance
- **Scaling Control**: Use `lora_alpha` to control adaptation magnitude

### 2. Parameter Efficiency
```python
# Example parameter reduction
Original Model: 125M parameters
LoRA Adapted:   1.2M trainable parameters  
Reduction:      104x fewer parameters!
```

### 3. Memory Efficiency
- Only LoRA parameters require gradients
- Significantly reduced memory usage during training
- Original weights remain frozen

### 4. Weight Merging
```python
# During training: separate weights
output = W₀x + (scaling × B × A × x)

# During inference: merged weights (no latency overhead)  
W_merged = W₀ + (scaling × B × A)
output = W_merged × x
```

## 📈 Performance Tips

### 1. Rank Selection
- **Low rank (r=1-4)**: Maximum efficiency, may limit performance
- **Medium rank (r=8-16)**: Good balance for most tasks
- **High rank (r=32+)**: Better performance, less efficient

### 2. Layer Selection
Based on the paper's findings:
- **Query & Value**: Most important for performance
- **Key**: Less critical, can skip for efficiency
- **Feedforward**: Often unnecessary, focus on attention

### 3. Scaling Factor
- **lora_alpha = r**: No scaling (original paper default)
- **lora_alpha > r**: Amplify LoRA contributions
- **lora_alpha < r**: Reduce LoRA contributions

## 🔬 Research Applications

### 1. Few-Shot Learning
```python
# Quick adaptation to new tasks with minimal data
model.configure_lora_training()
# Train on 100-1000 examples per class
```

### 2. Multi-Task Learning
```python
# Save different adapters for different tasks
torch.save(model.lora_state_dict(), 'task_A_adapter.pt')
# Switch adapters without retraining base model
```

### 3. Domain Adaptation
```python
# Adapt pre-trained models to specific domains
# Medical, Legal, Scientific, etc.
```

## 📚 Paper Implementation Details

This implementation faithfully follows the original paper:

1. **Matrix Initialization**: A ~ N(0, σ²), B = 0
2. **Scaling**: Uses α/r scaling factor  
3. **Gradient Flow**: Only through LoRA parameters
4. **Weight Merging**: For inference efficiency
5. **Selective Application**: Configurable layer adaptation

## 🤝 Contributing

Feel free to contribute improvements:

1. **Optimizations**: Memory/compute efficiency improvements
2. **Features**: Additional layer types, initialization schemes
3. **Examples**: More diverse use cases and datasets
4. **Documentation**: Better explanations and tutorials

## 📄 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```

## 📞 Support

For questions or issues:
1. Check the code comments for detailed explanations
2. Run the test scripts to verify functionality
3. Refer to the original paper for theoretical details

---

**Happy fine-tuning with LoRA! 🚀**
