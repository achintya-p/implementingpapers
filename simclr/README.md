# SimCLR Implementation

A clean, well-documented PyTorch implementation of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) following the original paper and best practices.

## Overview

This implementation provides:
- **Complete SimCLR pipeline**: Data augmentation, contrastive learning, and linear probe evaluation
- **Modular design**: Separate components for easy understanding and modification
- **Comprehensive testing**: Unit tests for all major components
- **Detailed logging**: TensorBoard integration and progress tracking
- **Flexible configuration**: Command-line arguments for all hyperparameters

## Project Structure

```
simclr/
├── data.py              # Dataset loading and augmentations
├── model.py             # Encoder and projection head
├── loss.py              # NT-Xent contrastive loss
├── train_pretrain.py    # Self-supervised pretraining script
├── train_linear.py      # Linear probe evaluation script
├── utils.py             # Utilities for logging, checkpoints, etc.
├── tests/               # Unit tests
│   └── test_components.py
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Quick Start

### 1. Installation

```bash
# Clone or download this implementation
cd simclr

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Run unit tests to verify everything works
python -m pytest tests/ -v

# Or run directly
cd tests && python test_components.py
```

### 3. Pretraining

```bash
# Basic pretraining with default settings
python train_pretrain.py

# With custom parameters
python train_pretrain.py \
    --epochs 100 \
    --batch_size 256 \
    --lr 3e-4 \
    --temperature 0.1 \
    --save_dir ./checkpoints \
    --experiment_name simclr_resnet18
```

### 4. Linear Probe Evaluation

```bash
# Evaluate pretrained model
python train_linear.py \
    --pretrained_path ./checkpoints/model_best.pth.tar \
    --save_dir ./linear_probe_results
```

## Algorithm Details

### SimCLR Overview

1. **Augmentation**: Create two random augmentations of each image
2. **Encoding**: Pass through shared ResNet encoder f(·)
3. **Projection**: Apply MLP projection head g(·)
4. **Contrastive Loss**: Optimize NT-Xent loss to bring positive pairs closer
5. **Linear Probe**: Train linear classifier on frozen encoder features

### Augmentation Pipeline

Following the SimCLR paper, we use:
1. RandomResizedCrop (scale 0.2-1.0)
2. RandomHorizontalFlip (p=0.5)
3. ColorJitter (brightness, contrast, saturation=0.8, hue=0.2, p=0.8)
4. RandomGrayscale (p=0.2)
5. GaussianBlur (p=0.5)

### Architecture

- **Encoder**: ResNet-18 (default) or ResNet-50 without classifier head
- **Projector**: 2-layer MLP (512→2048→128) with BatchNorm and ReLU
- **Output**: L2-normalized 128-dimensional embeddings

### Loss Function

NT-Xent (Normalized Temperature-scaled Cross-Entropy):
```
ℓ(i,j) = -log(exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ))
```

Where:
- `sim(u,v) = u^T v / (||u|| ||v||)` (cosine similarity)
- `τ` is temperature (default: 0.1)
- Sum over all negatives in the batch

## Training Configuration

### Recommended Hyperparameters

**Pretraining (CIFAR-10)**:
- Batch size: 256+ (use gradient accumulation if GPU memory limited)
- Learning rate: 3e-4 with cosine decay + 10-epoch warmup
- Weight decay: 1e-4
- Temperature: 0.1
- Epochs: 100 (50 minimum for reasonable results)
- Optimizer: AdamW

**Linear Probe**:
- Learning rate: 1e-3
- Optimizer: Adam
- Epochs: 100 with early stopping
- Weight decay: 1e-4

### Expected Results

On CIFAR-10 with ResNet-18:
- **Random baseline**: ~10% accuracy
- **After 50 epochs**: ~60-70% linear probe accuracy  
- **After 100 epochs**: ~70-80% linear probe accuracy
- **Supervised baseline**: ~85-90% accuracy

## Advanced Usage

### Custom Datasets

```python
from data import PairTransform, get_simclr_transforms

# Create custom dataset with SimCLR augmentations
transform = PairTransform(get_simclr_transforms(size=224))  # For ImageNet size
dataset = YourCustomDataset(transform=transform)
```

### Monitoring Training

```bash
# View training progress with TensorBoard
tensorboard --logdir ./logs/tensorboard
```

### k-NN Evaluation

The pretraining script includes k-NN evaluation every 10 epochs as a proxy for representation quality:

```python
from utils import knn_evaluation

# Evaluate representations using k-NN
knn_acc = knn_evaluation(model, train_loader, test_loader, device, k=200)
```

## Implementation Notes

### Key Features

1. **Numerical Stability**: Proper temperature scaling and max subtraction in softmax
2. **Memory Efficiency**: Gradient accumulation support for large effective batch sizes
3. **Reproducibility**: Seed setting and deterministic operations
4. **Flexible Architecture**: Easy to swap backbones and projection heads
5. **Comprehensive Logging**: Loss curves, similarities, learning rates, and k-NN scores

### Design Choices

- **Two NT-Xent implementations**: Standard and simplified (cross-entropy based)
- **Feature extraction**: Separate method for linear probe evaluation
- **Modular components**: Easy to modify individual parts
- **Extensive testing**: Unit tests for all critical components

## Troubleshooting

### Common Issues

**Low performance:**
- Check batch size (need 256+ for best results)
- Verify augmentations are working (views should be different)
- Ensure temperature is reasonable (0.05-0.2)
- Check learning rate schedule

**Memory issues:**
- Reduce batch size and use gradient accumulation
- Use mixed precision training (add to train_pretrain.py)
- Consider smaller projection dimensions

**Debugging:**
```python
# Quick verification
python data.py      # Test data loading
python model.py     # Test model forward pass  
python loss.py      # Test loss computation
python utils.py     # Test utilities
```

## Extensions

### Implemented in Roadmap

The following extensions from the roadmap can be easily added:

1. **Memory Bank**: Modify loss.py to include cross-batch negatives
2. **Alternative Losses**: Implement BYOL or SimSiam variants
3. **Multi-crop**: Extend PairTransform for SwAV-style augmentations
4. **Transfer Learning**: Evaluate on other datasets
5. **Vision Transformer**: Replace ResNet with ViT backbone

### Example: Adding BYOL

```python
# In model.py, add predictor for BYOL
class BYOLPredictor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
```

## Citation

If you use this implementation, please cite the original SimCLR paper:

```bibtex
@article{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={International conference on machine learning},
  pages={1597--1607},
  year={2020},
  organization={PMLR}
}
```

## License

This implementation is provided for educational and research purposes.
