"""
Data loading and augmentations for SimCLR.
Implements PairTransform for creating augmented pairs and CIFAR-10 dataloaders.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import ImageFilter
import random


class GaussianBlur:
    """Gaussian blur augmentation from SimCLR paper"""
    
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class PairTransform:
    """Transform that returns two augmented views of the same image"""
    
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        # Apply the same base transform twice independently
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)
        return view1, view2


def get_simclr_transforms(size=32):
    """
    Get SimCLR augmentation pipeline following the paper:
    1. RandomResizedCrop
    2. RandomHorizontalFlip  
    3. ColorJitter with strong settings
    4. RandomGrayscale
    5. GaussianBlur
    """
    
    # Color jitter parameters from SimCLR paper
    color_jitter = transforms.ColorJitter(
        brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
    )
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur()], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 stats
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    return transform


def get_test_transform(size=32):
    """Standard test transform without augmentation"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])


def get_cifar10_dataloaders(batch_size=256, num_workers=4, download=True):
    """
    Get CIFAR-10 dataloaders for SimCLR pretraining and linear probe.
    
    Returns:
        pretrain_loader: DataLoader with PairTransform for SSL training
        train_loader: Standard train loader for linear probe  
        test_loader: Standard test loader for evaluation
    """
    
    # Transforms
    simclr_transform = get_simclr_transforms()
    pair_transform = PairTransform(simclr_transform)
    test_transform = get_test_transform()
    
    # Datasets
    pretrain_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=download, transform=pair_transform
    )
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=test_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=test_transform
    )
    
    # DataLoaders
    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return pretrain_loader, train_loader, test_loader


if __name__ == "__main__":
    # Quick test
    pretrain_loader, train_loader, test_loader = get_cifar10_dataloaders(batch_size=4)
    
    # Test PairTransform
    for batch_idx, ((view1, view2), labels) in enumerate(pretrain_loader):
        print(f"Batch {batch_idx}")
        print(f"View1 shape: {view1.shape}")
        print(f"View2 shape: {view2.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Check that views are different (they should be due to random augmentation)
        diff = torch.abs(view1 - view2).mean().item()
        print(f"Mean absolute difference between views: {diff:.4f}")
        
        if batch_idx >= 2:
            break
    
    print("\nData loading test passed!")
