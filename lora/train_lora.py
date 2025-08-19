"""
LoRA Fine-tuning Training Script

This script demonstrates how to use LoRA (Low-Rank Adaptation) to efficiently 
fine-tune a pre-trained transformer model on a downstream task.

The example shows:
1. Loading a pre-trained model
2. Adding LoRA adapters 
3. Freezing original parameters
4. Training only LoRA parameters
5. Evaluating the adapted model

Usage:
    python train_lora.py --task classification --epochs 10 --lora_r 8
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import time
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

from lora_transformer import LoRATransformer
from lora_layer import mark_only_lora_as_trainable, count_parameters, get_lora_parameters


class LoRATrainer:
    """
    Trainer class for LoRA fine-tuning experiments.
    
    This class handles the complete training pipeline including:
    - Model setup with LoRA adapters
    - Training loop with proper parameter freezing
    - Evaluation and metric tracking
    - Visualization of training progress
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
    ):
        """
        Initialize the LoRA trainer.
        
        Args:
            model: Model with LoRA adapters
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use for training
            learning_rate: Learning rate for LoRA parameters
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps for learning rate schedule
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Configure model for LoRA training
        self.model.configure_lora_training()
        
        # Setup optimizer - only optimize LoRA parameters
        lora_params = get_lora_parameters(self.model)
        self.optimizer = optim.AdamW(lora_params, lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=10,  # Will be updated when training starts
            pct_start=warmup_steps / (len(train_loader) * 10)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Print model statistics
        self._print_model_stats()
    
    def _print_model_stats(self):
        """Print model parameter statistics."""
        total_params, trainable_params = count_parameters(self.model)
        frozen_params = total_params - trainable_params
        reduction_factor = total_params / trainable_params if trainable_params > 0 else float('inf')
        
        print(f"Model Parameter Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable (LoRA) parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Parameter reduction: {reduction_factor:.2f}x")
        print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # For classification, we typically use the last token or pool
            # Here we'll use the mean of all token outputs for simplicity
            if outputs.dim() == 3:  # [batch, seq_len, vocab_size]
                outputs = outputs.mean(dim=1)  # [batch, vocab_size]
            
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(get_lora_parameters(self.model), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            self.history['learning_rates'].append(current_lr)
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'LR: {current_lr:.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                
                # Handle output dimensions
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1)
                
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs: int) -> Dict:
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        print(f"Starting LoRA fine-tuning for {epochs} epochs...")
        
        # Update scheduler for correct number of epochs
        current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else 1e-4
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=current_lr,
            steps_per_epoch=len(self.train_loader),
            epochs=epochs,
            pct_start=0.1
        )
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_lora_model.pt')
                print(f'  New best validation accuracy: {best_val_acc:.2f}%')
            
            print('-' * 50)
        
        total_time = time.time() - start_time
        print(f'Training completed in {total_time:.2f}s')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        return self.history
    
    def save_model(self, path: str):
        """Save LoRA parameters and model configuration."""
        # Save only LoRA parameters to reduce file size
        lora_state_dict = {
            name: param for name, param in self.model.named_parameters()
            if 'lora_' in name
        }
        
        torch.save({
            'lora_state_dict': lora_state_dict,
            'training_history': self.history,
        }, path)
        
    def load_model(self, path: str):
        """Load LoRA parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load only LoRA parameters
        model_dict = self.model.state_dict()
        lora_dict = checkpoint['lora_state_dict']
        model_dict.update(lora_dict)
        self.model.load_state_dict(model_dict)
        
        if 'training_history' in checkpoint:
            self.history = checkpoint['training_history']
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        if self.history['learning_rates']:
            ax3.plot(self.history['learning_rates'])
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True)
        
        # Parameter efficiency comparison
        total_params, trainable_params = count_parameters(self.model)
        labels = ['Frozen\nParameters', 'LoRA\nParameters']
        sizes = [total_params - trainable_params, trainable_params]
        colors = ['lightcoral', 'lightskyblue']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Parameter Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_dummy_dataset(vocab_size: int, seq_len: int, num_classes: int, 
                        num_samples: int) -> TensorDataset:
    """Create a dummy dataset for demonstration."""
    # Random token sequences
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    # Random labels
    targets = torch.randint(0, num_classes, (num_samples,))
    
    return TensorDataset(inputs, targets)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='LoRA Fine-tuning Example')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create model with LoRA
    print("Creating LoRA Transformer model...")
    model = LoRATransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        adapt_attention=True,
        adapt_feedforward=False,  # Only adapt attention for efficiency
    )
    
    # Create dummy datasets
    print("Creating dummy datasets...")
    train_dataset = create_dummy_dataset(args.vocab_size, args.seq_len, 
                                       args.num_classes, 1000)
    val_dataset = create_dummy_dataset(args.vocab_size, args.seq_len, 
                                     args.num_classes, 200)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create trainer
    trainer = LoRATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate
    )
    
    # Train model
    history = trainer.train(args.epochs)
    
    # Plot results
    trainer.plot_training_history('lora_training_history.png')
    
    # Save training results
    with open('lora_training_results.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("Training completed! Results saved to lora_training_history.png and lora_training_results.json")


if __name__ == "__main__":
    main()
