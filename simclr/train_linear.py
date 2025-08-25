"""
Linear probe evaluation for SimCLR.
Train a linear classifier on top of frozen encoder features.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import get_cifar10_dataloaders
from model import create_simclr_model, LinearClassifier
from utils import (
    set_seed, setup_logging, AverageMeter, save_checkpoint, load_checkpoint,
    accuracy, EarlyStopping, save_training_history
)


def extract_features(model, dataloader, device):
    """Extract features from the frozen encoder"""
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            features = model.get_features(images)
            
            features_list.append(features.cpu())
            labels_list.append(labels)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Extracting features: {batch_idx + 1}/{len(dataloader)}")
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return features, labels


def train_epoch(classifier, train_features, train_labels, criterion, optimizer, device, batch_size=256):
    """Train linear classifier for one epoch"""
    classifier.train()
    
    # Create random permutation for mini-batch training
    num_samples = len(train_features)
    indices = torch.randperm(num_samples)
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_features = train_features[batch_indices].to(device, non_blocking=True)
        batch_labels = train_labels[batch_indices].to(device, non_blocking=True)
        
        # Forward pass
        outputs = classifier(batch_features)
        loss = criterion(outputs, batch_labels)
        
        # Compute accuracy
        acc1 = accuracy(outputs, batch_labels, topk=(1,))[0]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), batch_features.size(0))
        top1.update(acc1.item(), batch_features.size(0))
    
    return losses.avg, top1.avg


def validate(classifier, test_features, test_labels, criterion, device, batch_size=256):
    """Validate linear classifier"""
    classifier.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    num_samples = len(test_features)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_features = test_features[start_idx:end_idx].to(device, non_blocking=True)
            batch_labels = test_labels[start_idx:end_idx].to(device, non_blocking=True)
            
            # Forward pass
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Compute accuracy
            acc1 = accuracy(outputs, batch_labels, topk=(1,))[0]
            
            # Update metrics
            losses.update(loss.item(), batch_features.size(0))
            top1.update(acc1.item(), batch_features.size(0))
    
    return losses.avg, top1.avg


def main():
    parser = argparse.ArgumentParser(description='SimCLR Linear Probe Evaluation')
    
    # Model parameters
    parser.add_argument('--pretrained_path', required=True, type=str,
                        help='Path to pretrained SimCLR model')
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'resnet50'],
                        help='Model architecture')
    parser.add_argument('--proj_hidden_dim', default=2048, type=int,
                        help='Projector hidden dimension')
    parser.add_argument('--proj_output_dim', default=128, type=int,
                        help='Projector output dimension')
    
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--patience', default=10, type=int,
                        help='Early stopping patience')
    
    # System parameters
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data workers')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    
    # Logging and checkpointing
    parser.add_argument('--save_dir', default='./linear_probe_results', type=str,
                        help='Directory to save results')
    parser.add_argument('--log_dir', default='./logs', type=str,
                        help='Directory to save logs')
    parser.add_argument('--experiment_name', default=None, type=str,
                        help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Set up device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up logging
    logger = setup_logging(args.log_dir, args.experiment_name)
    logger.info(f"Arguments: {args}")
    
    # Set up TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, 'linear_probe_tensorboard'))
    
    # Create data loaders
    _, train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"Dataset loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")
    
    # Load pretrained model
    model = create_simclr_model(
        arch=args.arch,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_output_dim=args.proj_output_dim
    ).to(device)
    
    if os.path.isfile(args.pretrained_path):
        logger.info(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.pretrained_path}")
    
    # Freeze the encoder
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info("Encoder frozen, extracting features...")
    
    # Extract features
    logger.info("Extracting training features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    
    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)
    
    logger.info(f"Features extracted: train {train_features.shape}, test {test_features.shape}")
    
    # Create linear classifier
    feature_dim = train_features.size(1)
    classifier = LinearClassifier(input_dim=feature_dim, num_classes=10).to(device)
    
    logger.info(f"Linear classifier created: {sum(p.numel() for p in classifier.parameters())} parameters")
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, restore_best_weights=True)
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_test_acc = 0.0
    start_time = time.time()
    
    logger.info("Starting linear probe training...")
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            classifier, train_features, train_labels, criterion, optimizer, device, args.batch_size
        )
        
        # Validate
        test_loss, test_acc = validate(
            classifier, test_features, test_labels, criterion, device, args.batch_size
        )
        
        # Update history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Update best accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            is_best = True
        else:
            is_best = False
        
        # Log epoch results
        logger.info(f'Epoch [{epoch}/{args.epochs}] '
                   f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% '
                   f'Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%')
        
        # TensorBoard logging
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_acc, epoch)
        
        # Early stopping check
        if early_stopping(test_loss, classifier):
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        # Save checkpoint
        if is_best:
            checkpoint_state = {
                'epoch': epoch + 1,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_test_acc': best_test_acc,
                'args': args,
                'history': history
            }
            
            save_checkpoint(
                checkpoint_state,
                args.save_dir,
                filename='best_linear_classifier.pth.tar'
            )
            
            logger.info(f'New best test accuracy: {best_test_acc:.2f}%')
    
    # Training completed
    total_time = time.time() - start_time
    logger.info(f'Linear probe training completed in {total_time:.0f}s')
    logger.info(f'Best test accuracy: {best_test_acc:.2f}%')
    
    # Save final results
    final_checkpoint = {
        'epoch': epoch + 1,
        'classifier_state_dict': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_test_acc': best_test_acc,
        'args': args,
        'history': history
    }
    
    save_checkpoint(
        final_checkpoint,
        args.save_dir,
        filename='final_linear_classifier.pth.tar'
    )
    
    # Save training history
    history_path = os.path.join(args.save_dir, 'linear_probe_history.json')
    save_training_history(history, history_path)
    
    writer.close()
    logger.info("Linear probe evaluation finished!")


if __name__ == '__main__':
    main()
