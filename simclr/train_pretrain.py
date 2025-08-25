"""
SimCLR pretraining script.
Self-supervised learning using contrastive loss on augmented image pairs.
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data import get_cifar10_dataloaders
from model import create_simclr_model
from loss import NTXentLoss
from utils import (
    set_seed, setup_logging, AverageMeter, save_checkpoint,
    cosine_lr_scheduler, knn_evaluation, save_training_history
)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pos_sims = AverageMeter()
    neg_sims = AverageMeter()
    
    end = time.time()
    
    for batch_idx, ((view1, view2), _) in enumerate(dataloader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Move to device
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)
        
        # Forward pass
        z1 = model(view1)
        z2 = model(view2)
        
        # Compute loss
        loss, pos_sim, neg_sim = criterion(z1, z2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), view1.size(0))
        pos_sims.update(pos_sim.item(), view1.size(0))
        neg_sims.update(neg_sim.item(), view1.size(0))
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Log progress
        if batch_idx % 50 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(dataloader)}] '
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f}) '
                  f'Pos {pos_sims.val:.3f} ({pos_sims.avg:.3f}) '
                  f'Neg {neg_sims.val:.3f} ({neg_sims.avg:.3f})')
        
        # TensorBoard logging
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/PosSim', pos_sim.item(), global_step)
            writer.add_scalar('Train/NegSim', neg_sim.item(), global_step)
    
    return losses.avg, pos_sims.avg, neg_sims.avg


def main():
    parser = argparse.ArgumentParser(description='SimCLR Pretraining')
    
    # Model parameters
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
    parser.add_argument('--lr', default=3e-4, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='Temperature for NT-Xent loss')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Warmup epochs for learning rate')
    
    # System parameters
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data workers')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    
    # Logging and checkpointing
    parser.add_argument('--save_dir', default='./checkpoints', type=str,
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', default='./logs', type=str,
                        help='Directory to save logs')
    parser.add_argument('--experiment_name', default=None, type=str,
                        help='Experiment name for logging')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='Save checkpoint every N epochs')
    
    # Evaluation parameters
    parser.add_argument('--eval_freq', default=10, type=int,
                        help='Evaluate using kNN every N epochs')
    parser.add_argument('--knn_k', default=200, type=int,
                        help='Number of neighbors for kNN evaluation')
    parser.add_argument('--knn_temperature', default=0.1, type=float,
                        help='Temperature for kNN evaluation')
    
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
    writer = SummaryWriter(os.path.join(args.log_dir, 'tensorboard'))
    
    # Create data loaders
    pretrain_loader, train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    logger.info(f"Dataset loaded: {len(pretrain_loader)} batches for pretraining")
    
    # Create model
    model = create_simclr_model(
        arch=args.arch,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_output_dim=args.proj_output_dim
    ).to(device)
    
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create loss function
    criterion = NTXentLoss(temperature=args.temperature).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training history
    history = {
        'epoch': [],
        'loss': [],
        'pos_sim': [],
        'neg_sim': [],
        'lr': [],
        'knn_acc': []
    }
    
    best_knn_acc = 0.0
    start_time = time.time()
    
    logger.info("Starting pretraining...")
    
    for epoch in range(args.epochs):
        # Adjust learning rate
        lr = cosine_lr_scheduler(
            optimizer, epoch, args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr
        )
        
        # Train for one epoch
        train_loss, train_pos_sim, train_neg_sim = train_epoch(
            model, pretrain_loader, criterion, optimizer, device, epoch, writer
        )
        
        # Update history
        history['epoch'].append(epoch)
        history['loss'].append(train_loss)
        history['pos_sim'].append(train_pos_sim)
        history['neg_sim'].append(train_neg_sim)
        history['lr'].append(lr)
        
        # Log epoch results
        logger.info(f'Epoch [{epoch}/{args.epochs}] '
                   f'Loss: {train_loss:.4f} '
                   f'Pos Sim: {train_pos_sim:.3f} '
                   f'Neg Sim: {train_neg_sim:.3f} '
                   f'LR: {lr:.6f}')
        
        # TensorBoard logging
        writer.add_scalar('Epoch/Loss', train_loss, epoch)
        writer.add_scalar('Epoch/PosSim', train_pos_sim, epoch)
        writer.add_scalar('Epoch/NegSim', train_neg_sim, epoch)
        writer.add_scalar('Epoch/LR', lr, epoch)
        
        # k-NN evaluation
        knn_acc = 0.0
        if (epoch + 1) % args.eval_freq == 0:
            logger.info("Running k-NN evaluation...")
            knn_acc = knn_evaluation(
                model, train_loader, test_loader, device,
                k=args.knn_k, temperature=args.knn_temperature
            )
            logger.info(f'k-NN Accuracy: {knn_acc:.3f}')
            writer.add_scalar('Eval/kNN_Accuracy', knn_acc, epoch)
            
            # Save best model
            if knn_acc > best_knn_acc:
                best_knn_acc = knn_acc
                is_best = True
            else:
                is_best = False
        else:
            is_best = False
        
        history['knn_acc'].append(knn_acc)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_knn_acc': best_knn_acc,
                'args': args,
                'history': history
            }
            
            save_checkpoint(
                checkpoint_state,
                args.save_dir,
                filename=f'checkpoint_epoch_{epoch+1}.pth.tar',
                is_best=is_best
            )
            
            if is_best:
                logger.info(f'New best k-NN accuracy: {best_knn_acc:.3f}')
    
    # Training completed
    total_time = time.time() - start_time
    logger.info(f'Training completed in {total_time:.0f}s')
    logger.info(f'Best k-NN accuracy: {best_knn_acc:.3f}')
    
    # Save final model and history
    final_checkpoint = {
        'epoch': args.epochs,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_knn_acc': best_knn_acc,
        'args': args,
        'history': history
    }
    
    save_checkpoint(
        final_checkpoint,
        args.save_dir,
        filename='final_model.pth.tar'
    )
    
    # Save training history
    history_path = os.path.join(args.save_dir, 'training_history.json')
    save_training_history(history, history_path)
    
    writer.close()
    logger.info("Pretraining finished!")


if __name__ == '__main__':
    main()
