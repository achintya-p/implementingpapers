"""
Demo script showing how to use the SimCLR implementation.
This script runs a mini training loop to demonstrate the usage.
"""

import torch
import torch.optim as optim
from data import get_cifar10_dataloaders
from model import create_simclr_model, LinearClassifier
from loss import NTXentLoss
from utils import set_seed, AverageMeter, cosine_lr_scheduler

def demo_pretraining(num_epochs=5, batch_size=64):
    """Demo pretraining for a few epochs"""
    print("🚀 Starting SimCLR demo pretraining...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data (small batch size for demo)
    pretrain_loader, train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=batch_size, num_workers=0
    )
    print(f"Data loaded: {len(pretrain_loader)} pretraining batches")
    
    # Create model
    model = create_simclr_model().to(device)
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create loss and optimizer
    criterion = NTXentLoss(temperature=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        losses = AverageMeter()
        pos_sims = AverageMeter()
        neg_sims = AverageMeter()
        
        # Adjust learning rate
        lr = cosine_lr_scheduler(optimizer, epoch, num_epochs, warmup_epochs=1, base_lr=3e-4)
        
        for batch_idx, ((view1, view2), _) in enumerate(pretrain_loader):
            view1, view2 = view1.to(device), view2.to(device)
            
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
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch}/{num_epochs}][{batch_idx}/{len(pretrain_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Pos: {pos_sim.item():.3f} '
                      f'Neg: {neg_sim.item():.3f} '
                      f'LR: {lr:.6f}')
            
            # For demo, only run a few batches
            if batch_idx >= 10:
                break
        
        print(f'Epoch {epoch} completed - Avg Loss: {losses.avg:.4f}')
    
    print("✅ Demo pretraining completed!")
    return model, train_loader, test_loader


def demo_linear_probe(model, train_loader, test_loader):
    """Demo linear probe evaluation"""
    print("\n🎯 Starting linear probe demo...")
    
    device = next(model.parameters()).device
    
    # Extract features from a subset of data
    model.eval()
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    
    with torch.no_grad():
        # Extract train features (first few batches for demo)
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx >= 10:  # Only use first 10 batches for demo
                break
            images = images.to(device)
            features = model.get_features(images)
            train_features.append(features.cpu())
            train_labels.append(labels)
        
        # Extract test features (first few batches for demo)
        for batch_idx, (images, labels) in enumerate(test_loader):
            if batch_idx >= 5:  # Only use first 5 batches for demo
                break
            images = images.to(device)
            features = model.get_features(images)
            test_features.append(features.cpu())
            test_labels.append(labels)
    
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    print(f"Features extracted: {train_features.shape} train, {test_features.shape} test")
    
    # Create and train linear classifier
    classifier = LinearClassifier(input_dim=train_features.size(1), num_classes=10).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train for a few epochs
    for epoch in range(5):
        classifier.train()
        
        # Simple mini-batch training
        batch_size = 64
        num_samples = len(train_features)
        indices = torch.randperm(num_samples)
        
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_indices = indices[i:end_idx]
            
            batch_features = train_features[batch_indices].to(device)
            batch_labels = train_labels[batch_indices].to(device)
            
            outputs = classifier(batch_features)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Linear probe epoch {epoch}: Loss {total_loss:.4f}')
    
    # Evaluate
    classifier.eval()
    with torch.no_grad():
        test_features_gpu = test_features.to(device)
        test_labels_gpu = test_labels.to(device)
        outputs = classifier(test_features_gpu)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == test_labels_gpu).float().mean().item()
    
    print(f"✅ Linear probe accuracy: {accuracy:.3f}")
    print("Note: This is just a demo with limited data - real performance will be higher!")


def main():
    """Run the complete demo"""
    print("=" * 60)
    print("SimCLR Implementation Demo")
    print("=" * 60)
    
    # Demo pretraining
    model, train_loader, test_loader = demo_pretraining(num_epochs=3, batch_size=64)
    
    # Demo linear probe
    demo_linear_probe(model, train_loader, test_loader)
    
    print("\n🎉 Demo completed! To run full training:")
    print("   python train_pretrain.py --epochs 100 --batch_size 256")
    print("   python train_linear.py --pretrained_path checkpoints/model_best.pth.tar")


if __name__ == '__main__':
    main()
