"""
Utility functions for SimCLR training and evaluation.
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import json
import logging
from datetime import datetime


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir="logs", experiment_name=None):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth.tar", is_best=False):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "model_best.pth.tar")
        torch.save(state, best_filepath)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    return epoch, best_acc


def cosine_lr_scheduler(optimizer, epoch, max_epochs, warmup_epochs=10, base_lr=3e-4, min_lr=1e-6):
    """Cosine learning rate schedule with warmup"""
    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


@torch.no_grad()
def knn_evaluation(model, train_loader, test_loader, device, k=200, temperature=0.1):
    """
    Evaluate model using k-NN classification on learned features.
    This provides a good proxy for the quality of learned representations.
    """
    model.eval()
    
    # Extract features and labels for training set
    train_features = []
    train_labels = []
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        features = model.get_features(images)
        features = F.normalize(features, dim=1)
        
        train_features.append(features.cpu())
        train_labels.append(labels)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Extracting train features: {batch_idx + 1}/{len(train_loader)}")
    
    train_features = torch.cat(train_features, dim=0).numpy()
    train_labels = torch.cat(train_labels, dim=0).numpy()
    
    # Extract features for test set
    test_features = []
    test_labels = []
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        features = model.get_features(images)
        features = F.normalize(features, dim=1)
        
        test_features.append(features.cpu())
        test_labels.append(labels)
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Extracting test features: {batch_idx + 1}/{len(test_loader)}")
    
    test_features = torch.cat(test_features, dim=0).numpy()
    test_labels = torch.cat(test_labels, dim=0).numpy()
    
    # Perform k-NN classification
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(train_features)
    
    # Find k nearest neighbors for each test sample
    distances, indices = knn.kneighbors(test_features)
    
    # Predict using weighted voting
    correct = 0
    total = len(test_features)
    
    for i in range(total):
        neighbor_labels = train_labels[indices[i]]
        neighbor_distances = distances[i]
        
        # Convert distances to similarities and apply temperature
        similarities = 1 - neighbor_distances
        weights = np.exp(similarities / temperature)
        
        # Weighted voting
        weighted_votes = np.bincount(neighbor_labels, weights=weights, minlength=10)
        predicted_label = np.argmax(weighted_votes)
        
        if predicted_label == test_labels[i]:
            correct += 1
    
    accuracy = correct / total
    return accuracy


def save_training_history(history, save_path):
    """Save training history to JSON file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)


def load_training_history(load_path):
    """Load training history from JSON file"""
    with open(load_path, 'r') as f:
        history = json.load(f)
    return history


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def accuracy(output, target, topk=(1,)):
    """Compute accuracy for specified topk values"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"AverageMeter test: avg={meter.avg:.2f}, expected=4.5")
    
    # Test cosine scheduler
    import torch.optim as optim
    model = torch.nn.Linear(10, 1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    lrs = []
    for epoch in range(100):
        lr = cosine_lr_scheduler(optimizer, epoch, 100, warmup_epochs=10, base_lr=3e-4)
        lrs.append(lr)
    
    print(f"LR schedule test: start={lrs[0]:.6f}, peak={max(lrs):.6f}, end={lrs[-1]:.6f}")
    
    print("Utilities test completed!")
