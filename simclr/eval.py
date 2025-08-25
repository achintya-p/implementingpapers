"""
Evaluation utilities for SimCLR.
Includes k-NN evaluation and linear probe evaluation helpers.
"""

import argparse
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os

from data import get_cifar10_dataloaders
from model import create_simclr_model, LinearClassifier
from utils import load_checkpoint, accuracy, setup_logging


def extract_features_and_labels(model, dataloader, device):
    """Extract features and labels from a dataloader"""
    model.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            features = model.get_features(images)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return features, labels


def knn_evaluation_detailed(train_features, train_labels, test_features, test_labels, 
                          k_values=[1, 5, 10, 20, 50, 100, 200], temperature=0.1):
    """
    Detailed k-NN evaluation with multiple k values.
    Returns accuracy for each k value.
    """
    train_features_np = F.normalize(train_features, dim=1).numpy()
    test_features_np = F.normalize(test_features, dim=1).numpy()
    train_labels_np = train_labels.numpy()
    test_labels_np = test_labels.numpy()
    
    results = {}
    
    for k in k_values:
        print(f"Evaluating k-NN with k={k}...")
        
        # Fit k-NN
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(train_features_np)
        
        # Find neighbors
        distances, indices = knn.kneighbors(test_features_np)
        
        # Predict using weighted voting
        correct = 0
        total = len(test_features_np)
        
        for i in range(total):
            neighbor_labels = train_labels_np[indices[i]]
            neighbor_distances = distances[i]
            
            # Convert distances to similarities and apply temperature
            similarities = 1 - neighbor_distances
            weights = np.exp(similarities / temperature)
            
            # Weighted voting
            weighted_votes = np.bincount(neighbor_labels, weights=weights, minlength=10)
            predicted_label = np.argmax(weighted_votes)
            
            if predicted_label == test_labels_np[i]:
                correct += 1
        
        accuracy = correct / total
        results[k] = accuracy
        print(f"k={k}: {accuracy:.3f}")
    
    return results


def evaluate_linear_probe(pretrained_model_path, classifier_path, device):
    """Evaluate a trained linear probe"""
    # Load data
    _, train_loader, test_loader = get_cifar10_dataloaders(batch_size=256, num_workers=4)
    
    # Load pretrained model
    model = create_simclr_model().to(device)
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Extract features
    print("Extracting test features...")
    test_features, test_labels = extract_features_and_labels(model, test_loader, device)
    
    # Load classifier
    classifier = LinearClassifier(input_dim=test_features.size(1), num_classes=10).to(device)
    classifier_checkpoint = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(classifier_checkpoint['classifier_state_dict'])
    
    # Evaluate
    classifier.eval()
    correct = 0
    total = 0
    
    batch_size = 256
    num_batches = (len(test_features) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_features))
            
            batch_features = test_features[start_idx:end_idx].to(device)
            batch_labels = test_labels[start_idx:end_idx].to(device)
            
            outputs = classifier(batch_features)
            _, predicted = torch.max(outputs, 1)
            
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='SimCLR Evaluation')
    
    parser.add_argument('--pretrained_path', required=True, type=str,
                        help='Path to pretrained SimCLR model')
    parser.add_argument('--eval_mode', choices=['knn', 'linear', 'both'], default='both',
                        help='Evaluation mode')
    parser.add_argument('--classifier_path', type=str,
                        help='Path to trained linear classifier (for linear eval)')
    parser.add_argument('--k_values', nargs='+', type=int, 
                        default=[1, 5, 10, 20, 50, 100, 200],
                        help='k values for k-NN evaluation')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='Temperature for k-NN evaluation')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data workers')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Set up device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Set up logging
    logger = setup_logging()
    
    # Load model
    logger.info(f"Loading model from {args.pretrained_path}")
    model = create_simclr_model().to(device)
    
    if os.path.isfile(args.pretrained_path):
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.pretrained_path}")
    
    # Load data
    _, train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # k-NN evaluation
    if args.eval_mode in ['knn', 'both']:
        logger.info("Starting k-NN evaluation...")
        
        # Extract features
        logger.info("Extracting training features...")
        train_features, train_labels = extract_features_and_labels(model, train_loader, device)
        
        logger.info("Extracting test features...")
        test_features, test_labels = extract_features_and_labels(model, test_loader, device)
        
        # Evaluate k-NN
        logger.info("Running k-NN evaluation...")
        knn_results = knn_evaluation_detailed(
            train_features, train_labels, test_features, test_labels,
            k_values=args.k_values, temperature=args.temperature
        )
        
        logger.info("k-NN Results:")
        for k, acc in knn_results.items():
            logger.info(f"  k={k}: {acc:.3f}")
    
    # Linear probe evaluation
    if args.eval_mode in ['linear', 'both']:
        if args.classifier_path is None:
            logger.warning("Linear evaluation requested but no classifier path provided")
        else:
            logger.info("Starting linear probe evaluation...")
            linear_acc = evaluate_linear_probe(args.pretrained_path, args.classifier_path, device)
            logger.info(f"Linear probe accuracy: {linear_acc:.3f}")
    
    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
