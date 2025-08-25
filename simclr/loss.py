"""
NT-Xent (InfoNCE) loss implementation for SimCLR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy (NT-Xent) Loss.
    
    This is the contrastive loss used in SimCLR that maximizes agreement between
    differently augmented views of the same image via a contrastive task in the latent space.
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Compute NT-Xent loss for a batch of paired embeddings.
        
        Args:
            z1: First view embeddings [batch_size, embedding_dim]
            z2: Second view embeddings [batch_size, embedding_dim]
            
        Returns:
            loss: Scalar loss value
            pos_sim: Mean similarity of positive pairs (for logging)
            neg_sim: Mean similarity of negative pairs (for logging)
        """
        batch_size = z1.shape[0]
        
        # Concatenate both views: [2*batch_size, embedding_dim]
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix: [2*batch_size, 2*batch_size]
        sim_matrix = torch.mm(z, z.T) / self.temperature
        
        # Create mask to identify positive pairs
        # For batch size N, positive pairs are:
        # (0, N), (1, N+1), ..., (N-1, 2N-1), (N, 0), (N+1, 1), ..., (2N-1, N-1)
        mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        for i in range(batch_size):
            mask[i, i + batch_size] = True
            mask[i + batch_size, i] = True
        
        # Create mask to exclude self-similarities (diagonal)
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        
        # Apply numerical stability: subtract max to prevent overflow
        sim_matrix_stable = sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0].detach()
        
        # Compute positive similarities
        pos_sim = sim_matrix_stable[mask]
        
        # Compute denominator: sum of exp(similarities) excluding self-similarities
        exp_sim = torch.exp(sim_matrix_stable)
        exp_sim = exp_sim.masked_fill(self_mask, 0)  # Exclude self-similarities
        denominator = exp_sim.sum(dim=1)
        
        # Compute loss for each positive pair
        # Loss = -log(exp(pos_sim) / denominator)
        numerator = torch.exp(pos_sim)
        # Get denominators for positive pair positions
        pos_indices = torch.where(mask)
        pos_denominators = denominator[pos_indices[0]]
        loss_per_sample = -torch.log(numerator / pos_denominators)
        loss = loss_per_sample.mean()
        
        # Compute statistics for logging
        with torch.no_grad():
            # Use original similarity matrix for statistics (not the stabilized version)
            pos_sim_orig = sim_matrix[mask]
            pos_sim_mean = pos_sim_orig.mean()
            
            # Negative similarities (all except positives and self)
            neg_mask = ~(mask | self_mask)
            neg_sim_mean = sim_matrix[neg_mask].mean()
        
        return loss, pos_sim_mean, neg_sim_mean


class NTXentLossSimplified(nn.Module):
    """
    Alternative implementation using cross-entropy directly.
    Mathematically equivalent but sometimes more numerically stable.
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature  # [2N, 2N]
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])  # [2N]
        
        # Mask out self-similarities (set to very negative value)
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = self.criterion(sim_matrix, labels)
        
        # Compute statistics for logging
        with torch.no_grad():
            # Positive similarities
            pos_indices = torch.arange(2 * batch_size, device=z.device)
            pos_sim = sim_matrix[pos_indices, labels].mean()
            
            # Negative similarities
            neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
            neg_mask[pos_indices, labels] = False
            neg_mask = neg_mask & ~mask
            neg_sim = sim_matrix[neg_mask].mean()
        
        return loss, pos_sim, neg_sim


def test_ntxent_loss():
    """Test NT-Xent loss function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = NTXentLoss(temperature=0.1).to(device)
    criterion_simplified = NTXentLossSimplified(temperature=0.1).to(device)
    
    batch_size = 8
    embedding_dim = 128
    
    # Test 1: Random embeddings
    z1 = F.normalize(torch.randn(batch_size, embedding_dim), dim=1).to(device)
    z2 = F.normalize(torch.randn(batch_size, embedding_dim), dim=1).to(device)
    
    loss1, pos_sim1, neg_sim1 = criterion(z1, z2)
    loss2, pos_sim2, neg_sim2 = criterion_simplified(z1, z2)
    
    print(f"Random embeddings:")
    print(f"  Original loss: {loss1:.4f}, pos_sim: {pos_sim1:.4f}, neg_sim: {neg_sim1:.4f}")
    print(f"  Simplified loss: {loss2:.4f}, pos_sim: {pos_sim2:.4f}, neg_sim: {neg_sim2:.4f}")
    print(f"  Loss difference: {abs(loss1 - loss2):.6f}")
    
    # Test 2: Identical embeddings (should have very low loss)
    z_identical = F.normalize(torch.randn(batch_size, embedding_dim), dim=1).to(device)
    loss_identical, pos_sim_identical, neg_sim_identical = criterion(z_identical, z_identical)
    
    print(f"\nIdentical embeddings:")
    print(f"  Loss: {loss_identical:.4f}, pos_sim: {pos_sim_identical:.4f}, neg_sim: {neg_sim_identical:.4f}")
    
    # Test 3: Symmetry test
    loss_12, _, _ = criterion(z1, z2)
    loss_21, _, _ = criterion(z2, z1)
    
    print(f"\nSymmetry test:")
    print(f"  Loss(z1, z2): {loss_12:.4f}")
    print(f"  Loss(z2, z1): {loss_21:.4f}")
    print(f"  Difference: {abs(loss_12 - loss_21):.6f}")
    
    print("\nNT-Xent loss test completed!")


if __name__ == "__main__":
    test_ntxent_loss()
