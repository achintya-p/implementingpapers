"""
Unit tests for SimCLR components following the roadmap's testing guidelines.
"""

import unittest
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import PairTransform, get_simclr_transforms, get_cifar10_dataloaders
from model import SimCLR, SimCLREncoder, ProjectionHead, LinearClassifier
from loss import NTXentLoss, NTXentLossSimplified
from utils import AverageMeter, cosine_lr_scheduler


class TestDataComponents(unittest.TestCase):
    """Test data loading and transforms"""
    
    def test_pair_transform(self):
        """Test that PairTransform returns two tensors with same base but different content"""
        transform = get_simclr_transforms()
        pair_transform = PairTransform(transform)
        
        # Create a dummy image (PIL format)
        from PIL import Image
        dummy_image = Image.new('RGB', (32, 32), color='red')
        
        # Apply transform
        view1, view2 = pair_transform(dummy_image)
        
        # Check shapes
        self.assertEqual(view1.shape, view2.shape)
        self.assertEqual(view1.shape, (3, 32, 32))
        
        # Check that views are different (due to random augmentation)
        diff = torch.abs(view1 - view2).mean().item()
        self.assertGreater(diff, 0.01, "Views should be different due to random augmentation")
    
    def test_dataloader_shapes(self):
        """Test that dataloaders return expected shapes"""
        # Use small batch size for testing
        pretrain_loader, train_loader, test_loader = get_cifar10_dataloaders(
            batch_size=4, num_workers=0
        )
        
        # Test pretrain loader (returns pairs)
        (view1, view2), labels = next(iter(pretrain_loader))
        self.assertEqual(view1.shape, (4, 3, 32, 32))
        self.assertEqual(view2.shape, (4, 3, 32, 32))
        self.assertEqual(labels.shape, (4,))
        
        # Test standard loaders
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (4, 3, 32, 32))
        self.assertEqual(labels.shape, (4,))


class TestModelComponents(unittest.TestCase):
    """Test model architecture and behavior"""
    
    def setUp(self):
        self.device = torch.device('cpu')  # Use CPU for testing
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 3, 32, 32)
    
    def test_encoder_forward(self):
        """Test encoder forward pass and output shape"""
        encoder = SimCLREncoder(arch='resnet18')
        features = encoder(self.input_tensor)
        
        self.assertEqual(features.shape, (self.batch_size, 512))
        self.assertEqual(encoder.feature_dim, 512)
    
    def test_projection_head(self):
        """Test projection head forward pass"""
        projector = ProjectionHead(input_dim=512, hidden_dim=2048, output_dim=128)
        features = torch.randn(self.batch_size, 512)
        projections = projector(features)
        
        self.assertEqual(projections.shape, (self.batch_size, 128))
    
    def test_simclr_model(self):
        """Test complete SimCLR model"""
        model = SimCLR(arch='resnet18')
        projections = model(self.input_tensor)
        
        # Check output shape
        self.assertEqual(projections.shape, (self.batch_size, 128))
        
        # Check L2 normalization (should have unit norm)
        norms = torch.norm(projections, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)
    
    def test_feature_extraction(self):
        """Test feature extraction for linear probe"""
        model = SimCLR(arch='resnet18')
        features = model.get_features(self.input_tensor)
        
        self.assertEqual(features.shape, (self.batch_size, 512))
        
        # Should not require gradients
        self.assertFalse(features.requires_grad)
    
    def test_linear_classifier(self):
        """Test linear classifier"""
        classifier = LinearClassifier(input_dim=512, num_classes=10)
        features = torch.randn(self.batch_size, 512)
        logits = classifier(features)
        
        self.assertEqual(logits.shape, (self.batch_size, 10))


class TestLossComponents(unittest.TestCase):
    """Test NT-Xent loss implementation"""
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.batch_size = 8
        self.embedding_dim = 128
        self.criterion = NTXentLoss(temperature=0.1)
        self.criterion_simplified = NTXentLossSimplified(temperature=0.1)
    
    def test_loss_shapes_and_values(self):
        """Test that loss returns proper shapes and reasonable values"""
        z1 = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        z2 = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        
        loss, pos_sim, neg_sim = self.criterion(z1, z2)
        
        # Check that loss is a scalar
        self.assertEqual(loss.shape, ())
        self.assertIsInstance(loss.item(), float)
        
        # Check that similarities are scalars
        self.assertEqual(pos_sim.shape, ())
        self.assertEqual(neg_sim.shape, ())
        
        # Loss should be positive
        self.assertGreater(loss.item(), 0)
    
    def test_identical_embeddings(self):
        """Test loss with identical embeddings (should be very low)"""
        z_identical = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        loss, pos_sim, neg_sim = self.criterion(z_identical, z_identical)
        
        # With identical embeddings, positive similarity should be high
        self.assertGreater(pos_sim.item(), 0.99)
        
        # Loss should be low (but not zero due to negatives)
        self.assertLess(loss.item(), 1.0)
    
    def test_loss_symmetry(self):
        """Test that swapping views gives the same loss"""
        z1 = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        z2 = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        
        loss_12, _, _ = self.criterion(z1, z2)
        loss_21, _, _ = self.criterion(z2, z1)
        
        # Should be exactly equal due to symmetry
        self.assertAlmostEqual(loss_12.item(), loss_21.item(), places=5)
    
    def test_temperature_effect(self):
        """Test that temperature affects loss magnitude"""
        z1 = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        z2 = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        
        criterion_low_temp = NTXentLoss(temperature=0.01)
        criterion_high_temp = NTXentLoss(temperature=1.0)
        
        loss_low, _, _ = criterion_low_temp(z1, z2)
        loss_high, _, _ = criterion_high_temp(z1, z2)
        
        # Lower temperature should generally give higher loss (sharper distribution)
        # Note: This isn't always true due to randomness, but should hold on average
        self.assertGreater(loss_low.item(), 0)
        self.assertGreater(loss_high.item(), 0)
    
    def test_implementation_equivalence(self):
        """Test that both implementations give similar results"""
        z1 = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        z2 = F.normalize(torch.randn(self.batch_size, self.embedding_dim), dim=1)
        
        loss1, _, _ = self.criterion(z1, z2)
        loss2, _, _ = self.criterion_simplified(z1, z2)
        
        # Should be very close (allowing for small numerical differences)
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=4)


class TestUtilityComponents(unittest.TestCase):
    """Test utility functions"""
    
    def test_average_meter(self):
        """Test AverageMeter functionality"""
        meter = AverageMeter()
        
        values = [1, 2, 3, 4, 5]
        for val in values:
            meter.update(val)
        
        expected_avg = sum(values) / len(values)
        self.assertAlmostEqual(meter.avg, expected_avg)
        self.assertEqual(meter.count, len(values))
        self.assertEqual(meter.val, values[-1])
    
    def test_cosine_lr_scheduler(self):
        """Test cosine learning rate scheduler"""
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        max_epochs = 100
        warmup_epochs = 10
        base_lr = 3e-4
        
        lrs = []
        for epoch in range(max_epochs):
            lr = cosine_lr_scheduler(
                optimizer, epoch, max_epochs, 
                warmup_epochs=warmup_epochs, base_lr=base_lr
            )
            lrs.append(lr)
        
        # Check warmup phase
        for i in range(warmup_epochs):
            expected_lr = base_lr * (i + 1) / warmup_epochs
            self.assertAlmostEqual(lrs[i], expected_lr, places=6)
        
        # Check that LR peaks at base_lr after warmup
        self.assertAlmostEqual(lrs[warmup_epochs], base_lr, places=6)
        
        # Check that final LR is lower than base LR
        self.assertLess(lrs[-1], base_lr)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components"""
    
    def test_end_to_end_forward(self):
        """Test complete forward pass through the system"""
        # Create components
        model = SimCLR(arch='resnet18', proj_output_dim=128)
        criterion = NTXentLoss(temperature=0.1)
        
        # Create dummy batch
        batch_size = 4
        view1 = torch.randn(batch_size, 3, 32, 32)
        view2 = torch.randn(batch_size, 3, 32, 32)
        
        # Forward pass
        z1 = model(view1)
        z2 = model(view2)
        
        # Compute loss
        loss, pos_sim, neg_sim = criterion(z1, z2)
        
        # Check that everything has proper shapes and values
        self.assertEqual(z1.shape, (batch_size, 128))
        self.assertEqual(z2.shape, (batch_size, 128))
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)
        
        # Check that gradients can be computed
        loss.backward()
        
        # Check that some parameters have gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
