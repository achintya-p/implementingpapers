"""
SimCLR model components: encoder f(·) and projector g(·).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimCLREncoder(nn.Module):
    """
    Encoder f(·) based on ResNet-18 without the classifier head.
    """
    
    def __init__(self, arch='resnet18', pretrained=False):
        super().__init__()
        
        # Load backbone
        if arch == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif arch == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Remove the classifier head (fc layer)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
    def forward(self, x):
        # x: [batch_size, 3, 32, 32] for CIFAR-10
        features = self.backbone(x)  # [batch_size, feature_dim, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch_size, feature_dim]
        return features


class ProjectionHead(nn.Module):
    """
    Projection head g(·): MLP with two layers, BatchNorm, and ReLU.
    """
    
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=128):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    """
    Complete SimCLR model combining encoder and projector.
    """
    
    def __init__(self, arch='resnet18', pretrained=False, proj_hidden_dim=2048, proj_output_dim=128):
        super().__init__()
        
        self.encoder = SimCLREncoder(arch=arch, pretrained=pretrained)
        self.projector = ProjectionHead(
            input_dim=self.encoder.feature_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_output_dim
        )
    
    def forward(self, x):
        """
        Forward pass through encoder and projector.
        Returns L2-normalized projected embeddings.
        """
        features = self.encoder(x)  # f(x)
        projections = self.projector(features)  # g(f(x))
        
        # L2 normalize projections
        projections = F.normalize(projections, dim=1)
        
        return projections
    
    def get_features(self, x):
        """
        Get encoder features (for linear probe evaluation).
        """
        with torch.no_grad():
            features = self.encoder(x)
        return features


class LinearClassifier(nn.Module):
    """
    Linear classifier for linear probe evaluation.
    """
    
    def __init__(self, input_dim=512, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


def create_simclr_model(arch='resnet18', pretrained=False, proj_hidden_dim=2048, proj_output_dim=128):
    """
    Factory function to create SimCLR model.
    """
    return SimCLR(
        arch=arch,
        pretrained=pretrained,
        proj_hidden_dim=proj_hidden_dim,
        proj_output_dim=proj_output_dim
    )


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create model
    model = create_simclr_model().to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 3, 32, 32).to(device)
    
    # Test projection output
    projections = model(x)
    print(f"Projection output shape: {projections.shape}")
    print(f"Projection norms (should be ~1): {torch.norm(projections, dim=1)}")
    
    # Test feature extraction
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
    
    # Test linear classifier
    classifier = LinearClassifier(input_dim=model.encoder.feature_dim, num_classes=10).to(device)
    logits = classifier(features)
    print(f"Classifier output shape: {logits.shape}")
    
    print("\nModel test passed!")
