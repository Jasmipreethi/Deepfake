# This file is used to store code for extracting the audio features.
"""
Audio encoder models for AV Deepfake Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleAudioEncoder(nn.Module):
    """Simple CNN for audio spectrogram (original mini pipeline)"""
    def __init__(self, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), 
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        # x: (B, 1, 128, 87)
        return self.encoder(x)


class ImprovedAudioEncoder(nn.Module):
    """Improved 2D CNN for audio with batch normalization"""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.Dropout(0.3),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, 1, 128, 87)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)


class PretrainedAudioEncoder(nn.Module):
    """Pre-trained ResNet18 for audio spectrogram"""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        
        # Change first conv for 1 channel (grayscale spectrogram)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        # x: (B, 1, 128, 87) -> resize to 224x224 for ResNet
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)


def get_audio_encoder(encoder_type='pretrained', feature_dim=256):
    """
    Factory function to get audio encoder by type
    
    Args:
        encoder_type: 'simple', 'improved', or 'pretrained'
        feature_dim: output feature dimension
    
    Returns:
        Audio encoder instance
    """
    if encoder_type == 'simple':
        return SimpleAudioEncoder(output_dim=feature_dim)
    elif encoder_type == 'improved':
        return ImprovedAudioEncoder(feature_dim=feature_dim)
    elif encoder_type == 'pretrained':
        return PretrainedAudioEncoder(feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
