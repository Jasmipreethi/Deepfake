# This file is used to extract the video features into a single file
"""
Video encoder models for AV Deepfake Detection
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class SimpleVideoEncoder(nn.Module):
    """Simple CNN for video frames (original mini pipeline)"""
    def __init__(self, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), 
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        # x: (B, C, H, W) - processes single frames
        return self.encoder(x)


class ImprovedVideoEncoder(nn.Module):
    """Improved 3D CNN for video with batch normalization"""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 4, 4))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.Dropout(0.3),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)


class PretrainedVideoEncoder(nn.Module):
    """Pre-trained ResNet3D-18 for video"""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        # x: (B, T, C, H, W) = (B, 50, 3, 224, 224)
        # ResNet3D expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)


class SimpleVideoProcessor(nn.Module):
    """
    Wrapper for SimpleVideoEncoder to handle temporal dimension
    Processes each frame independently and averages
    """
    def __init__(self, output_dim=64):
        super().__init__()
        self.frame_encoder = SimpleVideoEncoder(output_dim=output_dim)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t = x.shape[:2]
        # Encode each frame
        frame_features = []
        for i in range(t):
            frame_features.append(self.frame_encoder(x[:, i]))
        # Stack and average
        frame_features = torch.stack(frame_features, dim=1)  # (B, T, D)
        return frame_features.mean(dim=1)  # (B, D)


def get_video_encoder(encoder_type='pretrained', feature_dim=256):
    """
    Factory function to get video encoder by type
    
    Args:
        encoder_type: 'simple', 'improved', or 'pretrained'
        feature_dim: output feature dimension
    
    Returns:
        Video encoder instance
    """
    if encoder_type == 'simple':
        return SimpleVideoProcessor(output_dim=feature_dim)
    elif encoder_type == 'improved':
        return ImprovedVideoEncoder(feature_dim=feature_dim)
    elif encoder_type == 'pretrained':
        return PretrainedVideoEncoder(feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")