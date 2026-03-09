# This model is used to align the audio and video components and generate the predictive model.
"""
Cross-modal fusion models for AV Deepfake Detection
Combines audio and video features for final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFusion(nn.Module):
    """Simple fusion for mini model (original pipeline)"""
    def __init__(self, feature_dim=64, hidden_dim=32):
        super().__init__()
        self.fusion = nn.Linear(feature_dim * 2, hidden_dim)
        self.audio_head = nn.Linear(hidden_dim, 1)
        self.video_head = nn.Linear(hidden_dim, 1)

    def forward(self, video_feat, audio_feat):
        # Simple concatenation and linear fusion
        combined = torch.cat([video_feat, audio_feat], dim=1)
        fused = torch.relu(self.fusion(combined))
        
        return {
            'audio_pred': self.audio_head(fused),
            'video_pred': self.video_head(fused),
            'fused': fused
        }


class ImprovedFusion(nn.Module):
    """Improved fusion with batch norm, dropout, and joint prediction"""
    def __init__(self, feature_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        self.audio_classifier = nn.Linear(hidden_dim, 1)
        self.video_classifier = nn.Linear(hidden_dim, 1)
        self.joint_classifier = nn.Linear(hidden_dim, 1)
        
        self.temperature = 0.1  # Temperature for sigmoid scaling

    def forward(self, video_feat, audio_feat):
        combined = torch.cat([video_feat, audio_feat], dim=1)
        fused = self.fusion(combined)

        # Temperature-scaled predictions
        audio_pred = torch.sigmoid(
            self.audio_classifier(fused) / self.temperature
        )
        video_pred = torch.sigmoid(
            self.video_classifier(fused) / self.temperature
        )
        joint_pred = torch.sigmoid(
            self.joint_classifier(fused) / self.temperature
        )

        return {
            'audio_pred': audio_pred,
            'video_pred': video_pred,
            'joint_pred': joint_pred,
            'fused': fused
        }


class PretrainedFusion(nn.Module):
    """Fusion module for pretrained encoders"""
    def __init__(self, feature_dim=256, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        self.audio_classifier = nn.Linear(hidden_dim, 1)
        self.video_classifier = nn.Linear(hidden_dim, 1)
        self.joint_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, video_feat, audio_feat):
        combined = torch.cat([video_feat, audio_feat], dim=1)
        fused = self.fusion(combined)
        
        return {
            'audio_pred': torch.sigmoid(self.audio_classifier(fused)),
            'video_pred': torch.sigmoid(self.video_classifier(fused)),
            'joint_pred': torch.sigmoid(self.joint_classifier(fused)),
            'fused': fused
        }


class AttentionFusion(nn.Module):
    """
    Cross-modal attention fusion
    (Optional advanced fusion mechanism)
    """
    def __init__(self, feature_dim=256, hidden_dim=512, num_heads=8, dropout=0.4):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Cross-attention
        self.audio_to_video = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.video_to_audio = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
        self.audio_classifier = nn.Linear(hidden_dim, 1)
        self.video_classifier = nn.Linear(hidden_dim, 1)
        self.joint_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, video_feat, audio_feat):
        # Add sequence dimension for attention
        v = video_feat.unsqueeze(1)  # (B, 1, D)
        a = audio_feat.unsqueeze(1)  # (B, 1, D)
        
        # Cross-attention
        v_attended, _ = self.audio_to_video(v, a, a)
        a_attended, _ = self.video_to_audio(a, v, v)
        
        # Remove sequence dimension and concatenate
        v_out = v_attended.squeeze(1)
        a_out = a_attended.squeeze(1)
        
        combined = torch.cat([v_out, a_out], dim=1)
        fused = self.fusion(combined)
        
        return {
            'audio_pred': torch.sigmoid(self.audio_classifier(fused)),
            'video_pred': torch.sigmoid(self.video_classifier(fused)),
            'joint_pred': torch.sigmoid(self.joint_classifier(fused)),
            'fused': fused
        }

class TransformerFusion(nn.Module):
    """
    Transformer-based cross-modal fusion (GPU-optimized).
    
    Uses a transformer encoder with a learnable [CLS] token to fuse
    audio and video embeddings. More powerful than MLP fusion but
    requires more compute — best suited for GPU.
    """
    def __init__(self, feature_dim=256, hidden_dim=512, 
                 num_heads=8, num_layers=2, dropout=0.4):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Project both modalities to same dimension
        self.audio_proj = nn.Linear(feature_dim, hidden_dim)
        self.video_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Learnable [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional embeddings: [CLS], video, audio
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )
        
        # Classification heads from the [CLS] token output
        self.audio_classifier = nn.Linear(hidden_dim, 1)
        self.video_classifier = nn.Linear(hidden_dim, 1)
        self.joint_classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, video_feat, audio_feat):
        B = video_feat.shape[0]
        
        # Project to hidden dimension
        v = self.video_proj(video_feat).unsqueeze(1)  # (B, 1, H)
        a = self.audio_proj(audio_feat).unsqueeze(1)   # (B, 1, H)
        
        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        
        # Sequence: [CLS, video, audio] + positional embeddings
        tokens = torch.cat([cls, v, a], dim=1)  # (B, 3, H)
        tokens = tokens + self.pos_embedding
        
        # Transformer self-attention across modalities
        fused = self.transformer(tokens)  # (B, 3, H)
        
        # Use [CLS] token output for classification
        cls_out = fused[:, 0, :]  # (B, H)
        
        return {
            'audio_pred': torch.sigmoid(self.audio_classifier(cls_out)),
            'video_pred': torch.sigmoid(self.video_classifier(cls_out)),
            'joint_pred': torch.sigmoid(self.joint_classifier(cls_out)),
            'fused': cls_out
        }


def get_fusion_module(fusion_type='pretrained', feature_dim=256, hidden_dim=512, dropout=0.4):
    """
    Factory function to get fusion module by type
    
    Args:
        fusion_type: 'simple', 'improved', 'pretrained', 'attention', or 'transformer'
        feature_dim: input feature dimension from encoders
        hidden_dim: hidden dimension for fusion
        dropout: dropout rate
    
    Returns:
        Fusion module instance
    """
    if fusion_type == 'simple':
        return SimpleFusion(feature_dim=feature_dim, hidden_dim=hidden_dim//2)
    elif fusion_type == 'improved':
        return ImprovedFusion(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif fusion_type == 'pretrained':
        return PretrainedFusion(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif fusion_type == 'attention':
        return AttentionFusion(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif fusion_type == 'transformer':
        return TransformerFusion(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

