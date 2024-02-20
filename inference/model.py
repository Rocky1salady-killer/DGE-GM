import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import swin_transformer
import math

class GazeSwinTransformer(nn.Module):
    def __init__(self, img_feature_dim=1024):
        super(GazeSwinTransformer, self).__init__()

        # Load the Swin Transformer as the base model
        self.base_model = swin_transformer.swin_base_patch4_window7_224(
            pretrained=True, 
            num_classes=0, 
            drop_rate=0.0,
            drop_path_rate=0.0
        )

        # Image feature dimension
        self.img_feature_dim = img_feature_dim
        
        # Heads for gaze and bias
        self.gaze_head = nn.Linear(self.img_feature_dim, 2)
        self.bias_head = nn.Linear(self.img_feature_dim, 1)

    def forward(self, x_in):
        features = self.base_model(x_in["face"])

        # Predict gaze and bias
        gaze = self.gaze_head(features)
        gaze_bias = self.bias_head(features)
    
        # Apply activations
        gaze[:,0:1] = math.pi * torch.tanh(gaze[:,0:1])
        gaze[:,1:2] = (math.pi/2) * torch.tanh(gaze[:,1:2])
    
        gaze_bias = math.pi * torch.sigmoid(gaze_bias)

        return gaze, gaze_bias

class PinBallLoss(nn.Module):
    def __init__(self):
        super(PinBallLoss, self).__init__()
        self.q1 = 0.1
        self.q9 = 1 - self.q1

    def forward(self, output_o, target_o, var_o):
        q_10 = target_o - (output_o - var_o)
        q_90 = target_o - (output_o + var_o)

        loss_10 = torch.max(self.q1 * q_10, (self.q1 - 1) * q_10)
        loss_90 = torch.max(self.q9 * q_90, (self.q9 - 1) * q_90)

        loss_10 = torch.mean(loss_10)
        loss_90 = torch.mean(loss_90)

        return loss_10 + loss_90

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.pinball = PinBallLoss()
        
    def forward(self, output_o, target_o, var_o):
        # Calculate the PinBall loss
        pinball_loss = self.pinball(output_o, target_o, var_o)
        
        # Calculate the Mean Squared Error loss
        mse_loss = F.mse_loss(output_o, target_o, reduction='mean')
        
        # Combine the two losses
        combined_loss = pinball_loss + self.alpha * mse_loss
        
        return combined_loss
