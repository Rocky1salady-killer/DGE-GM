import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import swin_transformer
import math

class GazeSwinTransformer(nn.Module):
    def __init__(self, img_feature_dim=1024, dropout_rate=0.5):
        super(GazeSwinTransformer, self).__init__()

        
        self.base_model = swin_transformer.swin_base_patch4_window7_224(
            pretrained=True, 
            num_classes=0, 
            drop_rate=0.0,
            drop_path_rate=0.0
        )

        
        self.img_feature_dim = img_feature_dim
        
        
        self.bn1 = nn.BatchNorm1d(self.img_feature_dim)

        
        self.dropout = nn.Dropout(p=dropout_rate)

       
        self.gaze_head = nn.Linear(self.img_feature_dim, 2)
        self.bias_head = nn.Linear(self.img_feature_dim, 1)
        self.dense = nn.Linear(self.img_feature_dim, self.img_feature_dim)

    def forward(self, x_in):
        features = self.base_model(x_in["face"])

        
        features = F.leaky_relu(self.bn1(features))

        
        dense_output = F.leaky_relu(self.dense(self.dropout(features)))
        gaze = self.gaze_head(dense_output)
        gaze_bias = self.bias_head(dense_output)

        
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
    def __init__(self, alpha=0.8, beta=0.1):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pinball = PinBallLoss()
        
    def forward(self, output_o, target_o, var_o):
        
        pinball_loss = self.pinball(output_o, target_o, var_o)
        
        
        mse_loss = F.mse_loss(output_o, target_o, reduction='mean')
        
        
        
        bias_penalty = self.beta * torch.abs(output_o - var_o).mean()
        combined_loss = pinball_loss + self.alpha * mse_loss + bias_penalty

        
        return combined_loss
