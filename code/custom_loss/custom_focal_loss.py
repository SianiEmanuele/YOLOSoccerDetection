import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='none', eps=1e-6): 
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets, bbox_areas=None):
        B, N, C = inputs.shape
        inputs = inputs.view(-1, C)
        targets = targets.view(-1, C)

        probs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_factor * focal_factor * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: # self.reduction == 'none'
            return loss.view(B, N, C) 