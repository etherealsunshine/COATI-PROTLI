import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self,in_dim,out_dim=512):
        super().__init__()
        self.fc =nn.Linear(in_dim,out_dim)
        self.ln=nn.LayerNorm(out_dim)

    def forward(self,x):
        x = self.fc(x)
        x = self.ln(x)
        x = F.normalize(x,dim=-1)
        return x