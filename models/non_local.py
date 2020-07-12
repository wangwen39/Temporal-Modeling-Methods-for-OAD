import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """Self attention layer for nd."""


    def __init__(self, n_channels: int):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(n_channels, n_channels // 8, kernel_size=1) # 8 is a hyper-parameter
        self.key = nn.Conv1d(n_channels, n_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(n_channels, n_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        size = x.size()

        x = x.view(*size[:2], -1)  # print('x size: ', x.size()) # (16, 64, 4096)

        x = x.transpose(2, 1)
        q, k, v = self.query(x), self.key(x), self.value(x) 
      
        beta = F.softmax(torch.bmm(q.permute(0, 2, 1).contiguous(), k), dim=1) # [16, 16, 16]
        o = self.gamma * torch.bmm(v, beta) + x # Residual connection

        return o.view(*size).contiguous()
