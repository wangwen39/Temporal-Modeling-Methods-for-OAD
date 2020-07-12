# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out = self.net(x)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class model(nn.Module):

    def __init__(self, input_size=4096, num_channels=[4096, 2048, 4096], num_classes=31, kernel_size=2, dropout=0.3):

        super(model, self).__init__()

        layers = []

        num_levels = len(num_channels)

        for i in range(num_levels):

            dilation_size = 2 ** i

            in_channels = input_size if i == 0 else num_channels[i - 1]

            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.classifier = nn.Sequential(nn.Linear(input_size, num_classes), nn.Dropout(p=0.3))

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)  # input:[N, C, L]

    def forward(self, x):

        outputs = self.network(x.transpose(2, 1))  # [16, 4096, 16]
      
        avg_out = self.avgpool(outputs)

        prob = self.classifier(avg_out.squeeze(2))

        return prob




