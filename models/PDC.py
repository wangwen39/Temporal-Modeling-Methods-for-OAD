# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from multihead import PositionalEncoding
from non_local import PSPModule


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

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        # self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

        # if self.downsample is not None:
        #     self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out = self.net(x)

        return self.relu(out)


class model(nn.Module):

    def __init__(self, input_size=4096, num_classes=31, kernel_size=2, dropout=0.3):

        super(model, self).__init__()

        self.temporal_block_1 = TemporalBlock(n_inputs=input_size, n_outputs=input_size, kernel_size=2, stride=1,
                                     dilation=1, padding=(kernel_size - 1) * 1,
                                     dropout=dropout)

        self.temporal_block_2 = TemporalBlock(n_inputs=input_size, n_outputs=input_size, kernel_size=2, stride=1,
                                              dilation=2, padding=(kernel_size - 1) * 2,
                                              dropout=dropout)

        self.temporal_block_4 = TemporalBlock(n_inputs=input_size, n_outputs=input_size, kernel_size=2, stride=1,
                                              dilation=4, padding=(kernel_size - 1) * 4,
                                              dropout=dropout)

        self.conv1d = nn.Conv1d(in_channels=input_size*3, stride=1, out_channels=input_size, kernel_size=1, padding=0, dilation=1)

        self.classifier = nn.Sequential(nn.Linear(input_size, num_classes), nn.Dropout(p=0.3))

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)  # input:[N, C, L]

    def forward(self, x):

        out_1 = self.temporal_block_1(x.transpose(2, 1))  # [16, 4096, 16]

        out_2 = self.temporal_block_2(x.transpose(2, 1))

        out_4 = self.temporal_block_4(x.transpose(2, 1))

        global_out = torch.cat((out_1, out_2, out_4), 1)

        global_out = self.conv1d(global_out)  # [16, 4096, 16]

        global_out = self.avgpool(global_out)

        prob = self.classifier(global_out.squeeze(2))

        return prob



